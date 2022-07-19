"""
Recipe for singer gender identification task.

To run this recipe, do the following:
> python train.py train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

"""
import os
import sys

import numpy as np
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group
from speechbrain.utils.data_pipeline import provides
from speechbrain.dataio import dataio, dataset
from speechbrain.nnet import schedulers, losses
from deezer.audio.signal import SignalFactory
from prepare_data import prepare_json_files
from utils import *


# Brain class for speech enhancement training
class gender_rec_Brain(sb.Brain):
    # Class Brain is in the core.py
    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        My note: this function computes on the batch of data that I pass

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions, features are spectrograms, not mfcc
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats, lens)
        # embeddings.shape = torch.Size([32, 1, 512])

        if stage == sb.Stage.TEST:
            # save embedding for visualization
            self.embeddings = torch.cat([self.embeddings, embeddings.squeeze()])
            # save metadata also for visualization
            gender_id = np.vstack((batch.gender_encoded.data.squeeze().cpu().detach().numpy(), np.array(batch.id))).T
            import pdbr;pdbr.set_trace()
            for item in gender_id.tolist():
                self.metadata.append(item)
        predictions = self.modules.classifier(embeddings)
        # predictions.shape = torch.Size([32, 1, 28]) because of 28 singers

        return predictions

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                # if this module has attribute env_corrupt, original waves and
                # augmented ones are concatenated together
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        # we can pass modules that we want on this object
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        return feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for back-propagating the gradient.
        """
        _, lens = batch.sig
        gender, _ = batch.gender_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            gender = torch.cat([gender, gender], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function
        loss = sb.nnet.losses.nll_loss(predictions, gender, lens)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, gender, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, gender, lens)
            # Print predictions and gender to see why always 34.9%

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up trackers of output of the embedding model
        if stage == sb.Stage.TEST:
            self.embeddings = torch.Tensor().to(run_opts["device"])

            import numpy as np
            self.metadata = []
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            self.hparams.tensorboard_train_logger.log_stats(
                stats_meta={"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

            self.hparams.tensorboard_train_logger.writer.add_embedding(
                self.embeddings,
                metadata=self.metadata,
                metadata_header=["gender", "song_id"],
                global_step=self.hparams.epoch_counter.current
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "start", "end")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, end):
        """
        Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        It resamples the signal if the sampling rate is not the same as in the hyperparameter file.
        """
        signal = SignalFactory().load(
            wav, sampling_frequency=hparams["sample_rate"], offset=start, duration=end-start
        ).to_mono(inplace=True)
        return signal.data.squeeze()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("gender")
    @sb.utils.data_pipeline.provides("gender", "gender_encoded")
    def label_pipeline(gender):
        yield gender
        gender_encoded = label_encoder.encode_label_torch(gender)
        yield gender_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = True
    # we sort the dataset based on length to speed-up training because there will be less padding
    # It can also to shuffle the dataset
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "gender_encoded"],
        ).filtered_sorted(sort_key="length")

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        # TODO change the output key
        output_key="gender",
    )
    return datasets


@profile(filename="profile.ps")
def fit_func():
    gender_rec_brain.fit(
        epoch_counter=gender_rec_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    run_opts["device"] = set_gpus()
    run_opts["debug"] = False

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # This function will download files needed for augmentation and put them under ./data
    # corresponding function is here: speechbrain.lobes.augment.EnvCorrupt

    # Create experiment directory
    # This function puts train.py, train.yaml, env and log in a subdirectory that has the same name as seed
    # which is defined in train.yaml: output_folder: !ref ./results/speaker_id/<seed>
    # All the experiment will happen there
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from utils import MyTensorboardLogger as TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    # Data preparation, to be run on only one process.
    # This will download
    sb.utils.distributed.run_on_main(
        prepare_json_files,
        kwargs={
            "training_path": hparams["training_data"],
            "original_folder": hparams["original_folder"],
            "voice_folder": hparams["voice_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": [90, 10],
            "golden_path": hparams["golden_data"]
            # "min_n_track": hparams["min_n_track"],
            # "max_n_track": hparams["max_n_track"]
        },
    )

    # Create dataset objects "train", "valid", and "test".
    # Load dataset in objects
    datasets = dataio_prep(hparams)

    # Fetch and load pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    gender_rec_brain = gender_rec_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    import pdbr;pdbr.set_trace()

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    # Speechbrain.utils.checkpoints - Would load a checkpoint here, but none found yet.
    # fit() function is from Brain class, I can pass dataloader shuffle in train_loader_kwargs, which is saved in
    # train.yaml
    # fit_func()

    # Load the best checkpoint for evaluation
    test_stats = gender_rec_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )