#!/usr/bin/env python3
"""Recipe for training a speaker-id system. The template can use used as a
basic example for any signal classification task such as language_id,
emotion recognition, command classification, etc. The proposed task classifies
28 speakers using Mini Librispeech. This task is very easy. In a real
scenario, you need to use datasets with a larger number of speakers such as
the voxceleb one (see recipes/VoxCeleb). Speechbrain has already some built-in
models for signal classifications (see the ECAPA one in
speechbrain.lobes.models.ECAPA_TDNN.py or the xvector in
speechbrain/lobes/models/Xvector.py)

To run this recipe, do the following:
> python train.py train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Mirco Ravanelli 2021
"""
import os
import sys
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
class singer_idBrain(sb.Brain):
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
        # import pdbr;pdbr.set_trace()
        embeddings = self.modules.embedding_model(feats, lens)
        # embeddings.shape = torch.Size([32, 1, 512])

        # save embedding for visualization
        self.embeddings = torch.cat([self.embeddings, embeddings.squeeze()])

        # save metadata also for visualization
        self.metadata = torch.cat([self.metadata, batch.singer_id_encoded.data.squeeze()])

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
        singer_id, _ = batch.singer_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            singer_id = torch.cat([singer_id, singer_id], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function
        loss = sb.nnet.losses.nll_loss(predictions, singer_id, lens)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, singer_id, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, singer_id, lens)
            # Print predictions and singer_id to see why always 34.9%
            # import pdbr;pdbr.set_trace()

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
        self.embeddings = torch.Tensor().to(run_opts["device"])
        self.metadata = torch.Tensor().to(run_opts["device"])

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
                train_stats=self.train_stats,
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
                global_step=self.hparams.epoch_counter.current  # TODO
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
    @sb.utils.data_pipeline.takes("singer_id")
    @sb.utils.data_pipeline.provides("singer_id", "singer_id_encoded")
    def label_pipeline(singer_id):
        yield singer_id
        singer_id_encoded = label_encoder.encode_label_torch(singer_id)
        yield singer_id_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    # we sort the dataset based on length to speed-up training because there will be less padding
    # It can also to shuffle the dataset
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "singer_id_encoded"],
        ).filtered_sorted(sort_key="length")

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="singer_id",
    )
    return datasets


@profile(filename="profile.ps")
def fit_func():
    singer_id_brain.fit(
        epoch_counter=singer_id_brain.hparams.epoch_counter,
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
            "split_ratio": [80, 10, 10],
            "min_n_track": hparams["min_n_track"],
            "max_n_track": hparams["max_n_track"]
        },
    )

    # Create dataset objects "train", "valid", and "test".
    # Load dataset in objects
    datasets = dataio_prep(hparams)

    # Fetch and load pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    singer_id_brain = singer_idBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    # Speechbrain.utils.checkpoints - Would load a checkpoint here, but none found yet.
    # fit() function is from Brain class, I can pass dataloader shuffle in train_loader_kwargs, which is saved in
    # train.yaml
    # fit_func()

    # Load the best checkpoint for evaluation
    test_stats = singer_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )