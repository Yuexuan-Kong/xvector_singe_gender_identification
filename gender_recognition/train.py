"""
Recipe for singer gender identification task.

To run this recipe, do the following:
> python train.py train_1998.yaml

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
from typing import Dict

import numpy as np
import wandb
import torch
import torchaudio
import speechbrain as sb

import utils
from utils import *
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group
from speechbrain.utils.data_pipeline import provides
from speechbrain.dataio import dataio, dataset
from speechbrain.nnet import schedulers, losses
from prepare_data import prepare_json_files, random
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader



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
            self.embeddings = torch.cat([self.embeddings, embeddings.squeeze(1)])
            # save metadata also for visualization
            gender_id = np.vstack((batch.gender_encoded.data.squeeze().cpu().detach().numpy(), np.array(batch.id))).T
            for item in gender_id.tolist():
                self.metadata.append(item)
        predictions = self.modules.classifier(embeddings)

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
            # if hasattr(self.modules, "env_corrupt"):
            #     # if this module has attribute env_corrupt, original waves and
            #     # augmented ones are concatenated together
            #     wavs_noise = self.modules.env_corrupt(wavs, lens)
            #     wavs = torch.cat([wavs, wavs_noise], dim=0)
            #     lens = torch.cat([lens, lens])

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

        # # Concatenate labels (due to data augmentation)
        # if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
        #     gender = torch.cat([gender, gender], dim=0)
        #     lens = torch.cat([lens, lens])

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
            wandb.log({"training_loss": stage_loss})

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }
            wandb.log({'validation_error': stats['error']})

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

            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_valid_logger.log_stats(
                    stats_meta={"Epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stats,
                )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

            wandb.log({
                'lr': new_lr,
                'validation_loss': stage_loss,
            })

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST :

            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_test_logger.writer.add_embedding(
                    self.embeddings,
                    metadata=self.metadata,
                    metadata_header=["gender", "song_id"],
                    global_step=self.hparams.epoch_counter.current
                )
            wandb.log({
                'test_loss': stage_loss,
            })
        torch.cuda.empty_cache()

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        On top of the default evaluate function, I added a bloc of code to save the
        wrong predictions.

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(test_set, DataLoader)
                or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_loader_kwargs["shuffle"] = False
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        # Now we iterate over the dataset and we simply compute_forward and decode
        wrong_ids = []
        avg_test_loss = 0.0

        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1

                # calculate preditions from probablities and wrong predictions
                probabilities = self.compute_forward(batch, stage=sb.Stage.TEST)
                prediction_batch = torch.argmax(probabilities, dim=-1).squeeze().cpu().detach().numpy()
                label = batch.gender_encoded.data.squeeze().cpu().detach().numpy()
                id = np.array(batch.id)

                # ID of wrong predictions
                wrong_id = id[prediction_batch!=label]
                wrong_ids.extend(wrong_id.tolist())

                # Calculate loss of the test dataset
                loss = self.compute_objectives(probabilities, batch, stage=sb.Stage.TEST).detach().cpu()
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            from speechbrain.utils.distributed import run_on_main
            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[sb.Stage.TEST, avg_test_loss, None]
            )
        self.step = 0

        # save wrong ids to a local pickle file
        wandb.log({"wrong_ids": wrong_ids})
        import pickle
        with open('wrong_predictions.pkl', 'wb') as f:
            pickle.dump(wrong_ids, f)
        print(len(wrong_ids), avg_test_loss)
        return wrong_ids, avg_test_loss
    # TODO: change inference dataset to json file, simplify the inference process
    
    def inference(
            self,
            dataset,  # Must be obtained from the dataio_function
            min_key,  # We load the model with the lowest key
            loader_kwargs  # opts for the dataloading
    ):

        self.on_evaluate_start(min_key)
        self.on_stage_start(sb.Stage.TEST)  # We call the on_evaluate_start that will load the best model
        self.modules.eval()  # We set the model to eval mode (remove dropout etc)

        if not (
            isinstance(dataset, DataLoader)
            or isinstance(dataset, LoopedLoader)
        ):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            from tqdm import tqdm
            predictions = []
            ids = []
            for batch in tqdm(dataset, dynamic_ncols=True):
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                probabilities = self.compute_forward(batch, stage=sb.Stage.TEST)
                # Difference between probabilities define the uncertainty
                prediction_batch = torch.argmax(probabilities, dim=-1).squeeze().cpu().detach().numpy()
                label = batch.gender_encoded.data.squeeze().cpu().detach().numpy()
                id = np.array(batch.id)

                predictions.append(prediction_batch.tolist())
                ids.append(id.tolist())

        return ids, predictions


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train_1998.yaml` file, and it includes
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
        sig, read_sr = torchaudio.load(wav, frame_offset=int(16000*start), num_frames=int(16000*3))

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("gender")
    @sb.utils.data_pipeline.provides("gender", "gender_encoded")
    def label_pipeline(gender):
        yield gender
        gender_encoded = label_encoder.encode_label_torch(gender)
        yield gender_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    global datasets
    datasets = {}
    # we sort the dataset based on length to speed-up training because there will be less padding
    # It can also to shuffle the dataset
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "gender_encoded"],
        )

    # for dataset in ["subset_0.json", "subset_1.json", "subset_2.json", "subset_3.json", "subset_4.json"]:
    #     datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
    #         json_path=hparams["test_bootstrap"]+dataset,
    #         replacements={"data_root": hparams["data_folder"]},
    #         dynamic_items=[audio_pipeline, label_pipeline],
    #         output_keys=["id", "sig", "gender_encoded"],
    #     )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="gender",
    )
    return datasets


# @profile(filename="profile.ps")
def fit_func(hparams):
    gender_rec_brain.fit(
        epoch_counter=gender_rec_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

def main():
    run = wandb.init(project='ISMIR-2023')
    # Reading command line arguments.
    global run_opts  # Makes run_opts global
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    run_opts["debug"] = False

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides. Have to change directory before it, because it calls
    # downloading if data we need is not in the right directory

    # Note to baobao: this is where hyperparameters from sweep are made visible
    # to the existing code in this script
    overrides = {
        'seed': random.randint(0, 10000),
        'classifier': {'dropout': wandb.config.classifier_dropout},
        'embedding_model': {'final_dropout': wandb.config.embedding_dropout},
        'batch_size': wandb.config.batch_size,
        'lr_start': wandb.config.lr_start,
        'lr_final': wandb.config.lr_final,
        'emb_dim': wandb.config.emb_dim,
        'train_annotation': wandb.config.training_data
    }
    print(f"Setting seed to: {overrides['seed']}")
    wandb.log({'seed': overrides['seed']})

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # change work dir to the parent folder
    run_opts["device"] = set_gpus()
    # run_opts["device"] = "cpu"
    # run_opts["debug"] = True
    hparams["dataloader_options"]["shuffle"] = True

    # This function will download files needed for augmentation and put them under ./data
    # corresponding function is here: speechbrain.lobes.augment.EnvCorrupt
    # Create experiment directory
    # This function puts train.py, train_1998.yaml, env and log in a subdirectory that has the same name as seed
    # which is defined in train_1998.yaml: output_folder: !ref ./results/speaker_id/<seed>
    # All the experiment will happen there
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from utils import MyTensorboardLogger as TensorboardLogger

        hparams["tensorboard_valid_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_valid_folder"]
        )

        hparams["tensorboard_test_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_test_folder"]
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
        },
    )

    # # Create dataset objects "train", "valid", and "test".
    # Load dataset in objects
    datasets = dataio_prep(hparams)
    # Fetch and load pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize the Brain object to prepare for mask training.
    global gender_rec_brain
    gender_rec_brain = gender_rec_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    wandb.watch(gender_rec_brain.modules.embedding_model)
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    # Speechbrain.utils.checkpoints - Would load a checkpoint here, but none found yet.
    # fit() function is from Brain class, I can pass dataloader shuffle in train_loader_kwargs, which is saved in
    # train_1998.yaml
    fit_func(hparams)

    # Load the best checkpoint for evaluation
    test_stats = gender_rec_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

# Recipe begins!
if __name__ == "__main__":
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            # Note for baobao:
            # The name of this variable MUST match the name of a key going into 'wandb.log'
            'name': 'validation_error'
        },
        'parameters': {
            'batch_size': {'values': [16]},
            'lr_start': {'values': [0.005]},
            'lr_final': {'values': [0.001]},
            'emb_dim': {'values': [64]},
            'classifier_dropout': {'values': [0.3]},
            'embedding_dropout': {'values': [0.0]},
            'training_data': {'values': ['train.json']}
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='ISMIR-2023')
    print(f'Starting wandb run for sweep_id: {sweep_id}')
    wandb.agent(sweep_id, function=main, count=5)
