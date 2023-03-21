"""
This file contains some functions for data preparing and tensorboard loggers.
"""

from speechbrain.utils.train_logger import TrainLogger
import pydub
import numpy as np


def choose_track_number(df, n_min, n_max=10000):
    """
    This function chooses rows with singers that have more than n tracks that have more than 3s.
    Parameters. I also have to drop artist with the id 92664640 because this guy is a bass player for
    the Voice. No matter how many tracks he has, he's only a bass player whose role is bass(vocal) in
    the database!!!!!!!!!!!
    ----------
    df: pandas.Dataframe
        all raw information.
    n: int
        To keep rows with singers that have more than n tracks

    Returns
    -------
    df: pandas.Dataframe
    """
    df = df.drop_duplicates()
    df = df.drop(df[df.unique_artist_id == "92664640"].index)
    df = df[df["lyrics_end"]-df["lyrics_start"] > 3]
    g = df.groupby("unique_artist_id")
    df = g.filter(lambda x: n_min < len(x) < n_max)

    return df
def choose_artist_id(df, ids):
    """
    Choose dataset based on artists' unique IDs.
    Parameters
    ----------
    df: pandas.Dataframe
    ids: list
        List of artists' unique IDs whose tracks we want to keep.

    Returns
    -------
    df: pandas.Dataframe
    """
    df = df.drop_duplicates()
    df = df[df["unique_artist_id"].isin(ids)]
    return df


class MyTensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs.

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Raises ImportError if TensorBoard is not installed

        from torch.utils.tensorboard import SummaryWriter
        import tensorflow as tf
        import tensorboard as tb

        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

        self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "test": {}, "meta": 0}

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        self.global_step["meta"] += 1
        for name, value in stats_meta.items():
            self.writer.add_scalar(name, value, self.global_step["meta"])

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"

                # Both single value (per Epoch) and list (Per batch) logging is supported
                if isinstance(value_list, list):
                    for value in value_list:
                        new_global_step = self.global_step[dataset][stat] + 1
                        self.writer.add_scalar(tag, value, new_global_step)
                        self.global_step[dataset][stat] = new_global_step
                else:
                    value = value_list
                    new_global_step = self.global_step[dataset][stat] + 1
                    self.writer.add_scalar(tag, value, new_global_step)
                    self.global_step[dataset][stat] = new_global_step

    def log_audio(self, name, value, sample_rate):
        """Add audio signal in the logs."""
        self.writer.add_audio(
            name, value, self.global_step["meta"], sample_rate=sample_rate
        )
    #
    # def log_figure(self, name, value):
    #     """Add a figure in the logs."""
    #     fig = plot_spectrogram(value)
    #     if fig is not None:
    #         self.writer.add_figure(name, fig, self.global_step["meta"]

def set_gpus():
    import logging
    import GPUtil
    import torch
    logging.info("TORCH version: {}".format(torch.__version__))
    gpu_index = GPUtil.getAvailable(limit=4, maxMemory=0.1)
    # setting gpu for tensorflow
    try:
        gpu_index = [gpu_index[0]]
    except Exception as e:
        raise ValueError("No GPU available!!")
    logging.info("\t Using GPUs: {}".format(gpu_index))
    device = "cuda:{}".format(",".join([str(i) for i in gpu_index]))
    return device


def read(f, start, duration, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_wav(f, start, duration)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.mean(axis=1)
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y



def profile(filename: str = ''):
    '''
    A decorator that uses cProfile to profile a function.
    If :param: filename is used, a pstats.Stats object will
    be stored in a file under the same name. Otherwise the
    profiling will be printed to standard output.
    NOTE: A function which uses this profiling MUST NOT have
          a keyword argument called 'filename'.

    :param filename: string path to file where pstats.Stats will be saved
    '''
    import cProfile
    import pstats

    def wrap(func):
        def wrapped_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            sortby = 'cumulative'
            ps = pstats.Stats(pr).sort_stats(sortby)
            if filename != '': ps.dump_stats(filename)
            else: print(ps.print_stats())
            return retval
        return wrapped_f
    return wrap