"""
Downloads and creates data manifest files for Mini LibriSpeech (spk-id).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.

Authors:
 * Mirco Ravanelli, 2021
"""

import os
import ujson as json
import shutil
import random
import logging
# from speechbrain.utils.data_utils import get_all_files, download_file
# from speechbrain.dataio.dataio import read_audio
import numpy as np

from path import *
import pandas as pd
from Audio import Audio
import path
from utils import *
from tqdm import tqdm

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_json_files(
    training_path,
    original_folder,
    voice_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    min_n_track,
    max_n_track,
    split_ratio=[80, 10, 10],

):
    """
    Prepares the json files for training x-vectors.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    training_path : str
        Path to the file where all information of training data. Default to path.TRAIN
    original_folder : str
        Path to the folder where origined cropped tracks are saved.
    voice_folder : str
        Path to the folder where only voices of cropped tracks are saved.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = path.JSON
    >>> prepare_json_files(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    df = pd.read_pickle(training_path)
    df = choose_track_number(df, min_n_track, max_n_track)

    # These two are Justin Bieber and Taylor Swift
    # df = choose_artist_id(df, ["288166e0140a67-e4d1-4f13-8a01-364355bee46e",
    #                            "1224620244d07-534f-4eff-b4d4-930878889970"])

    # Random split the signal list track-wise into train, valid, and test sets.
    data_split = split_sets(df, split_ratio)

    # Creating json files
    func_create_json(data_split, original_folder, save_json_train, voice_folder)
    # TODO in the future, we don't add spleeter for validation and test
    create_json(data_split["valid"], save_json_valid, True, original_folder, voice_folder)
    create_json(data_split["test"], save_json_test, True, original_folder, voice_folder)


@profile(filename="profile_create_json.ps")
def func_create_json(data_split, original_folder, save_json_train, voice_folder):
    create_json(data_split["train"], save_json_train, True, original_folder, voice_folder)


def create_json(df, json_file, add_spleeter, original_folder, voice_folder):
    """
    Creates the json file given a dataframe of track's and singer's information. For female singers, tracks sample two
    time than normal.

    Arguments
    ---------
    df : pandas.Dataframe
        Dataframe of gender, lyrics_start, lyrics_end and song_id
    json_file : str
        The path of the output json file
    add_spleeter : bool
        If add the same track but separated voice.
    original_folder : str
        Path to the folder where origined cropped tracks are saved.
    voice_folder : str
        Path to the folder where only voices of cropped tracks are saved.
    """
    # Processing all the wav files in the list
    json_dict = {}
    for s, row in tqdm(df.iterrows(), total=df.shape[0]):

        # Use Audio class to take care of audio processing
        # Reading the signal (to retrieve duration in seconds)
        gender = 1 if np.isnan(row["gender"]) else row["gender"]
        # audio_mixed = Audio(song_id, type="mixed", lyrics_start=row["lyrics_start"], lyrics_end=row["lyrics_end"])
        duration = row["lyrics_end"]-row["lyrics_start"]

        # Create entry for this utterance. For each track with duration x seconds, we sample it
        # x/6 times. ex. A track of 12s, we sample two times from it. Each time, the utterance length
        # is a random number between 3 and x seconds. For the same example, the duration of two samples
        # vary from 3s to 12s. If the singer is female, then we sample it x/3 times.
        for i in range(0, int(round(duration/6)*gender)):

            # Random select segment that are larger than 3s and 10s
            path = original_folder+str(row["SNG_ID"])+".wav"
            start = round(random.uniform(0, duration-3), 1)
            end = round(random.uniform(max(3, start+3), min(start+10, duration)), 1)

            json_dict[str(row["SNG_ID"])+f"_origin_{i+1}"] = {
                "wav": path,
                "start": start,
                "end": end,
                "length": round(end-start, 1),
                "singer_id": row["unique_artist_id"],
            }
            if add_spleeter:
                start = round(random.uniform(0, duration-3), 1)
                end = round(random.uniform(max(3, start+3), min(start+10, duration)), 1)
                path = voice_folder+str(row["SNG_ID"]) + "/" +"vocals.wav"

                json_dict[str(row["SNG_ID"])+f"_voice_{i+1}"] = {
                    "wav": path,
                    "start": start,
                    "end": end,
                    "length": round(end-start, 1),
                    "singer_id": row["unique_artist_id"],
                }
        # Writing the dictionary to the json file
        with open(json_file, mode="w") as json_f:
            json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_sets(df, split_ratio):
    """Split the Dataframe of all raw information so that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.).

    Arguments
    ---------
    df: pandas.Dataframe
        Dataframe of all the information about tracks used for training.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    ------
    Dictionary of Pandas Dataframe containing train, valid, and test splits.
    """
    train = df.groupby('unique_artist_id').sample(frac=split_ratio[0]*0.01, random_state=43)
    rest = df.drop(train.index)
    test = rest.groupby('unique_artist_id').sample(frac=split_ratio[1]/(split_ratio[2]+split_ratio[1]), random_state=43)
    valid = rest.drop(test.index)
    data_split = {"train": train, "test": test, "valid": valid}
    return data_split
