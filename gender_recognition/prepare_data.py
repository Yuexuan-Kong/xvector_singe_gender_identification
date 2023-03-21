"""
Downloads and creates data manifest files for gender identification.
For this task, different singers must appear in train,
validation, and test sets.
"""

import os
import ujson as json
import random
import logging
import numpy as np

import pandas as pd
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
    min_n_track=None,
    max_n_track=None,
    split_ratio=None,
    golden_path=None,

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
    golden_path: str
        Path to manually annotated data.

    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # extract_create_json_sid(original_folder, save_json_test, save_json_train, save_json_valid, split_ratio, training_path,
    #                 voice_folder)
    extract_create_json_gender(original_folder, golden_path, save_json_test, save_json_train, save_json_valid,
                               split_ratio, training_path, voice_folder)


def extract_create_json_gender(original_folder, golden_path, save_json_test, save_json_train, save_json_valid, split_ratio, training_path,
                    voice_folder):
    df = pd.read_pickle(training_path)
    df = df[df["lyrics_end"]-df["lyrics_start"]>=3]
    # singer-wise split into train and validation.
    data_split = split_sets_gender(df, golden_path, split_ratio)

    # Creating json files
    # TODO in the future, we don't add spleeter for validation and test
    # create_golden_json(data_split["test"], save_json_test, False, original_folder, voice_folder)
    # create_json(data_split["valid"], save_json_valid, True, [3, 3], original_folder, voice_folder)
    # create_json_whole_segment(data_split["valid"], save_json_valid, original_folder)
    # create_json(data_split["train"], save_json_train, True, [3, 3], original_folder, voice_folder)
    create_json(df, save_json_train, True, [3, 3], original_folder, voice_folder)

def create_json(df, json_file, add_spleeter, chunk_length, original_folder, voice_folder):
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
    chunk_length : list
        list[0] is the shortest chunk we want, list[1] is the longest chunk we want
    original_folder : str
        Path to the folder where origined cropped tracks are saved.
    voice_folder : str
        Path to the folder where only voices of cropped tracks are saved.
    """
    # Processing all the wav files in the list
    json_dict = {}
    min_c = chunk_length[0]
    max_c = chunk_length[1]
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
        for i in range(0, int(round(duration/10)*gender)):

            # Random select segment that are larger than 3s and 10s
            path = original_folder+str(row["SNG_ID"])+".wav"
            start = round(random.uniform(0, duration-min_c), 1)
            end = round(random.uniform(max(min_c, start+min_c), min(start+max_c, duration)), 1)

            g = "man" if gender==1 else "woman"
            json_dict[str(row["SNG_ID"])+f"_origin_{i+1}"] = {
                "wav": path,
                "start": start,
                "end": end,
                "length": round(end-start, 1),
                "gender": g,
            }
            if add_spleeter:
                start = round(random.uniform(0, duration-min_c), 1)
                end = round(random.uniform(max(min_c, start+min_c), min(start+max_c, duration)), 1)
                path = voice_folder+str(row["SNG_ID"]) + "/" +"vocals.wav"

                json_dict[str(row["SNG_ID"])+f"_voice_{i+1}"] = {
                    "wav": path,
                    "start": start,
                    "end": end,
                    "length": round(end-start, 1),
                    "gender": g,
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


def split_sets_sid(df, split_ratio):
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


def split_sets_gender(df, golden_path, split_ratio):
    """
    This function is used to separate dataset by singer-wise for gender recognition. We want singers in test sets have
    not showed up in the training set yet. Test set is reading from the manually annotated golden dataset.
    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe to be separated into training and validation datasets
    split_ratio: list
        Split ratio for training and validation

    Returns
    -------
    Dictionary of Pandas Dataframe containing train, valid, and test splits.
    """

    # Read golden test dataset for test.
    test = pd.read_pickle(golden_path)

    # Group by artist id
    artist_count = df.groupby(["unique_artist_id"])["SNG_ID"].count().reset_index(name="count").sort_values(["count"],
                                                                                              ascending=False)
    train_artist = artist_count.sample(frac=split_ratio[0]*0.01, random_state=43)
    train = df[df["unique_artist_id"].isin(train_artist.unique_artist_id)]
    valid = df.drop(train.index)
    data_split = {"train": train, "test": test, "valid": valid}
    return data_split


def create_golden_json(df, json_file, add_spleeter, original_folder, voice_folder):
    """
    Similar function to create_json, but for test dataset, we take the whole segments instead of random
    sampled ones.
    Parameters
    ----------
    df: pandas.Dataframe
    json_file: str
        Path to save golden json file
    add_spleeter: Bool
        If add spleeter or not.
    original_folder: str
        Path to folder where original tracks are saved.
    voice_folder: str
        Path to folder where separated voices are saved.

    """
    json_dict = {}
    for s, row in tqdm(df.iterrows(), total=df.shape[0]):

        duration = round(row["lyrics_end"]-row["lyrics_start"], 1)
        try:
            gender = "man" if row["gender_gt"]==1 else "woman"
        except KeyError:
            gender = "man" if row["gender"]==1 else "woman"

        # Select the whole segment, so that we are sure there's singing voice
        path = original_folder+str(s)+".wav"

        json_dict[str(s)+f"_origin"] = {
            "wav": path,
            "start": 0,
            "end": duration,
            "length": duration,
            "gender": gender
        }
        if add_spleeter:
            path = voice_folder + "spleetered_golden_test/" + str(s) + "/" + os.listdir(voice_folder+"spleetered_golden_test/"
                                                                                  + str(s))[0] +"/vocals.wav"

            json_dict[str(s)+f"_voice"] = {
                "wav": path,
                "start": 0,
                "end": duration,
                "length": duration,
                "gender": gender
            }

            # path = voice_folder + "spleetered_golden_test/" + str(s) + "/" + os.listdir(voice_folder+"spleetered_golden_test/"
            #                                                                             + str(s))[0] +"/accompaniment.wav"
            # json_dict[str(s)+f"_acc"] = {
            #     "wav": path,
            #     "start": 0,
            #     "end": duration,
            #     "length": duration,
            #     "gender": gender
            # }
        # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")

def create_json_1200(df, json_file, test_folder):
    """
    Similar function to create_json, but for test dataset, we take the whole segments instead of random
    sampled ones.
    Parameters
    ----------
    df: pandas.Dataframe
    json_file: str
        Path to save golden json file
    1200_folder: str
        Path to folder where 3s segments are saved.

    """
    json_dict = {}
    for s, row in tqdm(df.iterrows(), total=df.shape[0]):
        for i in range(1,7):
            duration = 3
            gender = "man" if int(row["gender"])==1 else "woman"

            # Select the whole segment, so that we are sure there's singing voice
            path = test_folder+row["song_name"]+"_"+str(i)+".wav"

            json_dict[row["song_name"]+"_"+str(i)] = {
                "wav": path,
                "start": 0,
                "end": duration,
                "length": duration,
                "gender": gender
            }
        # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")


def create_json_new_training(df, json_file, add_spleeter, chunk_length, original_folder, voice_folder):
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
    chunk_length : list
        list[0] is the shortest chunk we want, list[1] is the longest chunk we want
    original_folder : str
        Path to the folder where origined cropped tracks are saved.
    voice_folder : str
        Path to the folder where only voices of cropped tracks are saved.
    """
    # Processing all the wav files in the list
    json_dict = {}
    min_c = chunk_length[0]
    max_c = chunk_length[1]
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
        for i in range(0, int(round(duration/10)*gender)):

            # Random select segment that are larger than 3s and 10s
            path = original_folder+str(row["SNG_ID"])+".wav"
            start = round(random.uniform(0, duration-min_c), 1)
            end = round(random.uniform(start+min_c, min(start+max_c, duration)), 1)

            g = "man" if gender==1 else "woman"
            json_dict[str(row["SNG_ID"])+f"_origin_{i+1}"] = {
                "wav": path,
                "start": start,
                "end": end,
                "length": round(end-start, 1),
                "gender": g,
            }
            if add_spleeter:
                start = round(random.uniform(0, duration-min_c), 1)
                end = round(random.uniform(start+min_c, min(start+max_c, duration)), 1)
                path = voice_folder+str(row["SNG_ID"])+".wav"

                json_dict[str(row["SNG_ID"])+f"_voice_{i+1}"] = {
                    "wav": path,
                    "start": start,
                    "end": end,
                    "length": round(end-start, 1),
                    "gender": g,
                }
        # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")


def create_json_whole_segment(df, json_file, original_folder):
    """
    Create a json file for validation dataset that has all segments of tracks, instead of sampling randomly, used to
    calculate bias, and to see if the error rate is still a lot higher than golden test dataset.
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

        gender = 1 if np.isnan(row["gender"]) else row["gender"]
        # Random select segment that are larger than 3s and 10s
        path = original_folder+str(row["SNG_ID"])+".wav"
        duration = round(row["lyrics_end"] - row["lyrics_start"], 1)

        g = "man" if gender==1 else "woman"
        json_dict[str(row["SNG_ID"])] = {
            "wav": path,
            "start": 0,
            "end": duration,
            "length": duration,
            "gender": g,
        }
        # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")

if __name__ == "__main__":
    df_test = pd.read_csv("../data/test_annotation.csv", sep=";")
    create_json_1200(df_test, "test.json", "1200_segments/")
    df_train = pd.read_csv("../data/train_exclude_unique.csv")
    create_json_new_training(df_train, "train.json", True, [3,3], "origins/", "voices/")
