import os.path

from Audio import Audio
import pandas as pd
from tqdm import tqdm
from deezer.audio.signal import SignalFactory
# from spleeter.separator import Separator

def download_origin(datapath, origin_folder):

    df = pd.read_pickle(datapath)
    sf = SignalFactory()
    # separator = Separator('spleeter:2stems')
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = origin_folder+str(row["SNG_ID"])+".wav"
        if not os.path.exists(path):
            audio = Audio(row["SNG_ID"], "mixed", row["lyrics_start"], row["lyrics_end"])
            audio.load_chunk(audio.lyrics_start, audio.duration, sf, sr=16000)
            audio.write(origin_folder+str(row["SNG_ID"])+".wav")
        else:
            print("exists")

        # Use spleeter to extract only voice from the same segment
        # separator.separate_to_file(origin_folder+row["SNG_ID"], destination=split_folder+row["SNG_ID"])

def download_origin_preview(datapath, origin_folder):

    df = pd.read_pickle(datapath)
    sf = SignalFactory()
    # separator = Separator('spleeter:2stems')
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        path = origin_folder+str(i)+".wav"
        if not os.path.exists(path):
            audio = Audio(i, "mixed", row["preview_start"], row["preview_end"])
            audio.load_chunk(audio.lyrics_start, audio.duration, sf, sr=16000)
            audio.write(path)
        else:
            print("exists")

def download_voice(origin_folder, split_folder):
    from spleeter.separator import Separator
    import os

    # Using embedded configuration.
    separator = Separator('spleeter:2stems')

    for i in tqdm(os.listdir(origin_folder)):
        if not os.path.exists(split_folder+i.split(".")[0]):
            separator.separate_to_file(audio_descriptor=origin_folder+i, destination=split_folder)
        else: continue

def download_human_jury_3s(json_path):
    """
    Download segments of 3 seconds for human jury.
    Parameters
    ----------
    json_path: str
        Path to the json file that contains information of each segment
    """
    from scipy.io.wavfile import write
    rootpath = "../data/human_jury_3s/"
    df = pd.read_json(json_path).transpose()
    sf = SignalFactory()
    for s, row in tqdm(df.iterrows(), total=df.shape[0]):

        signal = sf.load(row["wav"], offset=row["start"],
                    duration=row["end"]-row["start"], sampling_frequency=16000).to_mono(inplace=True)

        # Save different kinds of segments into different files
        if row["gender"] == "man":
            savepath = rootpath+str(s)+"_1.wav"
        elif row["gender"] == "woman":
            savepath = rootpath+str(s)+"_2.wav"

        write(savepath, 16000, signal.data)


def download_background_3s(json_path):
    """
    Download segments of 3 seconds for instrumental/female/male classification.
    Parameters
    ----------
    json_path: str
        Path to the json file that contains information of each segment
    """
    from scipy.io.wavfile import write
    rootpath = "../data/instrumental/"
    df = pd.read_json(json_path).transpose()
    sf = SignalFactory()
    for s, row in tqdm(df.iterrows(), total=df.shape[0]):

        signal = sf.load(row["wav"], offset=row["start"],
                    duration=row["end"]-row["start"], sampling_frequency=16000).to_mono(inplace=True)

        savepath = rootpath+str(s)+"_3.wav"

        write(savepath, 16000, signal.data)


# download_origin("../data/training_gender", "../data/origins/")
# download_voice("../data/origins/", "../data/voices/")
# download_origin_preview("../data/test_raw_207", "../data/golden_preview/")
download_background_3s("../instrumental_female_male/dali_test.json")