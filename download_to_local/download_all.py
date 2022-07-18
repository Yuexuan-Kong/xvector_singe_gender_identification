import os.path

from Audio import Audio
import pandas as pd
from tqdm import tqdm
# from deezer.audio.signal import SignalFactory
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

def download_voice(origin_folder, split_folder):
    from spleeter.separator import Separator
    import os

    # Using embedded configuration.
    separator = Separator('spleeter:2stems')

    for i in tqdm(os.listdir(origin_folder)):
        if not os.path.exists(split_folder+i.split(".")[0]):
            separator.separate_to_file(audio_descriptor=origin_folder+i, destination=split_folder)
        else: continue


# download_origin("../data/training_gender", "../data/origins/")
download_voice("../data/origins/", "../data/voices/")