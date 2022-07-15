import os
# import torch
import numpy as np
# import matplotlib.pyplot as plt
from deezer.datasource.filesystem import NetappFilesystem
from deezer.environment.provider import ConfigurationProvider
from deezer.environment.provider.vault import VaultConfigurationProvider


class Audio:

    def __init__(self, song_id, type, lyrics_start, lyrics_end, audiopath = None):

        """

        :param int song_id: song ID of deezer
        :param str type: "mixed", "voice" or "bgm". "mixed" is the original mp3 music, "voice" is the separated voice by spleeter, "bgm" is the seperated accompaniment
        :param filesystem:
        """
        self.type = type
        self.lyrics_start = lyrics_start
        self.duration = lyrics_end - lyrics_start
        self.lyrics_end = lyrics_end
        if type=="mixed":
            token = os.environ.get('VAULT_TOKEN', None)
            if token is None:
                raise EnvironmentError('VAULT_TOKEN envvar is not set')
            ConfigurationProvider.use(VaultConfigurationProvider(token))
            MNT_POINT = os.environ.get("MNT_POINT", "/data/music/output/")
            filesystem = NetappFilesystem(mount_point=MNT_POINT)
            self.path = filesystem.get_track_path(song_id, quality="mp3_128")
            self.format = "mp3"
        if type=="voice":
            self.path = audiopath + str(song_id) + "/" + os.listdir(audiopath + str(song_id))[0] +"/vocals.wav"
            self.format = "wav"
        if type=="bgm":
            self.path = audiopath + str(song_id) + "/" + os.listdir(audiopath + str(song_id))[0] + "/accompaniment.wav"
            self.format = "wav"

    def load_chunk(self, chunk_start, chunk_duration, sf, sr=None):
        """
        This function is to load a chunk from an audio.
        Parameters
        ----------
        chunk_start: float
            Where this chunk should start. This number should be the relative one to where lyrics start.
            ex. if chunk_start=0, then the chunk starts at where lyrics start.
        chunk_duration: fload
            How long the chunk should last.
        sf: SignalFactory
            Deezer tool to load process audio.
        sr: int
            Sampling frequency or resampling frequency. If sr=None, then there won't be resampling while loading.

        """
        self.duration = chunk_duration
        if type=="mixed":
            audio = sf.load(self.path, offset=chunk_start+self.lyrics_start,
                            duration=chunk_duration, sampling_frequency=sr).to_mono(inplace=True)
        else:
            audio = sf.load(self.path, offset=chunk_start,
                            duration=chunk_duration, sampling_frequency=sr).to_mono(inplace=True)

        self.data = audio.data
        self.sr = audio.sampling_frequency

    def write(self, savepath):
        from scipy.io.wavfile import write
        write(savepath, self.sr, self.data)

    def _f0(self):
        import crepe
        time, f0, confidence, activation = crepe.predict(self.data, self.sr, viterbi=True)
        return f0

    def f0_histo(self, num_bins=100, savepath=None):
        f0 = self._f0()
        plt.hist(f0, num_bins)
        if savepath:
            plt.savefig(savepath)

    def xvectors(self, inference):
        embedding = inference({"waveform": torch.from_numpy(np.transpose(self.data)), "sample_rate": self.sr})
        return embedding

    def mfcc(self, n_mfcc=13):
        from librosa.feature import mfcc as librosa_mfcc
        self.mfcc_coe = librosa_mfcc(y=self.data.mean(1), sr=self.sr) # downmix multiple channel data
        return self.mfcc_coe

    def musicnn_front_fea(self, offset, duration, model="MTT_musicnn", feature=None):
        """
        Use github repo https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb to extract audio features.
        :param str feature: feature is the name of the level of feature that we want extract
        """
        from musicnn.extractor import extractor
        if self.type=="mixed":
            taggram, tags, features = extractor(file_name=self.path, offset=offset+self.lyrics_start, duration=duration, model=model, extract_features=True)
        else:
            taggram, tags, features = extractor(file_name=self.path, offset=offset, duration=duration, model=model, extract_features=True)
        if "timbre" in feature or "temporal" in feature:
            frontend_features = np.concatenate([features['temporal'], features['timbral']], axis=1)
            return frontend_features

        if "cnn" in feature:
            return features["cnn3"]

        if "pool" in feature:
            return features["max_pool"]

        if "ultimate" in feature:
            return features["penultimate"]