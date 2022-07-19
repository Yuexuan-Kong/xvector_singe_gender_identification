
# Description
Code in this repository could be used for singer identification and singer gender recognition task. They both originated from [speechbrain recipes](https://github.com/speechbrain/speechbrain) and modifications are added due to our use case. The code is written with [speechbrain](https://github.com/speechbrain/speechbrain), which is a python package designed for speech processing, and based on pytorch.

# Data
Data used in this repository is mainly from Deezer's catalogue, with information merged from various external sources, such as *musicbrainz*, *wikidata*, *discogs and musicstory*. All the dataset creation is from [singer_feature_dataset](https://github.deezerdev.com/ykong/singer_features_dataset). Original audios and separated voices are pre-downloaded locally.

# Data preprocessing
This part is written in file [prepare_data.py](https://github.deezerdev.com/ykong/xvectors_training/blob/master/gender_recognition/prepare_data.py). For training and validation, segments of 5-15 seconds are randomly chosen. Because there is less tracks of women than men, so we sampled twice more from tracks of women. ID of tracks, starting and ending points, singer's ID (or singer's gender) will be saved in three json files after running the script. For gender recognition task, test dataset is a manually annotated dataset that contains 203 tracks.

# Model
The model uses TDNN architecture from [this paper](https://ieeexplore.ieee.org/abstract/document/8461375). Inputs are 24 bins log mel-spectrograms, calculated on-the-fly. It is a model trained for singer identification task that is divided into an embedding part and a classification part. All the hyperparameters and paths are saved and could be changed in [train.yaml](https://github.deezerdev.com/ykong/xvectors_training/blob/master/gender_recognition/train.yaml).
