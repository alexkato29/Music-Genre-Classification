import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from audio_backbone import AudioNet

DATA_PATH = '/usr/xtmp/aak61/music-genre/genres_original'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

print(settings.DATA_PATH)
print(settings.GENRES)
print(AudioNet.test_import())
print(torch.cuda.is_available())