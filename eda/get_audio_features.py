import os
import numpy as np
import pandas as pd

import librosa

directory = "/usr/xtmp/aak61/music-genre/split_genres/push/"

data = []

for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename.endswith(".wav"):
            parts = filename.split('.')
            genre = parts[0]

            file_path = os.path.join(directory, genre)
            file_path = os.path.join(file_path, filename)
            y, sr = librosa.load(file_path, sr=22050)

            # Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

            # Spectral Features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

            data.append({
                'name': filename, 
                'genre': genre,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_flatness': spectral_flatness
            })


audio = pd.DataFrame(data)
audio.to_csv("features.csv")