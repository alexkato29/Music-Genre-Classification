import os
import numpy as np

import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

def get_wav(wav_path, save_path):
    y, sr = librosa.load(wav_path, sr=22050)
    sf.write(save_path, y, sr)

def plot_mel_spectrogram(wav_path, save_path):
    y, sr = librosa.load(wav_path, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Frequency Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


genre = "blues"
for i in range(100):
    num = i
    path = f'/usr/xtmp/aak61/music-genre/split_genres/push/{genre}/{genre}.000{num}.wav'
    if os.path.isfile(path):
        plot_mel_spectrogram(f'/usr/xtmp/aak61/music-genre/split_genres/push/{genre}/{genre}.000{num}.wav', f'blues/{genre}{num}_mel.png')
        get_wav(f'/usr/xtmp/aak61/music-genre/split_genres/push/{genre}/{genre}.000{num}.wav', f'blues/{genre}{num}.wav')
