import os
import numpy as np

import librosa
import soundfile as sf

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def pitch(audio_path, save_path, file, n_semitones):
    y, sr = librosa.load(audio_path, sr=None) 

    # Go +/- semitones
    for semitone in range(-n_semitones, n_semitones + 1):
        # We copy the original files over, so no need to recopy
        if semitone == 0:
            continue

        # Named up/down accordingly
        if semitone > 0:
            aug_file_path = os.path.join(save_path, f"up_{semitone}_{file}")
        else:
            aug_file_path = os.path.join(save_path, f"down_{abs(semitone)}_{file}")

        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
        sf.write(aug_file_path, y_shifted, sr)


def add_white_noise(audio_path, save_path, file, noise_level):
    y, sr = librosa.load(audio_path, sr=None)
    noise = np.random.randn(len(y)) * noise_level
    y_noisy = y + noise
    y_noisy = np.clip(y_noisy, -1, 1)
    noisy_file_path = os.path.join(save_path, f"noisy_{file}")
    sf.write(noisy_file_path, y_noisy, sr)

def quiet(audio_path, save_path, file, quiet_factor):
    y, sr = librosa.load(audio_path, sr=None)
    y_quiet = y * quiet_factor
    noisy_file_path = os.path.join(save_path, f"quiet_{file}")
    sf.write(noisy_file_path, y_quiet, sr)



def augment_training_data(train_dir, aug_dir):
    for genre in GENRES:
        genre_dir = os.path.join(train_dir, genre)
        aug_genre_dir = os.path.join(aug_dir, genre)
        os.makedirs(aug_genre_dir, exist_ok=True)
        
        for file in os.listdir(genre_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_dir, file)
                pitch(file_path, aug_genre_dir, file, n_semitones=2)
                add_white_noise(file_path, aug_genre_dir, file, noise_level=0.005)
                quiet(file_path, aug_genre_dir, file, quiet_factor=0.4)
        print(f"{genre} genre complete")

# Example usage
DATA_PATH = '/usr/xtmp/aak61/music-genre/split_genres/push/'
AUG_PATH = '/usr/xtmp/aak61/music-genre/split_genres/train_augmented/'
augment_training_data(DATA_PATH, AUG_PATH)

