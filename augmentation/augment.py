import os
import random
import numpy as np

import librosa
import torchaudio
import soundfile as sf
from scipy.signal import butter, sosfilt

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def apply_polarity_inversion(waveform):
    return -waveform

def apply_noise(waveform, snr_min=7, snr_max=12):
    snr = random.uniform(snr_min, snr_max)
    noise_amp = np.std(waveform) / snr
    noise = np.random.normal(loc=0, scale=1, size=waveform.shape) * noise_amp
    return waveform + noise

def apply_reverb(waveform, sample_rate):
    # Using torchaudio's convolutional reverb
    # Note: This requires an impulse response file
    ir_path = 'path_to_impulse_response.wav'
    ir_waveform, ir_sr = torchaudio.load(ir_path)
    if ir_sr != sample_rate:
        ir_waveform = torchaudio.transforms.Resample(orig_freq=ir_sr, new_freq=sample_rate)(ir_waveform)
    reverb = torchaudio.transforms.Convolution(reverb=ir_waveform[0:1])
    return reverb(waveform)

def apply_pitch_shift(waveform, sample_rate, n_steps):
    shifted = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
    return shifted

def apply_high_low_pass(waveform, sample_rate, high=True):
    cutoff = 2000  # cutoff frequency in Hz
    order = 2  # filter order
    sos = butter(order, cutoff, 'hp' if high else 'lp', fs=sample_rate, output='sos')
    filtered = sosfilt(sos, waveform)
    return filtered

def normalize_waveform(waveform):
    max_amplitude = np.max(np.abs(waveform))
    return waveform / max_amplitude

def get_augmentations(waveform, sample_rate):
    # Probability aug = 0.9475
    # E[num copies of orig song] = 1.0525
    if random.random() < 0.5:
        waveform = apply_polarity_inversion(waveform)
    if random.random() < 0.7:
        waveform = apply_noise(waveform)
    if random.random() < 0.5:
        waveform = apply_high_low_pass(waveform, sample_rate, high=random.choice([True, False]))
    if random.random() < 0.3:
        semitones = random.randint(-4, 4)
        waveform = apply_pitch_shift(waveform, sample_rate, n_steps=semitones)

    return normalize_waveform(waveform)

def random_augment(audio_path, save_path, file, n_times):
    y, sr = librosa.load(audio_path, sr=None) 

    # Save the original once, guaranteed
    aug_file_path = os.path.join(save_path, f"orig_{file}")
    sf.write(aug_file_path, y, sr)

    # Then do modifications
    for i in range(n_times):
        y_aug = get_augmentations(y, 22050)
        aug_file_path = os.path.join(save_path, f"aug{i}_{file}")
        sf.write(aug_file_path, y_aug, sr)


def augment_training_data(train_dir, aug_dir):
    for genre in GENRES:
        genre_dir = os.path.join(train_dir, genre)
        aug_genre_dir = os.path.join(aug_dir, genre)
        os.makedirs(aug_genre_dir, exist_ok=True)
        
        for file in os.listdir(genre_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_dir, file)
                random_augment(file_path, aug_genre_dir, file, n_times=9)
        print(f"{genre} genre complete", flush=True)


# Example usage
DATA_PATH = '/usr/xtmp/aak61/music-genre/split_genres/push/'
AUG_PATH = '/usr/xtmp/aak61/music-genre/split_genres/train_augmented_2/'
augment_training_data(DATA_PATH, AUG_PATH)

