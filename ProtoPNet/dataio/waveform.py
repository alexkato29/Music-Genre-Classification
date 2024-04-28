import os

import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio

class WaveformDataset(Dataset):
    def __init__(self, dataset_path, genres, augment=False):
        self.files = []
        self.labels = []
        self.output_length = 661500  # Songs are 30s samples at 22050Hz
        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                self.files.append(file_path)
                self.labels.append(genres.index(genre))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono if necessary
        
        if waveform.size(1) > self.output_length:
            waveform = waveform[:, :self.output_length]  # Trim to exactly 30s
        elif waveform.size(1) < self.output_length:
            padding_size = self.output_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding_size))  # Pad with zeros if less than 30s
        
        return waveform, self.labels[idx]