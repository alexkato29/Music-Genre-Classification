import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio

from audio_backbone import AudioNet

DATA_PATH = '/usr/xtmp/aak61/music-genre/genres_original'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

print(DATA_PATH)
print(GENRES)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs being used: {num_gpus}")
else:
    print("CUDA is not available. No GPUs are being used.")

# Audio Dataset
class RawAudioDataset(Dataset):
    def __init__(self, dataset_path, genres):
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
            waveform = waveform[:, :self.output_length]  # Trim to the fixed length
        elif waveform.size(1) < self.output_length:
            padding_size = self.output_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding_size))  # Pad with zeros
        
        return waveform, self.labels[idx]

dataset = RawAudioDataset(DATA_PATH, GENRES)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioNet(num_classes=len(GENRES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Basic training loop
num_epochs = 10
for epoch in range(num_epochs):  
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.shape)
        print(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')