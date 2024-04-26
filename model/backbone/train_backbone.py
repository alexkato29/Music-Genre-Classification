import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torchaudio

from audio_backbone import AudioNet

DATA_PATH = '/usr/xtmp/aak61/music-genre/genres_original'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs being used: {num_gpus}", flush=True)
else:
    print("CUDA is not available. No GPUs are being used.", flush=True)

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
            waveform = waveform[:, :self.output_length]  # Trim to exactly 30s
        elif waveform.size(1) < self.output_length:
            padding_size = self.output_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding_size))  # Pad with zeros if less than 30s
        
        return waveform, self.labels[idx]

dataset = RawAudioDataset(DATA_PATH, GENRES)

# Make a training and val set
total_size = len(dataset)
train_size = int(total_size * 0.8)  # 80% for training
val_size = total_size - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioNet(num_classes=len(GENRES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

# Training Code
num_epochs = 10
best_val_acc = 0
for epoch in range(num_epochs):
    print(f"Now Training Epoch {epoch+1}", flush=True)

    # Training Loop
    model.train()
    train_loss = 0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss for the epoch
    train_loss /= len(train_dataloader)

    # Validation Loop
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += accuracy(outputs, labels)

    # Calculate average loss and accuracy over all validation batches
    val_loss /= len(val_dataloader)
    val_accuracy /= len(val_dataloader)

    print(f'\tEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%', flush=True)
    
    # Save Condition
    if val_accuracy > best_val_acc:
        torch.save(model.state_dict(), 'best_backbone.pth')
