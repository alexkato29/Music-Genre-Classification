import os
import time

import numpy as np
import matplotlib.pyplot as plt

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


# Load the data onto the GPU and initialize training params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioNet(num_classes=len(GENRES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Accuracy function for validation set 
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

# Training Code
train_losses = []
val_losses = []
val_accuracies = []
val_acc_avg_5_epochs = []  

num_epochs = 30
best_val_acc_avg = 0

for epoch in range(num_epochs):
    print(f"Now Training Epoch {epoch+1}", flush=True)
    start = time.time()

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
    train_losses.append(train_loss / len(train_dataloader))

    # Validation Loop
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += accuracy(outputs, labels)
    val_losses.append(val_loss / len(val_dataloader))
    val_accuracies.append(val_accuracy / len(val_dataloader))

    # Calculate the moving average over the last 5 epochs for validation accuracy
    if len(val_accuracies) >= 5:
        mean_val_acc = np.mean(val_accuracies[-5:])
        val_acc_avg_5_epochs.append(mean_val_acc)
    else:
        val_acc_avg_5_epochs.append(val_accuracies[-1])  # Use the current accuracy if <5 epochs have completed

    end = time.time()

    print(f'\tEpoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%, Moving Avg Val Acc: {val_acc_avg_5_epochs[-1]:.2f}%, Time: {end-start:.2f}s', flush=True)
    
    # Save Condition based on moving average
    if len(val_accuracies) >= 5 and val_acc_avg_5_epochs[-1] > best_val_acc_avg:
        torch.save(model.state_dict(), 'best_backbone.pth')
        best_val_acc_avg = val_acc_avg_5_epochs[-1]
        print("Saved new best model based on moving average validation accuracy", flush=True)

# Plotting and saving the graph
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training.png')