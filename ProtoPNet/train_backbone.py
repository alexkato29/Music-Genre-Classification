import argparse, os
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from pretrained_backbones.audio_backbone import AudioNet
from dataio.dataset import get_dataset
from configs.cfg import get_cfg_defaults

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Get the config file to get the data
cfg = get_cfg_defaults()
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0') 
parser.add_argument('--configs', type=str, default='cub.yaml')
args = parser.parse_args()
cfg.merge_from_file(args.configs)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs being used: {num_gpus}", flush=True)
else:
    print("CUDA is not available. No GPUs are being used.", flush=True)

train_loader, _, val_loader, _ = get_dataset(cfg)

# Assuming model, train_dataloader, and val_dataloader are already defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioNet(num_classes=len(GENRES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Define accuracy calculation
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

# Initialize storage for epoch data
train_losses = []
val_losses = []
val_accuracies = []
val_acc_avg_5_epochs = []  
best_val_acc_avg = 0

num_epochs = 30
for epoch in range(num_epochs):
    print(f"Now Training Epoch {epoch+1}", flush=True)
    start = time.time()

    # Training Loop
    model.train()
    train_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Validation Loop
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += accuracy(outputs, labels)
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy / len(val_loader))

    # Update Learning Rate
    scheduler.step()

    # Calculate the moving average over the last 5 epochs for validation accuracy
    if len(val_accuracies) >= 5:
        mean_val_acc = np.mean(val_accuracies[-5:])
        val_acc_avg_5_epochs.append(mean_val_acc)
    else:
        val_acc_avg_5_epochs.append(val_accuracies[-1])

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


# Load the best model
model = AudioNet(num_classes=10)
state_dict = torch.load('best_backbone.pth')
model.load_state_dict(state_dict)
model = model.to(device)
print('Model Loaded from Training Successfully', flush=True)

# Run evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for wav, label in val_loader:
        wav = wav.to(device)
        label = label.to(device)

        outputs = model(wav)  # Corrected this line
        _, pred = torch.max(outputs, 1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(pred.cpu().numpy()) 

# Calculate confusion matrix and accuracy
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=GENRES, yticklabels=GENRES, cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')

print('Accuracy: %.4f' % accuracy)