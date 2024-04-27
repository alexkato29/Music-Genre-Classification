import torch
from audio_backbone import AudioNet

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Set Device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs being used: {num_gpus}", flush=True)
else:
    print("CUDA is not available. No GPUs are being used.", flush=True)

# Validation Dataset
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


VAL_PATH = '/usr/xtmp/aak61/music-genre/split_genres/val'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
val_dataset = RawAudioDataset(VAL_PATH, GENRES)


# Load the best model
model = AudioNet()
state_dict = torch.load('best_models/best_backbone_427.pth')
model.load_state_dict(state_dict)
model = model.to(device)
print('Model Successfully Loaded', flush=True)

# Run evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for wav, label in val_loader:
        wav = wav.to(device)
        label = label.to(device)

        # reshape and aggregate chunk-level predictions
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        # append labels and predictions
        y_true.append(label)
        y_pred.append(pred)


# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, xticklabels=GENRES, yticklabels=GENRES, cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix as a PNG file
plt.savefig('confusion_matrix.png')

# Print accuracy
print('Accuracy: %.4f' % accuracy)