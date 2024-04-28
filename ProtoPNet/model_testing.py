import argparse, os
import torch
from utils.util import create_logger
from os import mkdir

from  configs.cfg import get_cfg_defaults
from dataio.dataset import get_dataset

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Print GPUs (multiple GPU clusters at once?)
if torch.cuda.is_available():
    device = "cuda"
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs being used: {num_gpus}", flush=True)
else:
    device = "cpu"
    print("CUDA is not available. No GPUs are being used.", flush=True)

cfg = get_cfg_defaults()

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0') 
parser.add_argument('--configs', type=str, default='cub.yaml')
args = parser.parse_args()

# Update the hyperparameters from default to the ones we mentioned in arguments
cfg.merge_from_file(args.configs)

if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
    mkdir(cfg.OUTPUT.MODEL_DIR)
if not os.path.exists(cfg.OUTPUT.IMG_DIR):
    mkdir(cfg.OUTPUT.IMG_DIR)

# Create Logger Initially
log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))

# Get the datasets for exploring
_, push_loader, val_loader, test_dataset = get_dataset(cfg)

# Load model from memory
ppnet = torch.load(cfg.OUTPUT.MODEL_DIR + "/10_push0.6376.pth")


# Evaluating
ppnet.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for wav, label in val_loader:
        wav = wav.to(device)
        label = label.to(device)

        logits, min_distances = ppnet(wav)  # Corrected this line
        _, pred = torch.max(logits, 1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(pred.cpu().numpy()) 

# Calculate confusion matrix and accuracy
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=cfg.DATASET.WAVEFORM.GENRES, yticklabels=cfg.DATASET.WAVEFORM.GENRES, cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')

print('Accuracy: %.4f' % accuracy)