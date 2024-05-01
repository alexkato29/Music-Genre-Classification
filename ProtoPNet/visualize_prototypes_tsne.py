import argparse
import torch
import numpy as np

from  configs.cfg import get_cfg_defaults

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE



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


# Load model from memory
ppnet = torch.load(cfg.OUTPUT.MODEL_DIR + "/10_push0.6577.pth")

# Get the prototypes
genres = cfg.DATASET.WAVEFORM.GENRES
prototypes = ppnet.prototype_vectors
prototypes_reshaped = prototypes.view(100, 128).cpu().numpy()

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=25)
prototypes_2d = tsne.fit_transform(prototypes_reshaped)

# Repeat each genre 10 times for labeling.
genre_labels = np.repeat(genres, 10)

# Custom color map. Not a fan of tab10
custom_colors = ['#3E57C1', '#A25A00', '#B6B6B6', '#B533FF', '#27D86C', 
                 '#EFCB10', '#EA1F15', '#FC00FF', '#3CC3B3', '#010504']
cmap = ListedColormap(custom_colors)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], c=np.arange(100) // 10, cmap=cmap, label=genre_labels)
plt.title('Prototype Vectors by Genre')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)

# Create legend with genre labels
handles, _ = scatter.legend_elements(prop="colors")
plt.legend(handles, genres, title="Genres")

# Save the figure as a PNG file
plt.savefig('prototypes_tsne.png')
plt.show()