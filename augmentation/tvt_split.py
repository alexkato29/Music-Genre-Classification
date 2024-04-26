import os
import shutil
import random

# Only need to run this file once to do the train/val/test split
random.seed(256)

# Define paths
DATA_PATH = '/usr/xtmp/aak61/music-genre/genres_original'
TARGET_PATH = '/usr/xtmp/aak61/music-genre/split_genres'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Create train, val, and test directories
subsets = ['train', 'val', 'test']
for subset in subsets:
    subset_path = os.path.join(TARGET_PATH, subset)
    os.makedirs(subset_path, exist_ok=True)
    for genre in GENRES:
        os.makedirs(os.path.join(subset_path, genre), exist_ok=True)

# Function to split data
def split_data(genre_path, genre, files):
    random.shuffle(files)

    total_size = len(files)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)    

    # Assign files to train, val, test
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    # Copy files to their respective directories
    for file, subset in zip([train_files, val_files, test_files], subsets):
        for f in file:
            src_path = os.path.join(genre_path, f)
            dest_path = os.path.join(TARGET_PATH, subset, genre, f)
            shutil.copy2(src_path, dest_path)

# Process each genre directory
for genre in GENRES:
    genre_path = os.path.join(DATA_PATH, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    split_data(genre_path, genre, files)

print("Data successfully reorganized into train, val, and test subsets.")