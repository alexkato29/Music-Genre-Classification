from dataio.waveform import WaveformDataset
from torch.utils.data import DataLoader

def get_dataset(cfg):    
    train_dataset = WaveformDataset(cfg.DATASET.WAVEFORM.TRAIN_PATH, cfg.DATASET.WAVEFORM.GENRES)
    train_loader = DataLoader(train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True)  

    push_dataset = WaveformDataset(cfg.DATASET.WAVEFORM.PUSH_PATH, cfg.DATASET.WAVEFORM.GENRES)
    push_loader = DataLoader(push_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True)

    val_dataset = WaveformDataset(cfg.DATASET.WAVEFORM.VAL_PATH, cfg.DATASET.WAVEFORM.GENRES)
    val_loader = DataLoader(val_dataset, batch_size=cfg.DATASET.OTHER_BATCH_SIZE, shuffle=False) 

    test_dataset = WaveformDataset(cfg.DATASET.WAVEFORM.TEST_PATH, cfg.DATASET.WAVEFORM.GENRES)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATASET.OTHER_BATCH_SIZE, shuffle=False)
    
    
    
    return train_loader, push_loader, val_loader, test_loader
