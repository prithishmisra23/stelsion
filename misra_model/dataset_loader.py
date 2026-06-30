# misra_model/dataset_loader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .config import CONFIG


class KeplerDataset(Dataset):
    """
    Loads preprocessed .npz files and pairs with labels from catalog CSV.
    Each .npz contains: time (array), flux (array)
    Each label comes from dataset_index.csv signal_class column
    """
    
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = np.load(path)
        time = data['time'].astype(np.float32)
        flux = data['flux'].astype(np.float32)
        label = self.labels[idx]
        
        if 'period' in data:
            period = float(data['period'])
            t0 = float(data['t0'])
            duration = float(data['duration'])
        else:
            period, t0, duration = 0.0, 0.0, 0.0
        
        if self.transform:
            time, flux = self.transform(time, flux)
        
        return time, flux, label, period, t0, duration


def load_catalog(catalog_path, dataset_dir):
    """
    Read the catalog CSV and match to available .npz files.
    Returns matched file paths and binary labels.
    planet/candidate → 1
    false positive / stellar_eclipse / centroid_offset → 0
    """
    df = pd.read_csv(catalog_path)
    
    # Build label mapping — binary for now
    def map_label(row):
        label = str(row.get('label', '')).lower()
        
        if 'transit' in label and 'stellar' not in label:
            return 1
        return 0
    
    df['binary_label'] = df.apply(map_label, axis=1)
    
    # Match to actual downloaded .npz files
    file_paths = []
    binary_labels = []
    
    for _, row in df.iterrows():
        kepid = str(int(row['kepid']))
        npz_path = os.path.join(dataset_dir, f"{kepid}.npz")
        
        if os.path.exists(npz_path):
            file_paths.append(npz_path)
            binary_labels.append(row['binary_label'])
    
    print(f"Found {len(file_paths)} matched files")
    print(f"Planets: {sum(binary_labels)}, Non-planets: {len(binary_labels)-sum(binary_labels)}")
    
    return file_paths, binary_labels


def custom_collate(batch):
    time_list = [torch.tensor(item[0]) for item in batch]
    flux_list = [torch.tensor(item[1]) for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    periods = [item[3] for item in batch]
    t0s = [item[4] for item in batch]
    durations = [item[5] for item in batch]
    return time_list, flux_list, labels, periods, t0s, durations

def get_dataloaders(config=CONFIG):
    """
    Returns train, val, test DataLoaders
    Split is 80/10/10 stratified by label (star-level split)
    """
    file_paths, labels = load_catalog(
        config["catalog_path"],
        config["dataset_dir"]
    )
    
    # Star-level split — no star appears in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test,
        test_size=0.5,
        stratify=y_test,
        random_state=42
    )
    
    train_ds = KeplerDataset(X_train, y_train)
    val_ds = KeplerDataset(X_val, y_val)
    test_ds = KeplerDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2,
        collate_fn=custom_collate
    )

    
    return train_loader, val_loader, test_loader
