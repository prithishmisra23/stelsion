# misra_model/train.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from .config import CONFIG
from .model import TechTonicsExoplanetDetector
from .preprocessor import preprocess
from .period_finder import find_best_period
from .view_generator import generate_all_views


class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    
    Standard BCE treats every sample equally.
    Focal Loss down-weights easy examples (empty space)
    and forces model to focus on hard-to-classify transits.
    
    alpha=0.75: weights the positive (planet) class higher
    gamma=2.0:  down-weighting factor for easy negatives
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = nn.BCELoss(reduction='none')(pred, target)
        pt  = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def process_batch(time_list, flux_list, periods, t0s, durations, config=CONFIG):
    """
    Takes a batch of raw time/flux arrays + cached periods.
    Runs full preprocessing + period finding (if not cached) + view generation.
    Returns tensors ready for the model.
    """
    batch_global   = []
    batch_local    = []
    batch_matrix   = []
    batch_metadata = []
    
    for time, flux, p, t0, dur in zip(time_list, flux_list, periods, t0s, durations):
        time  = time.numpy()
        flux  = flux.numpy()
        
        # Preprocess
        time_clean, flux_clean = preprocess(time, flux, method='adaptive')
        
        # Find period only if it's missing (0.0)
        period_info = {'period': p, 't0': t0, 'duration': dur}
        if p == 0.0:
            period_info = find_best_period(time_clean, flux_clean, config)
        
        # Generate views
        gv, lv, m2d = generate_all_views(time_clean, flux_clean, period_info, config)
        
        # Placeholder metadata (stellar_radius, temperature, log_g)
        # In production: load from catalog CSV
        metadata = np.array([1.0, 5778.0, 4.44], dtype=np.float32)
        metadata = (metadata - np.array([1.0, 5778.0, 4.44])) / np.array([0.5, 1000.0, 0.5])
        
        batch_global.append(gv)
        batch_local.append(lv)
        batch_matrix.append(m2d)
        batch_metadata.append(metadata)
    
    return (
        torch.tensor(np.array(batch_global)).unsqueeze(1),    # (B, 1, 2001)
        torch.tensor(np.array(batch_local)).unsqueeze(1),     # (B, 1, 201)
        torch.tensor(np.array(batch_matrix)).unsqueeze(1),    # (B, 1, 10, 200)
        torch.tensor(np.array(batch_metadata)),               # (B, 3)
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for time_batch, flux_batch, labels, periods, t0s, durations in loader:
        # Process raw signals into model inputs
        global_v, local_v, matrix_2d, metadata = process_batch(time_batch, flux_batch, periods, t0s, durations)
        
        global_v   = global_v.float().to(device)
        local_v    = local_v.float().to(device)
        matrix_2d  = matrix_2d.float().to(device)
        metadata   = metadata.float().to(device)
        labels     = labels.float().to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (faster on modern GPUs)
        with autocast():
            preds = model(global_v, local_v, matrix_2d, metadata).squeeze()
        
        loss  = criterion(preds.float(), labels.float())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_preds.extend((preds > 0.5).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(loader), f1


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for time_batch, flux_batch, labels, periods, t0s, durations in loader:
            global_v, local_v, matrix_2d, metadata = process_batch(time_batch, flux_batch, periods, t0s, durations)
            
            global_v   = global_v.float().to(device)
            local_v    = local_v.float().to(device)
            matrix_2d  = matrix_2d.float().to(device)
            metadata   = metadata.float().to(device)
            labels     = labels.float().to(device)
            
            probs = model(global_v, local_v, matrix_2d, metadata).squeeze()
            loss  = criterion(probs, labels)
            
            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    accuracy  = accuracy_score(all_labels, all_preds)
    auc       = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'auc': auc
    }


def train(config=CONFIG):
    """Main training function"""
    
    os.makedirs(config["results_dir"], exist_ok=True)
    
    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on: {device}")
    
    # Model
    model = TechTonicsExoplanetDetector(dropout_rate=config["dropout_rate"])
    model = model.to(device)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Data
    from .dataset_loader import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Training components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    criterion = FocalLoss(
        alpha=config["focal_alpha"],
        gamma=config["focal_gamma"]
    )
    scaler = GradScaler()  # Mixed precision
    
    # Training loop
    best_f1 = 0
    history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    for epoch in range(config["epochs"]):
        start = time.time()
        
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - start
        
        print(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.3f} | "
            f"Val F1: {val_metrics['f1']:.3f} | "
            f"Prec: {val_metrics['precision']:.3f} | "
            f"Rec: {val_metrics['recall']:.3f} | "
            f"Acc: {val_metrics['accuracy']:.3f} | "
            f"AUC: {val_metrics['auc']:.3f} | "
            f"{elapsed:.1f}s"
        )
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
            }, config["model_save_path"])
            print(f"  * Saved new best model (F1={best_f1:.4f})")
        
        # Log history
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
    
    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    return model, history


if __name__ == "__main__":
    train()
