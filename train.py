import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model

N_FOLDS = 5              
EPOCHS_PER_FOLD = 3      # Ultra-fast iteration limit
BACKBONE = "resnet34"    
PATIENCE = 6

def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

def get_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, azimuth_feat, y in loader:
            images, azimuth_feat = images.to(device), azimuth_feat.to(device)
            p = torch.softmax(model(images, azimuth_feat), dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(probs), np.array(labels)

def tune_threshold(probs: np.ndarray, labels: np.ndarray):
    best_thresh, best_bacc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        b = balanced_accuracy_score(labels, (probs >= t).astype(int))
        if b > best_bacc:
            best_bacc, best_thresh = b, t
    return float(best_thresh), float(best_bacc)

def train_one_fold(fold: int, train_idx, val_idx, train_df, device):
    train_full = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=True)
    val_full = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=False)

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    sampler = build_sampler(train_df['label'].values[train_idx])
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = get_lunar_model(backbone_name=BACKBONE, pretrained=True, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    warmup_epochs = 0 # Removed warmup for short 3 epoch run
    def lr_lambda(epoch):
        progress = epoch / max(1, EPOCHS_PER_FOLD)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    ckpt_path = f'best_lunar_model_fold{fold}.pth'
    best_bacc = -1.0
    no_improve = 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        total_loss = 0.0
        for images, azimuth_feat, y in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{EPOCHS_PER_FOLD}"):
            images, azimuth_feat, y = images.to(device), azimuth_feat.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(images, azimuth_feat)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        probs, y_val = get_probs(model, val_loader, device)
        bacc = balanced_accuracy_score(y_val, (probs >= 0.5).astype(int))
        print(f"[Fold {fold}] Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} "
              f"val_bacc@0.5={bacc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if bacc > best_bacc:
            best_bacc = bacc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[Fold {fold}] Early stopping at epoch {epoch+1} (best={best_bacc:.4f}).")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    probs, y_val = get_probs(model, val_loader, device)
    print(f"[Fold {fold}] final best val_bacc@0.5 = {best_bacc:.4f}\n")
    return probs, val_idx

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df = pd.read_csv('data/train_metadata.csv')
    labels_all = train_df['label'].values
    n = len(train_df)

    n_folds = max(2, N_FOLDS) if N_FOLDS > 1 else 1

    if n_folds == 1:
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels_all)
        probs, val_idx_ret = train_one_fold(0, train_idx, val_idx, train_df, device)
        oof_probs = np.full(n, np.nan)
        oof_probs[val_idx_ret] = probs
        mask = ~np.isnan(oof_probs)
        oof_probs_eval, labels_eval = oof_probs[mask], labels_all[mask]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_probs = np.zeros(n)
        oof_assigned = np.zeros(n, dtype=bool)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), labels_all)):
            probs, val_idx_ret = train_one_fold(fold, train_idx, val_idx, train_df, device)
            oof_probs[val_idx_ret] = probs
            oof_assigned[val_idx_ret] = True

        assert oof_assigned.all(), "Some samples were never used as validation in any fold."
        oof_probs_eval, labels_eval = oof_probs, labels_all

    oof_df = train_df.copy()
    oof_df['oof_prob_class1'] = oof_probs
    oof_df.to_csv('oof_predictions.csv', index=False)

    best_thresh, best_bacc = tune_threshold(oof_probs_eval, labels_eval)
    default_bacc = balanced_accuracy_score(labels_eval, (oof_probs_eval >= 0.5).astype(int))

    print(f"\n=== Out-of-fold estimate across {len(labels_eval)} training samples ===")
    print(f"Balanced acc @ 0.5 threshold:        {default_bacc:.4f}")
    print(f"Balanced acc @ tuned threshold {best_thresh:.2f}: {best_bacc:.4f}")
    
    with open('best_threshold.json', 'w') as f:
        json.dump({"threshold": best_thresh}, f)
    print("Saved tuned threshold to best_threshold.json (used automatically by inference.py)")

if __name__ == '__main__':
    train()