import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model


def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_val_probs(model, val_loader, device):
    """Returns (probs_class1, true_labels) for the whole val set."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, azimuth_feat, labels in val_loader:
            images, azimuth_feat = images.to(device), azimuth_feat.to(device)
            logits = model(images, azimuth_feat)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def tune_threshold(probs: np.ndarray, labels: np.ndarray):
    """Sweep the decision threshold on P(class 1) to maximize balanced accuracy."""
    best_thresh, best_bacc = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        b = balanced_accuracy_score(labels, preds)
        if b > best_bacc:
            best_bacc, best_thresh = b, t
    return float(best_thresh), float(best_bacc)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df = pd.read_csv('data/train_metadata.csv')
    indices = list(range(len(train_df)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=train_df['label'].values
    )

    # Two separate dataset instances -- never share one object between
    # train/val Subsets (mutating .augment on one would mutate both).
    train_full = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=True)
    val_full = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=False)

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    train_labels = train_df['label'].values[train_idx]
    sampler = build_sampler(train_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_lunar_model(backbone_name="resnet34", pretrained=True).to(device)

    # Sampler already balances classes -- plain (unweighted) loss avoids
    # double-correcting, which is what caused the class-collapse swings
    # you saw between the vscode/colab runs.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 30
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_b_acc = -1.0
    patience, no_improve = 8, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, azimuth_feat, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, azimuth_feat, labels = images.to(device), azimuth_feat.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, azimuth_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        probs, labels_np = get_val_probs(model, val_loader, device)
        preds_default = (probs >= 0.5).astype(int)
        b_acc_default = balanced_accuracy_score(labels_np, preds_default)

        print(f"Loss: {train_loss/len(train_loader):.4f} | Val Balanced Acc (thresh=0.5): {b_acc_default:.4f}")
        print(f"  Val pred distribution -> 0: {(preds_default==0).sum()}, 1: {(preds_default==1).sum()}"
              f"  | true -> 0: {(labels_np==0).sum()}, 1: {(labels_np==1).sum()}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        if b_acc_default > best_b_acc:
            best_b_acc = b_acc_default
            no_improve = 0
            torch.save(model.state_dict(), 'best_lunar_model.pth')
            print("New best model saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    # --- Threshold tuning on the best checkpoint ---
    model.load_state_dict(torch.load('best_lunar_model.pth', map_location=device, weights_only=True))
    probs, labels_np = get_val_probs(model, val_loader, device)
    best_thresh, best_bacc_tuned = tune_threshold(probs, labels_np)
    print(f"\nBest val balanced acc @ default 0.5 threshold: {best_b_acc:.4f}")
    print(f"Best val balanced acc @ tuned threshold {best_thresh:.2f}: {best_bacc_tuned:.4f}")

    with open('best_threshold.json', 'w') as f:
        json.dump({"threshold": best_thresh}, f)
    print("Saved tuned threshold to best_threshold.json (used automatically by inference.py)")


if __name__ == '__main__':
    train()
