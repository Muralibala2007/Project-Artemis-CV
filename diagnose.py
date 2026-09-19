import json
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

from dataset import LunarDataset
from model import get_lunar_model


def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv('data/train_metadata.csv')
    indices = list(range(len(train_df)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=train_df['label'].values
    )

    val_full = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=False)
    val_dataset = Subset(val_full, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_lunar_model(backbone_name="resnet34", pretrained=False).to(device)
    model.load_state_dict(torch.load('best_lunar_model.pth', map_location=device, weights_only=True))
    model.eval()

    threshold = 0.5
    if os.path.exists('best_threshold.json'):
        with open('best_threshold.json') as f:
            threshold = json.load(f)["threshold"]

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, azimuth_feat, labels in val_loader:
            images, azimuth_feat = images.to(device), azimuth_feat.to(device)
            logits = model(images, azimuth_feat)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)

    print("=== Validation set (held-out, same split used in training) ===")
    print(f"n = {len(all_labels)}  |  threshold used: {threshold:.2f}")
    print(f"True label distribution:  0: {(all_labels==0).sum()}, 1: {(all_labels==1).sum()}")
    print(f"Pred label distribution:  0: {(all_preds==0).sum()}, 1: {(all_preds==1).sum()}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))
    print()
    print(classification_report(all_labels, all_preds, target_names=["Depth(0)", "Rise(1)"]))
    print(f"Balanced accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f}")
    print()
    conf = np.where(all_preds == 1, all_probs, 1 - all_probs)
    print(f"Mean prediction confidence: {conf.mean():.4f}")
    print(f"Min/Max confidence: {conf.min():.4f} / {conf.max():.4f}")


if __name__ == '__main__':
    diagnose()
