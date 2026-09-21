import os
import math
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# ==============================================================================
# 1. DATASET & AUGMENTATION MODULE (Integrated)
# ==============================================================================

def azimuth_to_direction(azimuth_deg: float):
    theta = math.radians(azimuth_deg)
    dx = -math.sin(theta)  # Fixed topographic-inversion sign
    dy = math.cos(theta)
    return dx, dy

def sun_aligned_gradient_map(
    img_gray: np.ndarray,
    azimuth_deg: float,
    blur_sigma: float = 0.0,
    ksize: int = 3,
) -> np.ndarray:
    img_f = img_gray.astype(np.float32)
    if blur_sigma > 0:
        img_f = cv2.GaussianBlur(img_f, (0, 0), sigmaX=blur_sigma)

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)

    dx, dy = azimuth_to_direction(azimuth_deg)
    d = gx * dx + gy * dy

    std = d.std() + 1e-6
    d = np.clip(d / std, -5.0, 5.0)
    return d.astype(np.float32)

class LunarDataset(Dataset):
    def __init__(self, csv_file, img_dir, is_train=True, augment=False, rotate_aug_deg=0.0):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        self.augment = augment
        self.rotate_aug_deg = rotate_aug_deg

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        img = np.array(Image.open(img_path).convert('L'))  

        azimuth_deg = float(row['sun_azimuth_angle'])

        if self.is_train and self.augment:
            if torch.rand(1).item() > 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                azimuth_deg = (360.0 - azimuth_deg) % 360.0

            if self.rotate_aug_deg > 0:
                delta = float(np.random.uniform(-self.rotate_aug_deg, self.rotate_aug_deg))
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), delta, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                azimuth_deg = (azimuth_deg - delta) % 360.0

        gray_norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5  
        grad_fine = sun_aligned_gradient_map(img, azimuth_deg, blur_sigma=0.0, ksize=3)
        grad_coarse = sun_aligned_gradient_map(img, azimuth_deg, blur_sigma=4.0, ksize=3)

        stacked = np.stack([gray_norm, grad_fine, grad_coarse], axis=0)  
        image_tensor = torch.from_numpy(stacked).float()

        theta = math.radians(azimuth_deg)
        azimuth_feat = torch.tensor(
            [math.sin(theta), math.cos(theta)], dtype=torch.float32
        )

        if self.is_train:
            label = torch.tensor(row['label'], dtype=torch.long)
            return image_tensor, azimuth_feat, label
        else:
            return image_tensor, azimuth_feat, row['image_id']


# ==============================================================================
# 2. MODEL MODULE (Integrated)
# ==============================================================================

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.num_features = num_features

    def forward(self, x, cond):
        gamma_beta = self.to_gamma_beta(cond)
        gamma = gamma_beta[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights, 128),
    "resnet34": (resnet34, ResNet34_Weights, 128),
    "resnet50": (resnet50, ResNet50_Weights, 512), 
}

class LunarModel(nn.Module):
    def __init__(self, backbone_name="resnet34", pretrained=True, dropout=0.3, in_channels=3):
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        ctor, weights_cls, film_channels = _BACKBONES[backbone_name]

        weights = weights_cls.DEFAULT if pretrained else None
        net = ctor(weights=weights)

        original_conv = net.conv1
        new_conv = nn.Conv2d(
            in_channels, original_conv.out_channels,
            kernel_size=original_conv.kernel_size, stride=original_conv.stride,
            padding=original_conv.padding, bias=False,
        )
        if pretrained:
            avg_w = original_conv.weight.data.mean(dim=1, keepdim=True)  
            new_conv.weight.data[:, 0:1, :, :] = avg_w
        net.conv1 = new_conv

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.feat_dim = net.fc.in_features

        self.film = FiLM(cond_dim=2, num_features=film_channels)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, image, azimuth_feat):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.film(x, azimuth_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        combined = torch.cat([x, azimuth_feat], dim=1)
        return self.classifier(combined)

def get_lunar_model(backbone_name: str = "resnet34", pretrained: bool = True, in_channels: int = 3):
    return LunarModel(backbone_name=backbone_name, pretrained=pretrained, in_channels=in_channels)


# ==============================================================================
# 3. TRAINING LOOP MODULE
# ==============================================================================

N_FOLDS = 5              
EPOCHS_PER_FOLD = 3      
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

    warmup_epochs = 0 
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