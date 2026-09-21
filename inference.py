import os
import math
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
)
from tqdm import tqdm

# ==============================================================================
# 1. DATASET MODULE (Integrated)
# ==============================================================================

def azimuth_to_direction(azimuth_deg: float):
    theta = math.radians(azimuth_deg)
    dx = -math.sin(theta)  
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
# 3. INFERENCE LOOP MODULE
# ==============================================================================

BACKBONE = "resnet34"  

def find_fold_checkpoints():
    return sorted(glob.glob('best_lunar_model_fold*.pth'))

def predict_with_tta(model, images, azimuth_feat, device):
    images, azimuth_feat = images.to(device), azimuth_feat.to(device)
    logits = model(images, azimuth_feat)
    probs = F.softmax(logits, dim=1)[:, 1]

    images_flipped = torch.flip(images, dims=[3])
    sin_a, cos_a = azimuth_feat[:, 0], azimuth_feat[:, 1]
    azimuth_feat_flipped = torch.stack([-sin_a, cos_a], dim=1)

    logits_flip = model(images_flipped, azimuth_feat_flipped)
    probs_flip = F.softmax(logits_flip, dim=1)[:, 1]

    return (probs + probs_flip) / 2.0

def generate_submission():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = LunarDataset('data/test_metadata.csv', 'data/test_images/', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    threshold = 0.5
    if os.path.exists('best_threshold.json'):
        with open('best_threshold.json') as f:
            threshold = json.load(f)['threshold']
        print(f"Using tuned threshold: {threshold:.2f}")
    else:
        print("No best_threshold.json found -- using default 0.5. Run train.py first.")

    ckpts = find_fold_checkpoints()
    if ckpts:
        print(f"Found {len(ckpts)} fold checkpoints -- ensembling: {ckpts}")
    elif os.path.exists('best_lunar_model.pth'):
        print("No fold checkpoints found -- falling back to single best_lunar_model.pth")
        ckpts = ['best_lunar_model.pth']
    else:
        raise FileNotFoundError("No model checkpoints found. Run train.py first.")

    models = []
    for ckpt in ckpts:
        m = get_lunar_model(backbone_name=BACKBONE, pretrained=False, in_channels=3).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    image_ids_all, probs_all = [], []

    with torch.no_grad():
        for images, azimuth_feat, img_ids in tqdm(test_loader, desc="Predicting Test Set"):
            batch_probs = torch.zeros(images.size(0), device=device)
            for m in models:
                batch_probs += predict_with_tta(m, images, azimuth_feat, device)
            batch_probs /= len(models)

            probs_all.extend(batch_probs.cpu().numpy())
            image_ids_all.extend(img_ids)

    probs_all = np.array(probs_all)
    preds = (probs_all >= threshold).astype(int)

    submission_df = pd.DataFrame({'image_id': image_ids_all, 'label': preds})
    submission_df.to_csv('submission.csv', index=False)
    print("submission.csv successfully generated!")
    print(submission_df['label'].value_counts())

if __name__ == '__main__':
    generate_submission()