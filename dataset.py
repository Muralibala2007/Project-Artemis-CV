import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

def azimuth_to_direction(azimuth_deg: float):
    theta = math.radians(azimuth_deg)
    # INVERTED BOTH SIGNS: This fixes the topographic-inversion illusion 
    # that caused the massive class imbalance in the previous run.
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