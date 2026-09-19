import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Azimuth-direction convention (ASSUMPTION -- see note below):
#   0 deg   = "up" in the image (toward row 0)
#   90 deg  = "right"
#   180 deg = "down"
#   270 deg = "left"
#   increasing CLOCKWISE (standard compass-style convention)
#
# If accuracy doesn't improve with the gradient channel enabled, the single
# highest-value thing to try next is flipping this convention (e.g. swap the
# sign of dx, or use counter-clockwise instead) -- we have no ground truth
# to verify which one this specific data generator used, so it's worth an
# empirical A/B test. Everything else in the pipeline is convention-agnostic.
# ---------------------------------------------------------------------------

def azimuth_to_direction(azimuth_deg: float):
    theta = math.radians(azimuth_deg)
    dx = math.sin(theta)
    dy = -math.cos(theta)
    return dx, dy


def sun_aligned_gradient_map(img_gray: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """
    Directional derivative of image intensity projected onto the sun
    direction. For a Lambertian surface, the SIGN of this quantity flips
    between convex (rise) and concave (depression) features under the same
    lighting -- which is exactly the ambiguity a human eye (and a naive CNN)
    falls for. Feeding this explicitly as a channel gives the network a much
    more direct signal than hoping it infers the relationship between a
    scalar azimuth value and raw pixels on its own.
    """
    img_f = img_gray.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=5)

    dx, dy = azimuth_to_direction(azimuth_deg)
    d = gx * dx + gy * dy

    std = d.std() + 1e-6
    d = d / std
    d = np.clip(d, -5.0, 5.0)
    return d.astype(np.float32)


class LunarDataset(Dataset):
    """
    Images are oblique horizon-view shots (fixed sky-up/ground-down
    orientation regardless of sun_azimuth_angle) -- confirmed by inspection,
    so the raw image is NEVER rotated as a whole (that would destroy the
    horizon).

    Two things ARE derived from sun_azimuth_angle instead:
      1. A sun-aligned directional-gradient map (channel 2 of the input),
         computed fresh (see sun_aligned_gradient_map above).
      2. A sin/cos encoding of azimuth, fed to the classifier head alongside
         the pooled image features (late fusion, cheap extra signal).

    Augmentation: horizontal flip only (safe -- doesn't change sky-up/
    ground-down). Applied to the raw grayscale array BEFORE computing the
    gradient map, and the azimuth is updated to (360 - azimuth) % 360 first,
    so channel 2 and the sin/cos feature stay internally consistent with
    whatever the (possibly mirrored) image actually shows. No vertical flip
    -- that would be physically invalid (see earlier notes).
    """
    def __init__(self, csv_file, img_dir, is_train=True, augment=False):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        img_path = os.path.join(self.img_dir, row['image_id'])
        img = np.array(Image.open(img_path).convert('L'))  # H x W uint8

        azimuth_deg = float(row['sun_azimuth_angle'])

        if self.is_train and self.augment and torch.rand(1).item() > 0.5:
            img = np.ascontiguousarray(np.fliplr(img))
            azimuth_deg = (360.0 - azimuth_deg) % 360.0

        gray_norm = (img.astype(np.float32) / 255.0 - 0.5) / 0.5  # [-1, 1]
        grad_map = sun_aligned_gradient_map(img, azimuth_deg)      # already ~[-5, 5]

        stacked = np.stack([gray_norm, grad_map], axis=0)  # (2, H, W)
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
