import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class LunarDataset(Dataset):
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
        image = Image.open(img_path).convert('L') 

        # --- THE FIX: Align shadows to a canonical Top/0° direction ---
        sun_angle = row['sun_azimuth_angle']
        image = TF.rotate(image, -sun_angle, fill=0)

        # Base transformations
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        # Lightweight augmentation
        if self.is_train and self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)

        if self.is_train:
            label = torch.tensor(row['label'], dtype=torch.long)
            return image, label
        else:
            return image, row['image_id']