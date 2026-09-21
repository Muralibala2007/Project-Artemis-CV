import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model

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