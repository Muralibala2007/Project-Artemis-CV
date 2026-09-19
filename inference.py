import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model


def generate_submission():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = LunarDataset('data/test_metadata.csv', 'data/test_images/', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_lunar_model(backbone_name="resnet34", pretrained=False).to(device)
    model.load_state_dict(torch.load('best_lunar_model.pth', map_location=device, weights_only=True))
    model.eval()

    threshold = 0.5
    if os.path.exists('best_threshold.json'):
        with open('best_threshold.json') as f:
            threshold = json.load(f)["threshold"]
        print(f"Using tuned threshold: {threshold:.2f}")
    else:
        print("No best_threshold.json found -- using default 0.5. Run train.py to generate it.")

    predictions, image_ids = [], []

    with torch.no_grad():
        for images, azimuth_feat, img_ids in tqdm(test_loader, desc="Predicting Test Set"):
            images, azimuth_feat = images.to(device), azimuth_feat.to(device)

            logits = model(images, azimuth_feat)
            probs = F.softmax(logits, dim=1)[:, 1]

            # TTA: horizontal flip is safe (doesn't change sky-up/ground-down).
            # channel 0 (grayscale) and channel 1 (gradient map) both flip
            # left-right together; azimuth_feat was computed at dataset load
            # time for the UN-flipped image, so we approximate the flipped
            # azimuth's sin/cos on the fly here rather than re-reading the CSV.
            images_flipped = torch.flip(images, dims=[3])
            sin_a, cos_a = azimuth_feat[:, 0], azimuth_feat[:, 1]
            # azimuth_new = (360 - azimuth) deg  =>  sin(-x)=-sin(x), cos(-x)=cos(x)
            azimuth_feat_flipped = torch.stack([-sin_a, cos_a], dim=1)

            logits_flip = model(images_flipped, azimuth_feat_flipped)
            probs_flip = F.softmax(logits_flip, dim=1)[:, 1]

            probs_avg = (probs + probs_flip) / 2.0
            preds = (probs_avg >= threshold).long()

            predictions.extend(preds.cpu().numpy())
            image_ids.extend(img_ids)

    submission_df = pd.DataFrame({'image_id': image_ids, 'label': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("submission.csv successfully generated!")
    print(submission_df['label'].value_counts())


if __name__ == '__main__':
    generate_submission()
