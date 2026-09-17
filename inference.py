import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model

def generate_submission():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = LunarDataset('data/test_metadata.csv', 'data/test_images/', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = get_lunar_model().to(device)
    model.load_state_dict(torch.load('best_lunar_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    predictions, image_ids = [], []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predicting Test Set"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(img_ids)
            
    submission_df = pd.DataFrame({'image_id': image_ids, 'label': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("submission.csv successfully generated!")

if __name__ == '__main__':
    generate_submission()