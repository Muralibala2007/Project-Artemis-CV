import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from tqdm import tqdm

from dataset import LunarDataset
from model import get_lunar_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata to calculate class weights for Balanced Accuracy
    train_df = pd.read_csv('data/train_metadata.csv')
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = class_counts.sum()
    class_weights = total_samples / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Datasets
    full_dataset = LunarDataset('data/train_metadata.csv', 'data/train_images/', is_train=True, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_lunar_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 10
    best_b_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = torch.argmax(model(images), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        b_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Loss: {train_loss/len(train_loader):.4f} | Val Balanced Acc: {b_acc:.4f}")

        if b_acc > best_b_acc:
            best_b_acc = b_acc
            torch.save(model.state_dict(), 'best_lunar_model.pth')
            print("New best model saved!")

if __name__ == '__main__':
    train()