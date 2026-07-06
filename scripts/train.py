"""
train.py — Trains MobileNetV3 on mel-spectrograms
Saves: models/layer1_mobilenet.pth
Run: python scripts/train.py  (from project root)
"""

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OMP DLL conflict on Windows
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _HERE)

import torch, torch.nn as nn, torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, specs, labels, augment=False):
        self.specs = specs; self.labels = labels
        self.augment = augment
        
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        spec = self.specs[idx].copy()
        if self.augment:
            spec += np.random.normal(0, 0.01, spec.shape)
            spec = np.clip(spec, 0, 1)
            spec = np.roll(spec, np.random.randint(-13,13), axis=1)
        return (torch.FloatTensor(spec).unsqueeze(0),
                torch.FloatTensor([self.labels[idx]]))

class PhaseGuardL1(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.backbone = models.mobilenet_v3_small(weights=None)
        except:
            self.backbone = models.mobilenet_v3_small(pretrained=False)
            
        # Replace first conv: 3-channel RGB -> 1-channel mel
        self.backbone.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            
        # Replace final layer: 1000 classes -> 1 (real/fake)
        self.backbone.classifier[3] = nn.Linear(1024, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    specs = np.load("data/processed/spectrograms.npy")
    labels = np.load("data/processed/labels.npy")
    
    split = int(0.8 * len(labels))
    train_ds = SpectrogramDataset(specs[:split], labels[:split], augment=True)
    test_ds = SpectrogramDataset(specs[split:], labels[split:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = PhaseGuardL1().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5)
        
    os.makedirs("models", exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(1, 26):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward(); optimizer.step()
            
            loss_sum += loss.item()
            correct += ((pred > 0.5).float() == y).sum().item()
            total += len(y)
            
        train_acc = correct / total * 100
        
        model.eval(); tc, tt = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                tc += ((pred > 0.5).float() == y).sum().item()
                tt += len(y)
                
        test_acc = tc / tt * 100
        scheduler.step(test_acc)
        
        print(f"Epoch {epoch:>2}/25 | Loss {loss_sum/len(train_loader):.4f}"
              f" | Train {train_acc:.1f}% | Test {test_acc:.1f}%")
              
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/layer1_mobilenet.pth")
            
    print(f"\nBest accuracy: {best_acc:.1f}% -> models/layer1_mobilenet.pth")

if __name__ == "__main__":
    train()
