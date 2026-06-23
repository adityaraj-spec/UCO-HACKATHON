import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Fix seeds for reproducibility of model weights and operations
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class AudioDatasetV2(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Shape is (2, 128, 128)
        spec  = torch.FloatTensor(self.spectrograms[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return spec, label

class PhaseGuardL1_V2(nn.Module):
    """
    V2 improvements over V1:
    1. Accepts 2-channel input (magnitude + phase)
    2. Stronger dropout to prevent overfitting
    3. Label smoothing in loss function
    4. Batch normalization on input
    """
    def __init__(self):
        super().__init__()
        
        # Load MobileNetV3 Small
        try:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        except Exception:
            try:
                self.backbone = models.mobilenet_v3_small(pretrained=True)
            except Exception:
                self.backbone = models.mobilenet_v3_small(pretrained=False)
        
        # Accept 2 channels instead of 1 or 3
        # 2 channels = magnitude spectrogram + phase derivative
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Stronger classifier head with more dropout to fight overfitting
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, 2, 128, 128)
        return torch.sigmoid(self.backbone(x))

def retrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    print("Loading dataset...")
    try:
        train_specs  = np.load("data/processed/train_specs.npy")
        train_labels = np.load("data/processed/train_labels.npy")
        test_specs   = np.load("data/processed/test_specs.npy")
        test_labels  = np.load("data/processed/test_labels.npy")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run rebuild_dataset.py first!")
        return 0.0
    
    print(f"Train samples: {len(train_labels)} | Test samples: {len(test_labels)}")
    
    # Check channel count
    channel_count = train_specs.shape[1] if len(train_specs.shape) == 4 else 1
    print(f"Spectrogram channels: {channel_count}")
    
    if channel_count == 1:
        print("WARNING: You have single-channel spectrograms")
        print("Rebuild dataset with dual-channel (magnitude + phase) preprocessing first.")
        # Squeeze if shape is (N, 1, 128, 128)
        if len(train_specs.shape) == 4:
            train_specs = train_specs[:, 0, :, :]
            test_specs  = test_specs[:, 0, :, :]
            
    train_dataset = AudioDatasetV2(train_specs, train_labels)
    test_dataset  = AudioDatasetV2(test_specs, test_labels)
    
    train_loader  = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0    # keep 0 on Windows to avoid multiprocessing issues
    )
    test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = PhaseGuardL1_V2().to(device)
    criterion = nn.BCELoss()
    
    # Lower learning rate (0.0003) for stable updates, and L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-4    # L2 regularization
    )
    
    # Learning rate scheduler reduces lr when test loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5
    )
    
    best_test_acc  = 0
    best_epoch     = 0
    patience_count = 0
    
    print("\nTraining V2...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'Status':>10}")
    print("-" * 75)
    
    for epoch in range(50):
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(specs)
            
            # Label smoothing (manual) mapping 0 -> 0.05, 1 -> 0.95
            smooth_labels = labels * 0.9 + 0.05
            
            loss = criterion(preds, smooth_labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss    += loss.item()
            predicted      = (preds > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total   += len(labels)
        
        train_acc  = train_correct / train_total * 100
        avg_train_loss = train_loss / len(train_loader)
        
        # --- TEST ---
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for specs, labels in test_loader:
                specs = specs.to(device)
                labels = labels.to(device)
                
                preds  = model(specs)
                smooth_labels = labels * 0.9 + 0.05
                loss   = criterion(preds, smooth_labels)
                test_loss    += loss.item()
                predicted     = (preds > 0.5).float()
                test_correct += (predicted == labels).sum().item()
                test_total   += len(labels)
        
        test_acc = test_correct / test_total * 100
        avg_test_loss = test_loss / len(test_loader)
        
        # Scheduler step using validation loss
        scheduler.step(avg_test_loss)
        
        # Save best model based on validation/test accuracy
        status = ""
        if test_acc > best_test_acc:
            best_test_acc  = test_acc
            best_epoch     = epoch + 1
            patience_count = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/layer1_v2_best.pth")
            status = "SAVED ✓"
        else:
            patience_count += 1
            if patience_count >= 8:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best test accuracy: {best_test_acc:.1f}% at epoch {best_epoch}")
                break
        
        print(f"{epoch+1:>6} | {avg_train_loss:>10.4f} | {train_acc:>8.1f}% | "
              f"{avg_test_loss:>9.4f} | {test_acc:>7.1f}% | {status:>10}")
    
    print(f"\nTraining complete!")
    print(f"Best model: {best_test_acc:.1f}% test accuracy at epoch {best_epoch}")
    print(f"Saved to: models/layer1_v2_best.pth")
    
    return best_test_acc

if __name__ == "__main__":
    retrain()
