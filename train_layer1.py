import os
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Fix seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Shape: (128, 128) -> (1, 128, 128)
        spec = torch.FloatTensor(self.spectrograms[idx]).unsqueeze(0)
        label = torch.FloatTensor([self.labels[idx]])
        return spec, label

class PhaseGuardL1(nn.Module):
    def __init__(self):
        super().__init__()
        
        try:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        except Exception:
            try:
                self.backbone = models.mobilenet_v3_small(pretrained=True)
            except Exception:
                self.backbone = models.mobilenet_v3_small(pretrained=False)
                
        # Grayscale input (1 channel)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Modify classifier to output 1 dimension
        self.backbone.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        self.backbone.classifier[3] = nn.Linear(1024, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    # Load dataset
    data_path = "data/processed"
    spec_file = os.path.join(data_path, "spectrograms.npy")
    label_file = os.path.join(data_path, "labels.npy")
    
    if not (os.path.exists(spec_file) and os.path.exists(label_file)):
        print(f"ERROR: Dataset files not found at {data_path}. Run build_dataset.py first!")
        return
        
    spectrograms = np.load(spec_file)
    labels = np.load(label_file)
    
    print(f"Loaded {len(labels)} samples.")
    print(f"Real (0): {np.sum(labels == 0)}, Fake (1): {np.sum(labels == 1)}")
    
    # Shuffle indices and split (80% train, 20% test)
    indices = np.random.permutation(len(labels))
    split = int(0.8 * len(labels))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    # Save the splits in data/processed for diagnostic utility files
    np.save(os.path.join(data_path, "train_specs.npy"), spectrograms[train_idx])
    np.save(os.path.join(data_path, "train_labels.npy"), labels[train_idx])
    np.save(os.path.join(data_path, "test_specs.npy"), spectrograms[test_idx])
    np.save(os.path.join(data_path, "test_labels.npy"), labels[test_idx])
    
    train_dataset = AudioDataset(spectrograms[train_idx], labels[train_idx])
    test_dataset = AudioDataset(spectrograms[test_idx], labels[test_idx])
    
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = PhaseGuardL1().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    
    epochs = 10
    print("\nStarting Layer 1 Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_specs, batch_labels in train_loader:
            batch_specs = batch_specs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_specs)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted_classes = (predictions > 0.5).float()
            correct += (predicted_classes == batch_labels).sum().item()
            total += len(batch_labels)
            
        train_accuracy = (correct / total) * 100
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_specs, batch_labels in test_loader:
                batch_specs = batch_specs.to(device)
                batch_labels = batch_labels.to(device)
                
                predictions = model(batch_specs)
                loss = criterion(predictions, batch_labels)
                test_loss += loss.item()
                
                predicted_classes = (predictions > 0.5).float()
                test_correct += (predicted_classes == batch_labels).sum().item()
                test_total += len(batch_labels)
                
        test_accuracy = (test_correct / test_total) * 100
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.1f}% | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.1f}%")
            
    # Save the trained model weights
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/layer1_mobilenet.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved successfully to: {model_save_path}")

if __name__ == "__main__":
    train()
