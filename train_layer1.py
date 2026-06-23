import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# ==========================================
# STEP 1: PyTorch Dataset Class
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Add a channel dimension: (128, 128) -> (1, 128, 128)
        spec = torch.FloatTensor(self.spectrograms[idx]).unsqueeze(0)
        label = torch.FloatTensor([self.labels[idx]])
        return spec, label

# ==========================================
# STEP 2: MobileNetV3 Custom Model
# ==========================================
class PhaseGuardL1(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load MobileNetV3 Small with fallbacks for older/newer torchvision versions
        try:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        except Exception:
            try:
                self.backbone = models.mobilenet_v3_small(pretrained=True)
            except Exception:
                self.backbone = models.mobilenet_v3_small(pretrained=False) # Fallback to un-pretrained if offline
                
        # Modify the first conv layer to accept 1 channel (grayscale spectrogram) instead of 3 (RGB)
        # MobileNetV3 Small features: features[0][0] is the Conv2d
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Modify the classification head to output 1 dimension (probability of being Fake/AI)
        # MobileNetV3 Small classifier is: Sequential(Linear(576, 1024), Hardswish(), Dropout(0.2), Linear(1024, num_classes))
        # Increase dropout to 0.5 to prevent overfitting
        self.backbone.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        self.backbone.classifier[3] = nn.Linear(1024, 1)
        
    def forward(self, x):
        # Forward pass through MobileNetV3
        out = self.backbone(x)
        # Squeeze output to [0, 1] probability range
        return torch.sigmoid(out)

# ==========================================
# STEP 3: Training Function
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    # Load dataset
    data_path = "data/processed"
    train_spec_file = os.path.join(data_path, "train_specs.npy")
    train_label_file = os.path.join(data_path, "train_labels.npy")
    test_spec_file = os.path.join(data_path, "test_specs.npy")
    test_label_file = os.path.join(data_path, "test_labels.npy")
    
    spec_file = os.path.join(data_path, "spectrograms.npy")
    label_file = os.path.join(data_path, "labels.npy")
    
    if os.path.exists(train_spec_file) and os.path.exists(train_label_file) and os.path.exists(test_spec_file) and os.path.exists(test_label_file):
        print("Found speaker-aware pre-split dataset files. Loading them directly...")
        train_specs = np.load(train_spec_file)
        train_labels = np.load(train_label_file)
        test_specs = np.load(test_spec_file)
        test_labels = np.load(test_label_file)
        
        print(f"Loaded speaker-aware split:")
        print(f"  Train samples: {len(train_labels)} (Real: {np.sum(train_labels == 0)}, Fake: {np.sum(train_labels == 1)})")
        print(f"  Test samples:  {len(test_labels)} (Real: {np.sum(test_labels == 0)}, Fake: {np.sum(test_labels == 1)})")
        
        train_dataset = AudioDataset(train_specs, train_labels)
        test_dataset = AudioDataset(test_specs, test_labels)
    elif os.path.exists(spec_file) and os.path.exists(label_file):
        print("Found unified dataset files. Loading and splitting randomly (80% train, 20% test)...")
        spectrograms = np.load(spec_file)
        labels = np.load(label_file)
        
        print(f"Loaded {len(labels)} samples.")
        print(f"Real (0): {np.sum(labels == 0)}, Fake (1): {np.sum(labels == 1)}")
        
        # Shuffle indices and split (80% train, 20% test)
        indices = np.random.permutation(len(labels))
        split = int(0.8 * len(labels))
        
        train_idx = indices[:split]
        test_idx = indices[split:]
        
        train_dataset = AudioDataset(spectrograms[train_idx], labels[train_idx])
        test_dataset = AudioDataset(spectrograms[test_idx], labels[test_idx])
    else:
        print(f"ERROR: Dataset files not found at {data_path}. Run build_dataset.py or process_phase1.py first!")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model, loss criterion, and optimizer
    model = PhaseGuardL1().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    
    epochs = 5
    print("\nStarting Training Layer 1...")
    
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
            
            # Compute accuracy
            predicted_classes = (predictions > 0.5).float()
            correct += (predicted_classes == batch_labels).sum().item()
            total += len(batch_labels)
            
        train_accuracy = (correct / total) * 100
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
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
