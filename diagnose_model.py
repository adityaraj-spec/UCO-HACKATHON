import torch
import numpy as np
import os
from train_layer1 import PhaseGuardL1  # your existing model file

# ============================================================
# CHECK 1: What does the model output on your test data?
# ============================================================

def check_model_outputs():
    print("="*55)
    print("CHECK 1: Model output distribution")
    print("="*55)
    
    # Load your saved model
    model = PhaseGuardL1()
    model_path = "models/layer1_mobilenet.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Change model_path to your actual .pth file location")
        return
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load test spectrograms
    try:
        test_specs   = np.load("data/processed/test_specs.npy")
        test_labels  = np.load("data/processed/test_labels.npy")
    except:
        try:
            test_specs  = np.load("data/processed/spectrograms.npy")
            test_labels = np.load("data/processed/labels.npy")
        except:
            print("ERROR: Cannot find test data .npy files")
            print("Check your data/processed/ folder")
            return
    
    print(f"Test samples: {len(test_labels)}")
    print(f"Real (0): {sum(test_labels==0)}, Fake (1): {sum(test_labels==1)}")
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for i in range(len(test_specs)):
            spec = torch.FloatTensor(test_specs[i]).unsqueeze(0).unsqueeze(0)
            prob = float(model(spec)[0][0])
            pred = 1 if prob > 0.5 else 0
            all_probs.append(prob)
            all_preds.append(pred)
    
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(test_labels)
    
    # What are the actual output probabilities?
    print(f"\nModel output probabilities:")
    print(f"  Min  : {all_probs.min():.4f}")
    print(f"  Max  : {all_probs.max():.4f}")
    print(f"  Mean : {all_probs.mean():.4f}")
    print(f"  Std  : {all_probs.std():.4f}")
    
    accuracy = np.mean(all_preds == all_labels) * 100
    
    # What would random guessing give?
    majority_class = max(sum(all_labels==0), sum(all_labels==1))
    random_baseline = majority_class / len(all_labels) * 100
    
    print(f"\nAccuracy       : {accuracy:.1f}%")
    print(f"Random baseline: {random_baseline:.1f}% (always guessing majority class)")
    
    if accuracy <= random_baseline + 2:
        print("\n🔴 DIAGNOSIS: Model learned NOTHING")
        print("   It is performing at or below random chance")
    elif accuracy < 70:
        print("\n🟡 DIAGNOSIS: Model learned something but poorly")
    else:
        print("\n🟢 Model is working — accuracy above random baseline")
    
    # Per-class accuracy
    real_correct = sum((all_preds[all_labels==0]) == 0)
    fake_correct = sum((all_preds[all_labels==1]) == 1)
    real_total   = sum(all_labels==0)
    fake_total   = sum(all_labels==1)
    
    print(f"\nPer-class accuracy:")
    print(f"  Real voices: {real_correct}/{real_total} = {real_correct/real_total*100:.1f}% correct")
    print(f"  Fake voices: {fake_correct}/{fake_total} = {fake_correct/fake_total*100:.1f}% correct")
    
    return all_probs, all_labels


# ============================================================
# CHECK 2: Is your training data actually correct?
# ============================================================

def check_data_integrity():
    print("\n" + "="*55)
    print("CHECK 2: Data integrity")
    print("="*55)
    
    try:
        specs  = np.load("data/processed/spectrograms.npy")
        labels = np.load("data/processed/labels.npy")
    except:
        try:
            specs  = np.load("data/processed/train_specs.npy")
            labels = np.load("data/processed/train_labels.npy")
        except:
            print("Cannot find .npy files")
            return
    
    print(f"Total samples  : {len(labels)}")
    print(f"Real (label=0) : {sum(labels==0)}")
    print(f"Fake (label=1) : {sum(labels==1)}")
    print(f"Balance        : {sum(labels==0)/len(labels)*100:.1f}% real")
    
    if sum(labels==0)/len(labels) > 0.80:
        print("\n🔴 PROBLEM: Dataset heavily imbalanced toward REAL")
        print("   Model learns 'always say real' to get high accuracy")
        print("   FIX: Add more fake samples or undersample real")
    
    elif sum(labels==1)/len(labels) > 0.80:
        print("\n🔴 PROBLEM: Dataset heavily imbalanced toward FAKE")
        print("   Model learns 'always say fake' to get high accuracy")
        print("   FIX: Add more real samples or undersample fake")
    
    else:
        print("\n✓ Balance looks reasonable")
    
    # Check if spectrograms look correct
    print(f"\nSpectrogram shape : {specs.shape}")
    print(f"Value range       : {specs.min():.3f} to {specs.max():.3f}")
    
    if specs.max() > 10:
        print("\n🔴 PROBLEM: Spectrogram values are NOT normalized (range should be 0-1)")
        print("   FIX: Re-run preprocessing with normalization")
    elif specs.min() == specs.max():
        print("\n🔴 PROBLEM: All spectrogram values are identical — data is corrupted")
    else:
        print("✓ Spectrogram values look normalized")
    
    # Check if real and fake spectrograms actually look different
    real_specs = specs[labels==0]
    fake_specs = specs[labels==1]
    
    real_mean = real_specs.mean()
    fake_mean = fake_specs.mean()
    
    print(f"\nReal spectrogram mean : {real_mean:.4f}")
    print(f"Fake spectrogram mean : {fake_mean:.4f}")
    print(f"Difference            : {abs(real_mean-fake_mean):.4f}")
    
    if abs(real_mean - fake_mean) < 0.01:
        print("\n🔴 PROBLEM: Real and fake spectrograms look almost identical")
        print("   The features you're using may not distinguish real from fake")
        print("   This is the core problem — your input data looks the same to the model")
    else:
        print("✓ Real and fake spectrograms have measurable differences")


# ============================================================
# CHECK 3: Did training actually progress?
# (only works if you saved training logs)
# ============================================================

def check_training_happened():
    print("\n" + "="*55)
    print("CHECK 3: Training progression")
    print("="*55)
    print("Look at your training output from when you ran train_layer1.py")
    print("Specifically the loss and accuracy numbers per epoch")
    print("")
    print("HEALTHY training looks like:")
    print("  Epoch 1  | Loss: 0.693 | Acc: 52%")
    print("  Epoch 5  | Loss: 0.581 | Acc: 68%")
    print("  Epoch 10 | Loss: 0.423 | Acc: 79%")
    print("  Epoch 20 | Loss: 0.201 | Acc: 91%")
    print("  → Loss going DOWN, Accuracy going UP = learning happened")
    print("")
    print("BROKEN training looks like:")
    print("  Epoch 1  | Loss: 0.693 | Acc: 50%")
    print("  Epoch 5  | Loss: 0.691 | Acc: 51%")
    print("  Epoch 10 | Loss: 0.692 | Acc: 50%")
    print("  Epoch 20 | Loss: 0.690 | Acc: 50%")
    print("  → Loss STUCK at ~0.693, Accuracy stuck at ~50% = nothing learned")
    print("")
    print("Loss of 0.693 specifically = ln(2) = completely random binary guessing")
    print("If your loss started AND ENDED near 0.693, the model never learned")


if __name__ == "__main__":
    check_model_outputs()
    check_data_integrity()
    check_training_happened()
