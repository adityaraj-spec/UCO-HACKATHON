import os
import numpy as np
import librosa
from scipy.ndimage import zoom
from preprocess import load_and_standardize, audio_to_mel_spectrogram

def build_dataset():
    print("Starting simplified dataset compilation...")
    
    real_dirs = ["data/real_voices", "data/sources/real"]
    fake_dirs = ["data/clones", "data/sources/fake"]
    
    all_specs = []
    all_labels = []
    
    # 1. Process Real files
    print("\nScanning REAL directories recursively...")
    real_count = 0
    for rdir in real_dirs:
        if not os.path.exists(rdir):
            print(f"Skipping missing real directory: {rdir}")
            continue
        print(f"Scanning: {rdir}")
        for root, dirs, files in os.walk(rdir):
            for fname in files:
                if fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    fpath = os.path.join(root, fname)
                    try:
                        audio = load_and_standardize(fpath)
                        if audio is None:
                            continue
                        mel = audio_to_mel_spectrogram(audio)
                        
                        # Resize to 128x128
                        if mel.shape[1] != 128:
                            mel_resized = zoom(mel, (1.0, 128.0 / mel.shape[1]))
                        else:
                            mel_resized = mel
                        
                        # Normalize to 0-1
                        mn, mx = mel_resized.min(), mel_resized.max()
                        mel_norm = (mel_resized - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(mel_resized)
                        
                        all_specs.append(mel_norm)
                        all_labels.append(0)
                        real_count += 1
                        if real_count % 200 == 0:
                            print(f"  Processed {real_count} real files...")
                    except Exception as e:
                        pass
                        
    print(f"Total REAL files processed: {real_count}")
    
    # 2. Process Fake files
    print("\nScanning FAKE directories recursively...")
    fake_count = 0
    for fdir in fake_dirs:
        if not os.path.exists(fdir):
            print(f"Skipping missing fake directory: {fdir}")
            continue
        print(f"Scanning: {fdir}")
        for root, dirs, files in os.walk(fdir):
            for fname in files:
                if fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    fpath = os.path.join(root, fname)
                    try:
                        audio = load_and_standardize(fpath)
                        if audio is None:
                            continue
                        mel = audio_to_mel_spectrogram(audio)
                        
                        # Resize to 128x128
                        if mel.shape[1] != 128:
                            mel_resized = zoom(mel, (1.0, 128.0 / mel.shape[1]))
                        else:
                            mel_resized = mel
                        
                        # Normalize to 0-1
                        mn, mx = mel_resized.min(), mel_resized.max()
                        mel_norm = (mel_resized - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(mel_resized)
                        
                        all_specs.append(mel_norm)
                        all_labels.append(1)
                        fake_count += 1
                        if fake_count % 200 == 0:
                            print(f"  Processed {fake_count} fake files...")
                    except Exception as e:
                        pass
                        
    print(f"Total FAKE files processed: {fake_count}")
    
    if len(all_labels) == 0:
        print("ERROR: No audio samples found!")
        return
        
    spectrograms = np.array(all_specs, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/spectrograms.npy", spectrograms)
    np.save("data/processed/labels.npy", labels)
    
    print("\nDataset compiled successfully!")
    print(f"Total samples: {len(labels)}")
    print(f"Real (0): {real_count} | Fake (1): {fake_count}")
    print(f"Saved to data/processed/spectrograms.npy and labels.npy")
    print(f"Spectrogram shape: {spectrograms.shape}")

if __name__ == "__main__":
    build_dataset()
