import os
import csv
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import zoom

# Fix seeds for reproducibility of speaker splits and operations
random.seed(42)
np.random.seed(42)

def build_manifest():
    rows = []  # each row = [filepath, label(0/1), speaker_id, source_name]
    
    # ---- Walk data/sources/real (Label = 0) ----
    real_sources_dir = "data/sources/real"
    if os.path.exists(real_sources_dir):
        for root, dirs, files in os.walk(real_sources_dir):
            for f in files:
                if f.lower().endswith(('.wav', '.flac')):
                    filepath = os.path.join(root, f)
                    if f.startswith("cv_"):
                        parts = f.split('_')
                        speaker_id = parts[1] if len(parts) > 1 else f.split('.')[0]
                        source = "common_voice"
                    elif f.startswith("svarah_"):
                        parts = f.split('_')
                        speaker_id = parts[1] if len(parts) > 1 else f.split('.')[0]
                        source = "svarah"
                    elif f.startswith(("la_", "pa_")):
                        parts = f.split('_')
                        speaker_id = f"{parts[3]}_{parts[4]}"
                        source = parts[0]
                    elif f.startswith(("vishal_", "abhinav_", "aditya_", "dhruv_")):
                        parts = f.split('_')
                        speaker_id = f"sim_{parts[0]}"
                        source = "simulated"
                    else:
                        speaker_id = "ljspeech_linda"
                        source = "ljspeech"
                    rows.append([filepath, 0, f"cv_{speaker_id}" if source == "common_voice" else f"svarah_{speaker_id}" if source == "svarah" else speaker_id, source])
    
    # ---- Walk data/sources/fake (Label = 1) ----
    fake_sources_dir = "data/sources/fake"
    if os.path.exists(fake_sources_dir):
        for root, dirs, files in os.walk(fake_sources_dir):
            for f in files:
                if f.lower().endswith(('.wav', '.flac')):
                    filepath = os.path.join(root, f)
                    if f.startswith("wf_"):
                        parts = f.split('_')
                        vocoder = parts[1] if len(parts) > 1 else "vocoder"
                        speaker_id = f"wavefake_{vocoder}"
                        source = f"wavefake_{vocoder}"
                    elif f.startswith(("la_", "pa_")):
                        parts = f.split('_')
                        speaker_id = f"{parts[3]}_{parts[4]}"
                        source = parts[0]
                    elif f.startswith(("vishal_", "abhinav_", "aditya_", "dhruv_")):
                        parts = f.split('_')
                        speaker_id = f"sim_{parts[0]}"
                        source = "simulated"
                    else:
                        speaker_id = "wavefake_linda"
                        source = "wavefake"
                    rows.append([filepath, 1, speaker_id, source])
    
    # Save manifest as CSV
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/manifest.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "speaker_id", "source"])
        writer.writerows(rows)
    
    print(f"Manifest built: {len(rows)} total files")
    print(f"  Real (0): {sum(1 for r in rows if r[1]==0)}")
    print(f"  Fake (1): {sum(1 for r in rows if r[1]==1)}")
    print(f"  Unique speakers: {len(set(r[2] for r in rows))}")
    
    return rows

def speaker_split(rows, test_fraction=0.2):
    # Get unique speakers
    speakers = list(set(r[2] for r in rows))
    random.shuffle(speakers)
    
    split_point = int(len(speakers) * (1 - test_fraction))
    train_speakers = set(speakers[:split_point])
    test_speakers = set(speakers[split_point:])
    
    train_rows = [r for r in rows if r[2] in train_speakers]
    test_rows = [r for r in rows if r[2] in test_speakers]
    
    print(f"\nTrain split: {len(train_rows)} files from {len(train_speakers)} speakers")
    print(f"Test split:  {len(test_rows)} files from {len(test_speakers)} speakers")
    
    return train_rows, test_rows

def load_and_standardize(file_path, target_sr=16000, duration=2.0):
    try:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
    except Exception:
        return None
    target_length = int(target_sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    y = y[:target_length]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y

def audio_to_dual_channel_spectrogram(y, sr=16000):
    """
    Creates a 2-channel spectrogram:
    Channel 1 = magnitude (normalized + CMVN) — strips speaker characteristics
    Channel 2 = phase derivative (normalized) — captures phase jump rate anomalies
    """
    # STFT gives complex numbers — magnitude AND phase together
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    
    # Channel 1: Magnitude mel-spectrogram
    magnitude = np.abs(stft)
    mel_filters = librosa.filters.mel(sr=sr, n_fft=512, n_mels=128)
    mel = np.dot(mel_filters, magnitude)
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    
    # CMVN on magnitude to remove speaker biases
    mean = mel_db.mean(axis=1, keepdims=True)
    std  = mel_db.std(axis=1, keepdims=True) + 1e-6
    channel1 = (mel_db - mean) / std
    
    # Channel 2: Phase derivative
    phase = np.angle(stft)
    
    # Instantaneous frequency / phase derivative
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.pad(phase_diff, ((0,0),(0,1)), mode='edge')
    phase_magnitude = np.abs(phase_diff)
    
    # Map phase differences using the same mel filterbank
    channel2 = np.dot(mel_filters, phase_magnitude)
    
    # Resize both channels to 128x128
    target_cols = 128
    c1 = zoom(channel1, (1.0, target_cols / channel1.shape[1]))
    c2 = zoom(channel2, (1.0, target_cols / channel2.shape[1]))
    
    # Normalize each channel to 0-1
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)
    
    c1 = norm(c1)
    c2 = norm(c2)
    
    return np.stack([c1, c2], axis=0)

def process_rows(rows, name):
    specs, labels = [], []
    for i, (filepath, label, speaker, source) in enumerate(rows):
        if (i+1) % 200 == 0 or (i+1) == len(rows):
            print(f"  [{name}] processed {i+1}/{len(rows)} files...")
        audio = load_and_standardize(filepath)
        if audio is None:
            continue
        specs.append(audio_to_dual_channel_spectrogram(audio))
        labels.append(label)
    return np.array(specs, dtype=np.float32), np.array(labels, dtype=np.int64)

if __name__ == "__main__":
    rows = build_manifest()
    if len(rows) > 0:
        train_rows, test_rows = speaker_split(rows)
        
        print("\nProcessing TRAIN set...")
        train_specs, train_labels = process_rows(train_rows, "train")
        
        print("\nProcessing TEST set...")
        test_specs, test_labels = process_rows(test_rows, "test")
        
        # Save output arrays
        output_folder = "data/processed"
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, "train_specs.npy"), train_specs)
        np.save(os.path.join(output_folder, "train_labels.npy"), train_labels)
        np.save(os.path.join(output_folder, "test_specs.npy"), test_specs)
        np.save(os.path.join(output_folder, "test_labels.npy"), test_labels)
        
        print(f"\nSaved to {output_folder}/")
        print(f"Spectrogram shape: {train_specs[0].shape}")  # should be (2, 128, 128)
        print(f"Train samples: {len(train_specs)}, Test samples: {len(test_specs)}")
    else:
        print("No files found. Populate data folders first.")
