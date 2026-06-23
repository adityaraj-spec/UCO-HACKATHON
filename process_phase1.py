"""
process_phase1.py  (v2 — multi-source, speaker-aware)

Folder structure expected:
  data/sources/ljspeech/wavs/*.wav            (1 speaker: "ljspeech")
  data/sources/wavefake_fake/*.wav            (label=fake, speaker="wavefake")
  data/sources/common_voice_hindi/*.wav       (many speakers — use filename or folder as ID)
  data/sources/svarah/*.wav
  data/sources/asvspoof_fake/*.wav
  data/raw_voices/<member>/*.wav              (label=real, speaker=<member>)
  data/clones/<member>/*.wav                  (label=fake, speaker=<member>)
"""

import os
import csv
import random
import numpy as np
import librosa
from scipy.ndimage import zoom

# ============================================================
# STEP 1: BUILD THE MANIFEST
# A manifest is just a table: filepath | label | speaker_id | source
# ============================================================

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


# ============================================================
# STEP 2: SPEAKER-AWARE TRAIN/TEST SPLIT
# Whole speakers go to train OR test, never split across both
# ============================================================

def speaker_split(rows, test_fraction=0.2):
    # Get unique speakers
    speakers = list(set(r[2] for r in rows))
    random.shuffle(speakers)
    
    split_point = int(len(speakers) * (1 - test_fraction))
    train_speakers = set(speakers[:split_point])
    test_speakers = set(speakers[split_point:])
    
    train_rows = [r for r in rows if r[2] in train_speakers]
    test_rows = [r for r in rows if r[2] in test_speakers]
    
    print(f"\nTrain: {len(train_rows)} files from {len(train_speakers)} speakers")
    print(f"Test:  {len(test_rows)} files from {len(test_speakers)} speakers")
    
    return train_rows, test_rows


# ============================================================
# STEP 3: AUDIO PROCESSING
# ============================================================

def load_and_standardize(file_path, target_sr=16000, duration=2.0):
    try:
        import soundfile as sf
        # Use soundfile directly to prevent Windows codec errors
        y, sr = sf.read(file_path)
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

def audio_to_mel_spectrogram(y, sr=16000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=512, hop_length=160)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = zoom(mel_db, (1.0, 128.0 / mel_db.shape[1]))
    mn, mx = mel_resized.min(), mel_resized.max()
    return (mel_resized - mn) / (mx - mn + 1e-10) if mx > mn else np.zeros_like(mel_resized)

def extract_5_signals(y, sr=16000):
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    
    magnitude = np.abs(stft)
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    voiced_mask = mag_db[:96, 2:] > -30
    phase_diff2_wrapped_low = phase_diff2_wrapped[:96, :]
    
    if np.sum(voiced_mask) > 0:
        phase_jump_rate = np.sum((np.abs(phase_diff2_wrapped_low) > np.pi * 0.5) & voiced_mask) / np.sum(voiced_mask)
    else:
        phase_jump_rate = np.sum(np.abs(phase_diff2_wrapped_low) > np.pi * 0.5) / phase_diff2_wrapped_low.size
        
    # Fast pitch jitter using numpy autocorrelation
    frame_len = 1024
    hop_len = 512
    num_frames = (len(y) - frame_len) // hop_len + 1
    f0s = []
    min_lag = int(sr / 400) # 400 Hz
    max_lag = int(sr / 65)  # 65 Hz
    for i in range(num_frames):
        frame = y[i*hop_len : i*hop_len + frame_len]
        if np.std(frame) < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        corr_range = corr[min_lag:max_lag]
        if len(corr_range) == 0:
            continue
        lag = np.argmax(corr_range) + min_lag
        f0 = sr / lag
        f0s.append(f0)
    f0s = np.array(f0s)
    if len(f0s) > 2:
        jitter = float(np.mean(np.abs(np.diff(f0s))) / (np.mean(f0s) + 1e-10))
    else:
        jitter = 0.0
        
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    noise_floor = float(np.percentile(rms, 10))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta_var = float(np.var(librosa.feature.delta(mfcc)))
    
    return [float(phase_jump_rate), jitter, flatness, noise_floor, mfcc_delta_var]



# ============================================================
# STEP 4: PROCESS A SET OF ROWS INTO ARRAYS
# ============================================================

def process_rows(rows, name):
    specs, feats, labels = [], [], []
    for i, (filepath, label, speaker, source) in enumerate(rows):
        if (i+1) % 100 == 0 or (i+1) == len(rows):
            print(f"  [{name}] processed {i+1}/{len(rows)} files...")
        audio = load_and_standardize(filepath)
        if audio is None:
            continue
        specs.append(audio_to_mel_spectrogram(audio))
        feats.append(extract_5_signals(audio))
        labels.append(label)
    return np.array(specs, dtype=np.float32), np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    rows = build_manifest()
    if len(rows) > 0:
        train_rows, test_rows = speaker_split(rows)
        
        print("\nProcessing TRAIN set...")
        train_specs, train_feats, train_labels = process_rows(train_rows, "train")
        
        print("\nProcessing TEST set...")
        test_specs, test_feats, test_labels = process_rows(test_rows, "test")
        
        np.save("data/processed/train_specs.npy", train_specs)
        np.save("data/processed/train_feats.npy", train_feats)
        np.save("data/processed/train_labels.npy", train_labels)
        np.save("data/processed/test_specs.npy", test_specs)
        np.save("data/processed/test_feats.npy", test_feats)
        np.save("data/processed/test_labels.npy", test_labels)
        
        print("\n[OK] Done. Train/test sets saved separately, split by speaker.")
    else:
        print("No files found. Populate data folders first.")
