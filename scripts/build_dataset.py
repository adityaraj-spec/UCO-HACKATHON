"""
build_dataset.py
Outputs:
  data/processed/spectrograms.npy   shape: (N, 128, 128)
  data/processed/labels.npy         shape: (N,)   0=real 1=fake
  data/processed/features.npy       shape: (N, 5)
Run: python scripts/build_dataset.py  (from project root)
"""

import os, sys
import numpy as np

# Ensure scripts/ dir is on path so 'features' can be imported
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # project root (UCO-HACKATHON)
sys.path.insert(0, _HERE)
os.chdir(_ROOT)                   # make all data/ paths resolve from root

from features import extract_all

SOURCES = [
    ("data/raw_voices/vishal", 0, "Vishal real"),
    ("data/raw_voices/abhinav", 0, "Abhinav real"),
    ("data/raw_voices/aditya", 0, "Aditya real"),
    ("data/raw_voices/dhruv", 0, "Dhruv real"),
    ("data/clones/vishal", 1, "Vishal clone"),
    ("data/clones/abhinav", 1, "Abhinav clone"),
    ("data/wavefake/real", 0, "WaveFake real"),
    ("data/wavefake/fake", 1, "WaveFake fake"),
    # New external datasets
    ("data/ljspeech", 0, "LJSpeech real"),
    ("data/common_voice_hindi", 0, "Common Voice Hindi real"),
    ("data/svarah", 0, "Svarah real"),
    ("data/wavefake_fake", 1, "Wavefake Fake clones")
]

AUDIO_EXTS = {'wav', 'flac', 'mp3'}

def process_folder(folder, label, desc):
    specs, feats, labels = [], [], []
    if not os.path.exists(folder):
        print(f"  Skipping (not found): {folder}")
        return specs, feats, labels
        
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.rsplit(".", 1)[-1].lower() in AUDIO_EXTS:
                files.append(os.path.join(root, f))
                
    print(f"  [{desc}] {len(files)} files, label={label}")
    
    for i, filepath in enumerate(files):
        result = extract_all(filepath)
        if result is None: continue
        specs.append(result['mel'])
        feats.append(list(result['signals'].values()))
        labels.append(label)
        if (i+1) % 5 == 0: 
            print(f"    -> {i+1}/{len(files)} done", flush=True)
            
    print(f"  Done: {len(labels)} extracted")
    return specs, feats, labels

def build():
    all_specs, all_feats, all_labels = [], [], []
    for folder, label, desc in SOURCES:
        s, f, l = process_folder(folder, label, desc)
        all_specs += s; all_feats += f; all_labels += l
        
    if not all_labels:
        print("No data found. Check folder paths.")
        return
        
    idx = np.random.permutation(len(all_labels))
    np.save("data/processed/spectrograms.npy", 
            np.array(all_specs, dtype=np.float32)[idx])
    np.save("data/processed/labels.npy", 
            np.array(all_labels, dtype=np.int64)[idx])
    np.save("data/processed/features.npy", 
            np.array(all_feats, dtype=np.float32)[idx])
            
    print(f"Done! {len(all_labels)} samples saved to data/processed/")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    build()
