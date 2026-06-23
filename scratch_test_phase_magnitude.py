import librosa
import numpy as np
from preprocess import load_and_standardize

def analyze_phase_features(filepath):
    y = load_and_standardize(filepath)
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # 1. 1st order phase difference wrapped
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    
    # 2. 2nd order phase difference wrapped
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    
    # Define a mask for high-energy (voiced) regions
    # E.g., bins that are above 10th percentile of log magnitude or 10% of max magnitude
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    voiced_mask = mag_db > -30  # within 30 dB of peak energy
    
    # Align masks to diff sizes
    voiced_mask_1 = voiced_mask[:, 1:]
    voiced_mask_2 = voiced_mask[:, 2:]
    
    # Phase Jumps (unwrapped, original)
    pjr_orig = np.sum(np.abs(phase_diff) > np.pi * 0.5) / phase_diff.size
    
    # Phase Jumps in voiced regions only
    pjr_voiced_orig = np.sum((np.abs(phase_diff) > np.pi * 0.5) & voiced_mask_1) / max(1, np.sum(voiced_mask_1))
    
    # 2nd order Phase Jumps in voiced regions
    pjr_voiced_2nd = np.sum((np.abs(phase_diff2_wrapped) > np.pi * 0.5) & voiced_mask_2) / max(1, np.sum(voiced_mask_2))
    
    # Standard deviation of 2nd order wrapped phase difference in voiced regions
    if np.sum(voiced_mask_2) > 0:
        std_voiced_2nd = np.std(phase_diff2_wrapped[voiced_mask_2])
        mean_abs_voiced_2nd = np.mean(np.abs(phase_diff2_wrapped[voiced_mask_2]))
    else:
        std_voiced_2nd = 0.0
        mean_abs_voiced_2nd = 0.0
        
    return {
        'pjr_orig': pjr_orig,
        'pjr_voiced_orig': pjr_voiced_orig,
        'pjr_voiced_2nd': pjr_voiced_2nd,
        'std_voiced_2nd': std_voiced_2nd,
        'mean_abs_voiced_2nd': mean_abs_voiced_2nd
    }

for name in ["vishal", "abhinav", "aditya", "dhruv"]:
    real_path = f"data/real_voices/{name}/{name}_001.wav"
    clone_path = f"data/clones/{name}/{name}_clone_001.wav"
    
    print(f"=== Speaker: {name} ===")
    r_feats = analyze_phase_features(real_path)
    c_feats = analyze_features = analyze_phase_features(clone_path)
    
    print("Real features:")
    for k, v in r_feats.items():
        print(f"  {k}: {v:.4f}")
    print("Clone features:")
    for k, v in c_feats.items():
        print(f"  {k}: {v:.4f}")
    print()
