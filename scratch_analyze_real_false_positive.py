import os
import librosa
import numpy as np

def analyze_file(filepath):
    print(f"\n==================================================")
    print(f"ANALYSIS FOR: {os.path.basename(filepath)}")
    print(f"==================================================")
    if not os.path.exists(filepath):
        print("File does not exist!")
        return

    # Load audio
    y, sr = librosa.load(filepath, sr=16000)
    print(f"Loaded length: {len(y)} samples ({len(y)/sr:.2f}s) at sr={sr}")
    
    # 1. Energy profile
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    mean_rms = np.mean(rms)
    noise_floor = np.percentile(rms, 10)
    print(f"RMS Mean: {mean_rms:.6f}, Noise Floor (10th percentile): {noise_floor:.6f}")
    
    # 2. Spectral Flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    print(f"Spectral Flatness: {flatness:.6f}")
    
    # 3. Phase Analysis
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    magnitude = np.abs(stft)
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    
    # Let's test PJR at various frequency bounds and magnitude thresholds
    freq_bounds = [1000, 2000, 3000, 4000, 8000]
    mag_thresholds = [-20, -25, -30, -35, -40]
    
    bin_spacing = 16000 / 512 # 31.25 Hz
    
    for f_bound in freq_bounds:
        max_bin = int(f_bound / bin_spacing)
        for m_thresh in mag_thresholds:
            voiced_mask = mag_db[:max_bin, 2:] > m_thresh
            phase_diff2_sliced = phase_diff2_wrapped[:max_bin, :]
            
            if np.sum(voiced_mask) > 0:
                pjr = np.sum((np.abs(phase_diff2_sliced) > np.pi * 0.5) & voiced_mask) / np.sum(voiced_mask)
            else:
                pjr = np.sum(np.abs(phase_diff2_sliced) > np.pi * 0.5) / phase_diff2_sliced.size
                
            print(f"  Freq < {f_bound}Hz, Mag > {m_thresh}dB: PJR = {pjr:.4f} (voiced pixels: {np.sum(voiced_mask)})")

if __name__ == "__main__":
    analyze_file(r"D:\Downloads!!\WhatsApp Audio 2026-06-21 at 4.34.52 PM.wav")
    analyze_file(r"D:\Downloads!!\demo.wav")
