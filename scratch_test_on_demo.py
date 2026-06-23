import os
import librosa
import numpy as np

def calculate_pjr(filepath, freq_limit_hz=None):
    # Load and standardize
    y, sr = librosa.load(filepath, sr=16000)
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    
    # 2nd order wrapped phase difference
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    
    # Magnitude voiced mask
    magnitude = np.abs(stft)
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Frequency bin limit
    if freq_limit_hz is not None:
        bin_spacing = 16000 / 512 # 31.25 Hz per bin
        max_bin = int(freq_limit_hz / bin_spacing)
        mag_db_sliced = mag_db[:max_bin, 2:]
        phase_diff2_wrapped_sliced = phase_diff2_wrapped[:max_bin, :]
    else:
        mag_db_sliced = mag_db[:, 2:]
        phase_diff2_wrapped_sliced = phase_diff2_wrapped
        
    voiced_mask = mag_db_sliced > -30
    
    if np.sum(voiced_mask) > 0:
        pjr = np.sum((np.abs(phase_diff2_wrapped_sliced) > np.pi * 0.5) & voiced_mask) / np.sum(voiced_mask)
    else:
        pjr = np.sum(np.abs(phase_diff2_wrapped_sliced) > np.pi * 0.5) / phase_diff2_wrapped_sliced.size
        
    return pjr

def main():
    files = ["demo.wav"]
    # Check if the WhatsApp file is there
    whatsapp_files = [f for f in os.listdir(".") if "whatsapp" in f.lower() and f.endswith(".wav")]
    files.extend(whatsapp_files)
    
    print("Testing files:", files)
    freqs = [None, 4000, 3000, 2000, 1500, 1000]
    
    for f in files:
        if not os.path.exists(f):
            print(f"File {f} not found!")
            continue
        print(f"\n--- File: {f} ---")
        for freq in freqs:
            pjr = calculate_pjr(f, freq)
            print(f"  Freq Limit {freq} Hz: PJR = {pjr:.4f}")

if __name__ == "__main__":
    main()
