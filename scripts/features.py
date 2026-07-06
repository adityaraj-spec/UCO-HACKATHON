"""
features.py — Extracts all 5 Layer 1 signals from any audio file.
"""
import librosa
import numpy as np
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore') # to silence pyin warnings

def load_audio(filepath, sr=16000, duration=2.0):
    """Load + standardize: exactly 2 seconds at 16kHz, normalized volume"""
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        print(f" ERROR loading {filepath}: {e}")
        return None
    target_len = int(sr * duration) # 32000 samples
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    y = y[:target_len]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y

def phase_jump_rate(y, sr=16000, n_fft=512, hop_length=160,
                    mag_threshold_db=-40, accel_threshold=1.2):
    """
    SIGNAL 1: Phase Acceleration Anomaly (v3 — acceleration-based)
    Detects sudden shifts in the RATE of phase change, not the rate itself.
    - Smooth speech (real): pitch glides gradually -> small 2nd derivative
    - Vocoder artifacts (fake): frame-boundary discontinuities -> spiked 2nd derivative
    Robust to natural pitch glides/formant transitions which change smoothly.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Step 1 — unwrap along time, per bin
    phase_unwrapped = np.unwrap(phase, axis=1)

    # Step 2 — first derivative = local instantaneous frequency estimate
    inst_freq = np.diff(phase_unwrapped, axis=1)              # (n_bins, T-1)

    # Step 3 — second derivative = how fast inst. freq ITSELF is changing
    # Smooth speech: small. Vocoder frame-boundary artifact: spikes.
    accel = np.diff(inst_freq, axis=1)                        # (n_bins, T-2)

    # Step 4 — mask to bins/frames with real energy across all 3 frames involved
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    significant = (mag_db[:, :-2] > mag_threshold_db) & \
                  (mag_db[:, 1:-1] > mag_threshold_db) & \
                  (mag_db[:, 2:] > mag_threshold_db)

    if np.sum(significant) == 0:
        return 0.0

    jumps = np.abs(accel) > accel_threshold
    return float(np.sum(jumps & significant) / np.sum(significant))

def pitch_jitter(y, sr=16000):
    """
    SIGNAL 2: Pitch Jitter
    Real voice -> natural biological tremor (0.01-0.05)
    AI voice   -> too-perfect pitch (near 0) or fake variation
    """
    f0, _, _ = librosa.pyin(y,
                            fmin=librosa.note_to_hz('C2'),  # 65 Hz
                            fmax=librosa.note_to_hz('C7'),  # 2093 Hz
                            sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 3:
        return 0.0
    diffs = np.abs(np.diff(f0_clean))
    return float(np.mean(diffs) / (np.mean(f0_clean) + 1e-10))

def spectral_flatness(y):
    """
    SIGNAL 3: Spectral Flatness
    Real speech -> lower (tonal peaks from vowels)
    AI audio    -> sometimes abnormally flat spectrum
    """
    return float(np.mean(librosa.feature.spectral_flatness(y=y)))

def noise_floor_rms(y, frame_length=512, hop_length=160):
    """
    SIGNAL 4: Noise Floor RMS (10th percentile energy)
    Real call -> room ambience > 0.001
    AI file   -> near-zero silence between words (~0.0001)
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                              hop_length=hop_length)[0]
    return float(np.percentile(rms, 10))

def mfcc_delta_variance(y, sr=16000, n_mfcc=13):
    """
    SIGNAL 5: MFCC Delta Variance
    Real voice -> natural dynamic texture changes
    AI voice   -> too smooth (low variance) or jerky (high)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    return float(np.var(delta))

def make_mel_spectrogram(y, sr=16000, n_mels=128, n_fft=512, hop_length=160):
    """Convert 2-second audio -> 128x128 image for the CNN"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_sq = zoom(mel_db, (1.0, 128.0 / mel_db.shape[1])) # resize to 128x128
    lo, hi = mel_sq.min(), mel_sq.max()
    if hi > lo:
        mel_sq = (mel_sq - lo) / (hi - lo) # normalize 0.0-1.0
    return mel_sq

def extract_all(filepath, sr=16000):
    """Master function — extract everything from one audio file"""
    y = load_audio(filepath, sr=sr)
    if y is None:
        return None
        
    signals = {
        'phase_jump_rate'   : phase_jump_rate(y, sr=sr),
        'jitter'            : pitch_jitter(y, sr),
        'spectral_flatness' : spectral_flatness(y),
        'noise_floor'       : noise_floor_rms(y),
        'mfcc_delta_var'    : mfcc_delta_variance(y, sr),
    }
    mel = make_mel_spectrogram(y, sr)
    return {'signals': signals, 'mel': mel, 'audio': y}

if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw_voices/vishal/vishal_real_001.wav"
    result = extract_all(test_file)
    if result:
        print("\n5 SIGNALS:")
        for name, val in result['signals'].items():
            print(f"  {name:<22}: {val:.5f}")
        print(f"\nMel-spectrogram shape: {result['mel'].shape}")
