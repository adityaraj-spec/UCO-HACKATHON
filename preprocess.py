import librosa
import numpy as np
import os
import warnings

# Suppress librosa audioread / soundfile warnings
warnings.filterwarnings('ignore')

def load_and_standardize(file_path, target_sr=16000, duration=2.0):
    """
    Loads an audio file, resamples it to 16000Hz, pads or truncates it 
    to exactly 2.0 seconds (32000 samples), and normalizes its volume.
    """
    # Load the audio file (librosa automatically handles mono conversion)
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Pad if shorter than 2.0 seconds
    target_length = int(target_sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    # Truncate if longer than 2.0 seconds
    y = y[:target_length]
    
    # Normalize loudness (scale peak absolute value to 1.0)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    return y

def audio_to_mel_spectrogram(y, sr=16000, n_mels=128, n_fft=512, hop_length=160):
    """
    Converts audio waveform into a log-scaled mel-spectrogram (decibel scale).
    n_mels=128: 128 frequency bands
    n_fft=512: FFT window size
    hop_length=160: hop length of 10ms (160 samples at 16kHz)
    """
    # Compute the spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    
    # Convert power to decibels (log scale)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_5_signals(y, sr=16000):
    """
    Extracts the 5 physical signals from audio waveform:
    1. Phase Jump Rate
    2. Pitch Jitter
    3. Spectral Flatness
    4. Noise Floor (10th percentile of RMS energy)
    5. MFCC Delta Variance
    """
    # ---- SIGNAL 1: Phase Jump Rate ----
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    # Compute 1st order wrapped difference
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    # Compute 2nd order wrapped difference (phase acceleration)
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    
    # Filter by log-magnitude and limit frequency range to < 3000 Hz (voiced harmonics range)
    # 3000 Hz / (16000 / 512) = 96 bins
    magnitude = np.abs(stft)
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    voiced_mask = mag_db[:96, 2:] > -30
    phase_diff2_wrapped_low = phase_diff2_wrapped[:96, :]
    
    if np.sum(voiced_mask) > 0:
        phase_jump_rate = np.sum((np.abs(phase_diff2_wrapped_low) > np.pi * 0.5) & voiced_mask) / np.sum(voiced_mask)
    else:
        phase_jump_rate = np.sum(np.abs(phase_diff2_wrapped_low) > np.pi * 0.5) / phase_diff2_wrapped_low.size

    # ---- SIGNAL 2: Pitch Jitter ----
    # Use probabilistic YIN algorithm to extract fundamental frequency
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),   # 65 Hz
            fmax=librosa.note_to_hz('C7'),   # 2093 Hz
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 2:
            # Jitter = Mean absolute differences of pitch / Mean pitch
            jitter = np.mean(np.abs(np.diff(f0_clean))) / (np.mean(f0_clean) + 1e-10)
        else:
            jitter = 0.0
    except Exception:
        # Fallback if pyin fails or has issues
        jitter = 0.0

    # ---- SIGNAL 3: Spectral Flatness ----
    # Value close to 1 means noisy/flat spectrum; close to 0 means highly harmonic
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # ---- SIGNAL 4: Noise Floor RMS ----
    # Find RMS (loudness/energy) of each frame
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    # The 10th percentile of energy acts as a proxy for the background noise floor
    noise_floor = float(np.percentile(rms, 10))

    # ---- SIGNAL 5: MFCC Delta Variance ----
    # Mel-Frequency Cepstral Coefficients (spectral texture)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Delta (first-order derivative over time)
    mfcc_delta = librosa.feature.delta(mfcc)
    # Variance of the delta coefficients across time
    mfcc_delta_var = float(np.var(mfcc_delta))

    return {
        'phase_jump_rate': float(phase_jump_rate),
        'jitter': float(jitter),
        'spectral_flatness': float(flatness),
        'noise_floor': float(noise_floor),
        'mfcc_delta_var': float(mfcc_delta_var)
    }

if __name__ == "__main__":
    # Test script if called directly
    print("Preprocess module loaded. Run main app to test.")
