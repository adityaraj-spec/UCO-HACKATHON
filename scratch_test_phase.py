import librosa
import numpy as np
from preprocess import load_and_standardize

def original_pjr(y):
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    phase_jump_rate = np.sum(np.abs(phase_diff) > np.pi * 0.5) / phase_diff.size
    return phase_jump_rate

def wrapped_pjr_first_order(y):
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_jump_rate = np.sum(np.abs(phase_diff_wrapped) > np.pi * 0.5) / phase_diff_wrapped.size
    return phase_jump_rate

def wrapped_pjr_second_order(y):
    stft = librosa.stft(y, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    phase_diff_wrapped = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_diff2 = np.diff(phase_diff_wrapped, axis=1)
    phase_diff2_wrapped = np.arctan2(np.sin(phase_diff2), np.cos(phase_diff2))
    phase_jump_rate = np.sum(np.abs(phase_diff2_wrapped) > np.pi * 0.5) / phase_diff2_wrapped.size
    return phase_jump_rate

# Real voice
y_real = load_and_standardize("data/real_voices/vishal/vishal_001.wav")
# Cloned voice
y_clone = load_and_standardize("data/clones/vishal/vishal_clone_001.wav")

print("Real Voice original PJR:", original_pjr(y_real))
print("Clone Voice original PJR:", original_pjr(y_clone))
print("---")
print("Real Voice wrapped 1st order PJR:", wrapped_pjr_first_order(y_real))
print("Clone Voice wrapped 1st order PJR:", wrapped_pjr_first_order(y_clone))
print("---")
print("Real Voice wrapped 2nd order PJR:", wrapped_pjr_second_order(y_real))
print("Clone Voice wrapped 2nd order PJR:", wrapped_pjr_second_order(y_clone))
