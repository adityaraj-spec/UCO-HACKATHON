import os
import sys
import argparse
import torch
import numpy as np
import librosa
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings("ignore")

# Import preprocessing
from preprocess import load_and_standardize, audio_to_mel_spectrogram, extract_5_signals
# Import PhaseGuard Layer 1 model definition
from train_layer1 import PhaseGuardL1

def main():
    parser = argparse.ArgumentParser(description="Test a voice file with PhaseGuard Layer 1 (AI Voice Authenticity)")
    parser.add_argument("file_path", type=str, help="Path to the WAV audio file to test")
    parser.add_argument("--no-overrides", action="store_true", help="Disable physics-based consistency override logic")
    args = parser.parse_args()
    
    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found: {file_path}")
        sys.exit(1)
        
    model_path = "models/layer1_mobilenet.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model weights not found at: {model_path}")
        print("Please run train_layer1.py first to train the model.")
        sys.exit(1)
        
    print(f"Loading PhaseGuard Layer 1 model weights from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseGuardL1()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loading and preprocessing audio: {file_path}...")
    try:
        # Load full audio for physical signal extraction (avoids padding/truncation artifacts)
        y_full, _ = librosa.load(file_path, sr=16000)
        
        # Load and pad/truncate to 2.0s for the CNN spectrogram
        audio = load_and_standardize(file_path)
        # Convert to Mel-Spectrogram
        mel = audio_to_mel_spectrogram(audio)
        
        # Standardize mel shape to (128, 128)
        if mel.shape[1] != 128:
            mel_resized = zoom(mel, (1, 128 / mel.shape[1]))
        else:
            mel_resized = mel
            
        mel_min = mel_resized.min()
        mel_max = mel_resized.max()
        mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-10)
        
        # Prepare tensor (shape: 1, 1, 128, 128)
        tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0).to(device)
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}")
        sys.exit(1)
        
    # Extract physical signal features for diagnosis and validation on the full-length audio
    try:
        feats = extract_5_signals(y_full)
    except Exception as e:
        feats = None
        
    print("Running AI Voice Authenticity inference...")
    with torch.no_grad():
        prediction = model(tensor).item()
        
    # Apply bidirectional physical signal consistency check to prevent out-of-distribution errors
    is_physically_real = False
    is_physically_fake = False
    
    if feats:
        # 1. Override overfitted CNN false positives (Real voice classified as Fake)
        # Case A: Very clean / studio real voice (low phase jumps and organic jitter)
        if feats['phase_jump_rate'] < 0.11 and 0.0015 <= feats['jitter'] <= 0.075:
            is_physically_real = True
        # Case B: Compressed/echo-cancelled real voice (e.g., WhatsApp audio)
        # We raise the noise floor threshold from 0.0006 to 0.002 to avoid misclassifying quiet fake voices
        elif feats['noise_floor'] > 0.002 and 0.0015 <= feats['jitter'] <= 0.075:
            if feats['phase_jump_rate'] < 0.23:
                is_physically_real = True
        # Case C: Noise-gated/edited real voice (e.g., edited in Audacity)
        elif feats['noise_floor'] <= 0.0006 and 0.0015 <= feats['jitter'] <= 0.075:
            if feats['phase_jump_rate'] < 0.14:
                is_physically_real = True
                
        # 2. Override false negatives (Fake voice classified as Real, e.g., demo.wav)
        has_ai_jitter = (feats['jitter'] < 0.0012) or (feats['jitter'] > 0.055)
        
        # Only override to FAKE when the definitive AI signature is present:
        # near-zero digital silence (< 0.0005) — real recordings ALWAYS have some background energy.
        # Removed Case B (noise_floor < 0.002) — too close to boundary for mobile/phone recordings
        # which can legitimately have low-ish noise floors while still being real.
        if feats['phase_jump_rate'] > 0.08 and feats['noise_floor'] < 0.0005 and has_ai_jitter:
            is_physically_fake = True

    raw_prediction = prediction
    override_applied = None
    
    if not args.no_overrides:
        if is_physically_real and prediction > 0.5:
            # Only override to REAL if CNN is NOT near-certain about fake (< 96%)
            # Gap: LJSpeech real CNN ~0.72-0.93, WaveFake fake CNN ~0.9989
            # 0.96 sits cleanly between both groups
            if prediction < 0.96:
                prediction = min(prediction, 0.12)
                override_applied = "REAL (Low phase jumps / organic jitter)"
        elif is_physically_fake and prediction < 0.5:
            # Override to FAKE
            prediction = max(prediction, 0.88)
            override_applied = "FAKE (AI vocoder signatures / digital silence / robotic pitch)"
        
    print("\n" + "="*65)
    print("                  PHASEGUARD DIAGNOSTIC REPORT")
    print("="*65)
    print(f"Audio File Tested  : {os.path.basename(file_path)}")
    print(f"Model Architecture : MobileNetV3 Small (Layer 1 Audio Authenticity)")
    print(f"CNN Sigmoid Output : {raw_prediction:.4f}")
    if override_applied:
        print(f"Physics Override   : Applied -> {override_applied}")
        print(f"Final Output       : {prediction:.4f}")
    else:
        print(f"Physics Override   : None (or disabled)")
    
    if prediction > 0.5:
        confidence = prediction * 100
        print(f"DECISION           : [ALERT] FAKE / AI VOICE CLONE (Certainty: {confidence:.2f}%)")
    else:
        confidence = (1 - prediction) * 100
        print(f"DECISION           : [OK] REAL HUMAN VOICE (Certainty: {confidence:.2f}%)")
        
    print("-"*65)
    print("               EXTRACTED ACOUSTIC PHYSICAL SIGNALS")
    print("-"*65)
    if feats:
        print(f"1. Phase Jump Rate     : {feats['phase_jump_rate']:.4f}  (High = AI vocoder frame transitions)")
        print(f"2. Pitch Jitter        : {feats['jitter']:.6f}  (Low = Flat/Monotone robotic voice)")
        print(f"3. Spectral Flatness   : {feats['spectral_flatness']:.4f}  (High = Noise, Low = Rich speech formants)")
        print(f"4. Noise Floor (RMS)   : {feats['noise_floor']:.6f}  (Near-zero = Digital silence / Gen AI)")
        print(f"5. MFCC Delta Variance : {feats['mfcc_delta_var']:.4f}  (Low = Rigidity in speech texture)")
    else:
        print("Acoustic feature extraction failed.")
    print("="*65)

if __name__ == "__main__":
    main()
