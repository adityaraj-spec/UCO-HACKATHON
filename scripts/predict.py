"""
predict.py — Test Layer 1 on any audio file
Usage: python scripts/predict.py path/to/audio.wav  (from project root)
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OMP DLL conflict on Windows
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _HERE)

import torch
from features import extract_all
from train import PhaseGuardL1

def predict(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseGuardL1().to(device)
    
    # Check if model exists
    if not os.path.exists("models/layer1_mobilenet.pth"):
        print("Error: models/layer1_mobilenet.pth not found. Run train.py first.")
        return
        
    model.load_state_dict(torch.load("models/layer1_mobilenet.pth", map_location=device))
    model.eval()
    
    result = extract_all(audio_path)
    if result is None:
        print("Could not process file"); return
        
    mel_t = torch.FloatTensor(result['mel']).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ai_prob = float(model(mel_t)[0][0])
        
    print(f"\nFILE: {os.path.basename(audio_path)}")
    print(f"AI Probability : {ai_prob*100:.1f}%")
    print(f"Layer 1 Result : ", end="")
    
    if ai_prob > 0.70: print("BLOCKED — AI Voice Detected")
    elif ai_prob > 0.50: print("SUSPICIOUS — Elevated risk")
    else: print("PASS — Likely Real Voice")
    
    print("\n5 Signal Meters:")
    for name, val in result['signals'].items():
        status = "Normal"
        # Calibrated on actual real vs fake data:
        # NoiseFlr: fakes HIGHER (0.002-0.015) vs real (0.000-0.001)
        # Jitter:   fakes slightly higher trend
        # PhaseAccel: still overlapping — treat as informational only
        if name == 'noise_floor' and val > 0.002:
            status = "ALERT (elevated — fake tendency)"
        elif name == 'jitter' and val > 0.065:
            status = "ALERT (elevated — fake tendency)"
        elif name == 'phase_jump_rate' and val > 0.38:
            status = "WATCH (elevated — calibrating)"

        print(f"  {name:<22}: {val:.5f} [{status}]")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_voices/vishal/vishal_real_001.wav"
    predict(path)
