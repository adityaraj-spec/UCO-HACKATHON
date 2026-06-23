import sys
from unittest.mock import MagicMock
sys.modules['k2'] = MagicMock()
sys.modules['flair'] = MagicMock()
sys.modules['flair.data'] = MagicMock()
sys.modules['flair.embeddings'] = MagicMock()
sys.modules['flair.models'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['spacy.tokens'] = MagicMock()

# Mock torchaudio.io to support SpeechBrain 1.0.0 imports on newer torchaudio versions
mock_io = MagicMock()
mock_io.StreamReader = MagicMock()
sys.modules['torchaudio.io'] = mock_io

import os
import json
import torch
import torchaudio
torchaudio.io = mock_io
try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    from speechbrain.pretrained import SpeakerRecognition

# Load the pre-trained ECAPA-TDNN model (downloads on first run, ~100MB)
print("Loading ECAPA-TDNN speaker recognition model...")
try:
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading ECAPA-TDNN from SpeechBrain: {e}")
    print("Please make sure you are connected to the internet to download the pretrained weights on the first run.")
    raise

def get_embedding(audio_file):
    """
    Extracts a 192-dimensional speaker embedding from the audio file.
    Resamples to 16kHz if needed.
    """
    import soundfile as sf
    import librosa
    
    # Load with soundfile
    signal_np, sr = sf.read(audio_file)
    
    # Resample if not 16000Hz
    if sr != 16000:
        signal_np = librosa.resample(signal_np, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    signal = torch.FloatTensor(signal_np)
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(0)
    else:
        signal = signal.t()
        
    # Get the embedding
    with torch.no_grad():
        embedding = verifier.encode_batch(signal)
        
    return embedding

def enroll_speaker(speaker_name, audio_folder):
    """
    Enrolls a speaker by processing multiple audio files and averaging their embeddings
    to form a stable voice template.
    """
    embeddings = []
    print(f"Enrolling speaker: {speaker_name}")
    
    if not os.path.exists(audio_folder):
        print(f"  ERROR: Audio folder '{audio_folder}' does not exist.")
        return None
        
    for filename in os.listdir(audio_folder):
        if filename.lower().endswith(('.wav', '.flac', '.mp3')):
            filepath = os.path.join(audio_folder, filename)
            try:
                emb = get_embedding(filepath)
                embeddings.append(emb)
                print(f"  Processed: {filename}")
            except Exception as e:
                print(f"  Failed to process {filename}: {e}")
                
    if len(embeddings) == 0:
        print(f"  ERROR: No valid audio files processed for {speaker_name}")
        return None
        
    # Average all embeddings (shape of each is (1, 1, 192) or similar)
    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    print(f"  Enrolled {len(embeddings)} recordings for {speaker_name}.")
    print(f"  Voiceprint shape: {avg_embedding.shape}")
    
    return avg_embedding

def cosine_similarity(emb1, emb2):
    """
    Computes cosine similarity between two voice embeddings.
    Values range from -1.0 to 1.0 (1.0 = identical).
    """
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    similarity = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    return float(similarity)

def main():
    enrolled_voiceprints = {}
    speakers = {
        "vishal": "data/real_voices/vishal",
        "abhinav": "data/real_voices/abhinav",
        "aditya": "data/real_voices/aditya",
        "dhruv": "data/real_voices/dhruv"
    }
    
    # Run enrollment
    for name, folder in speakers.items():
        if os.path.exists(folder):
            voiceprint = enroll_speaker(name, folder)
            if voiceprint is not None:
                enrolled_voiceprints[name] = voiceprint
        else:
            print(f"Speaker directory not found: {folder}. Skipping.")
            
    if len(enrolled_voiceprints) == 0:
        print("\nERROR: No voiceprints enrolled. Cannot save database.")
        return
        
    # Save the voiceprints dictionary
    os.makedirs("models", exist_ok=True)
    save_path = "models/voiceprints.pth"
    torch.save(enrolled_voiceprints, save_path)
    print(f"\nAll enrolled voiceprints saved to: {save_path}")
    
    # Verification test if files exist
    print("\n--- RUNNING ENROLLMENT VERIFICATION TEST ---")
    test_speaker = "vishal"
    if test_speaker in enrolled_voiceprints:
        real_test_file = "data/real_voices/vishal/vishal_001.wav"
        clone_test_file = "data/clones/vishal/vishal_clone_001.wav"
        
        if os.path.exists(real_test_file) and os.path.exists(clone_test_file):
            real_emb = get_embedding(real_test_file)
            clone_emb = get_embedding(clone_test_file)
            stored_emb = enrolled_voiceprints[test_speaker]
            
            real_score = cosine_similarity(real_emb, stored_emb)
            clone_score = cosine_similarity(clone_emb, stored_emb)
            
            print(f"Testing speaker identity against: '{test_speaker}' stored voiceprint")
            print(f"  Real voice file similarity: {real_score:.4f} -> {'PASS [OK]' if real_score >= 0.65 else 'FAIL [FAIL]'}")
            print(f"  Clone voice file similarity: {clone_score:.4f} -> {'PASS [OK]' if clone_score >= 0.65 else 'FRAUD ALERT (Identity Mismatch/Clone) [FRAUD]'}")
        else:
            print("Verification test files not found. Generate simulated data first.")
    else:
        print(f"Skipping verification test: '{test_speaker}' was not enrolled.")

if __name__ == "__main__":
    main()
