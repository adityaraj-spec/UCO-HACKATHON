import uuid
import json
import numpy as np
import io
import soundfile as sf
import torch
import librosa
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_db
from app.services.verification_service import VerificationService
from scripts.features import extract_all
from scripts.train import PhaseGuardL1
import os

router = APIRouter()

# Load Layer 1 Model Globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
l1_model = PhaseGuardL1().to(device)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "layer1_mobilenet.pth")
if os.path.exists(model_path):
    l1_model.load_state_dict(torch.load(model_path, map_location=device))
l1_model.eval()

@router.websocket("/stream/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    service = VerificationService(session=db)
    
    # We will accumulate audio bytes here
    audio_buffer = bytearray()
    
    try:
        while True:
            # Receive audio chunk from client (expecting raw PCM 16kHz float32 or webm, but let's assume webm/wav for simplicity)
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # For a real implementation, we would process audio_buffer every few seconds.
            # To simulate, if buffer gets large enough (e.g., > 32KB), we process it.
            if len(audio_buffer) > 64000:
                # Process buffer
                try:
                    # Try to read the buffer as an audio file
                    y, sr = sf.read(io.BytesIO(audio_buffer))
                    # Resample to 16000 if needed (assuming it is 16000)
                    
                    # Run Layer 1 (extract_all expects a filepath, so we need to bypass it or write to a temp file)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, y, sr)
                        tmp_path = tmp.name
                    
                    # 1. Layer 1 Prediction
                    result_l1 = extract_all(tmp_path, sr=16000)
                    ai_prob = 0.0
                    if result_l1 is not None:
                        mel_t = torch.FloatTensor(result_l1['mel']).unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            ai_prob = float(l1_model(mel_t)[0][0])
                    
                    # 2. Layer 2 Prediction
                    # We can use the VerificationService.verify method by passing a mocked UploadFile
                    # but verify expects an UploadFile. Let's create a dummy class or just call internal methods.
                    from fastapi import UploadFile
                    with open(tmp_path, "rb") as f:
                        file_obj = UploadFile(filename="stream.wav", file=f)
                        result_l2 = await service.verify(user_id=user_id, audio_file=file_obj, layer1_score=ai_prob)
                    
                    os.unlink(tmp_path)
                    
                    # Send response back
                    response = {
                        "layer1": {
                            "ai_probability": ai_prob,
                            "signals": result_l1['signals'] if result_l1 else {}
                        },
                        "layer2": {
                            "similarity": result_l2.similarity_score,
                            "verified": result_l2.verified,
                            "risk_score": result_l2.risk_score,
                            "risk_level": result_l2.risk_level
                        }
                    }
                    await websocket.send_text(json.dumps(response))
                    
                    # Clear buffer after processing
                    audio_buffer = bytearray()
                    
                except Exception as e:
                    print("Error processing stream chunk:", e)
                    # Clear buffer on error to prevent infinite buildup
                    audio_buffer = bytearray()
                    
    except WebSocketDisconnect:
        print(f"Client disconnected: {user_id}")
