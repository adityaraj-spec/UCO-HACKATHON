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

import streamlit as st
import torch
import torchaudio
torchaudio.io = mock_io
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import time
import soundfile as sf
from scipy.ndimage import zoom

# Import PhaseGuard helper modules
from preprocess import load_and_standardize, audio_to_mel_spectrogram, extract_5_signals
from train_layer1 import PhaseGuardL1

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="PhaseGuard - Voice Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (custom cards, fonts, and borders)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 90% 10%, #1e1e38 0%, #0d0d15 100%);
        color: #f1f3f9;
    }
    
    /* Premium Title Banner */
    .title-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    /* Telemetry Card styles */
    .sensor-card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
        backdrop-filter: blur(5px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .sensor-card:hover {
        transform: translateY(-2px);
        border-color: rgba(37, 99, 235, 0.4);
    }
    .sensor-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .sensor-value {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .sensor-status-normal {
        color: #10b981;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .sensor-status-anomaly {
        color: #f43f5e;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Verdict Cards */
    .verdict-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .verdict-clean {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6) 0.25%);
        color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .verdict-highrisk {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(234, 88, 12) 0.25%);
        color: #f97316;
        border-color: rgba(249, 115, 22, 0.3);
    }
    .verdict-fraud {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(225, 29, 72, 0.28) 100%);
        color: #f43f5e;
        border-color: rgba(244, 63, 94, 0.4);
        animation: pulse 2.0s infinite alternate;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(244, 63, 94, 0.2); }
        100% { box-shadow: 0 0 20px rgba(244, 63, 94, 0.4); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED MODEL LOADERS
# ==========================================
@st.cache_resource
def load_models_db():
    """Loads the Layer 1 MobileNet weights and Layer 2 ECAPA speaker profiles."""
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition
    
    # 1. Load Layer 1 Model
    l1_model = PhaseGuardL1()
    l1_model.load_state_dict(torch.load("models/layer1_mobilenet.pth", map_location='cpu'))
    l1_model.eval()
    
    # 2. Load Layer 2 Pretrained Model (Optional)
    verifier = None
    try:
        verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa"
        )
    except Exception as e:
        print(f"Layer 2 SpeechBrain verifier loading skipped or failed: {e}")
    
    # 3. Load Enrolled Database (Optional)
    voiceprints = {}
    if os.path.exists("models/voiceprints.pth"):
        try:
            voiceprints = torch.load("models/voiceprints.pth", map_location='cpu')
        except Exception as e:
            print(f"Layer 2 voiceprints loading failed: {e}")
    
    return l1_model, verifier, voiceprints

# Check model availability first (Only Layer 1 is strictly required now)
models_ready = os.path.exists("models/layer1_mobilenet.pth")

# ==========================================
# AUDIO PREDICTION PIPELINE
# ==========================================
def analyze_voice_clip(audio_path, l1_model, verifier, voiceprints, selected_speaker, l1_thresh, l2_thresh):
    """
    Executes PhaseGuard's dual-layer engine:
    Layer 1: Phase check & physical sensors (CNN model & metric limits)
    Layer 2: Speaker validation (ECAPA-TDNN cosine score)
    """
    # Load full audio for physical features to avoid padding/truncation artifacts
    try:
        y_full, _ = librosa.load(audio_path, sr=16000)
    except Exception:
        y_full = None
        
    # Load and standardize waveform for the CNN model spectrogram
    audio = load_and_standardize(audio_path)
    
    # Extract the 5 physical signals
    if y_full is not None:
        signals = extract_5_signals(y_full)
    else:
        signals = extract_5_signals(audio)
    
    # Convert waveform to Mel-spectrogram
    mel = audio_to_mel_spectrogram(audio)
    if mel.shape[1] != 128:
        mel_resized = zoom(mel, (1, 128 / mel.shape[1]))
    else:
        mel_resized = mel
    
    # Normalize mel
    mel_min = mel_resized.min()
    mel_max = mel_resized.max()
    mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-10)
    
    # Run Layer 1 CNN Inference
    mel_tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
    with torch.no_grad():
        ai_probability = float(l1_model(mel_tensor)[0][0])
        
    # Apply bidirectional physical signal consistency check to prevent out-of-distribution errors
    is_physically_real = False
    is_physically_fake = False
    
    if signals:
        # 1. Override overfitted CNN false positives (Real voice classified as Fake)
        # Case A: Very clean / studio real voice (low phase jumps and organic jitter)
        if signals['phase_jump_rate'] < 0.11 and 0.0015 <= signals['jitter'] <= 0.05:
            is_physically_real = True
        # Case B: Compressed/echo-cancelled real voice (e.g., WhatsApp audio)
        # It may have elevated PJR, but retains organic jitter and natural room noise floor
        elif signals['noise_floor'] > 0.0006 and 0.0015 <= signals['jitter'] <= 0.05:
            if signals['phase_jump_rate'] < 0.23:
                is_physically_real = True
        # Case C: Noise-gated/edited real voice (e.g., edited in Audacity)
        # Has digital silence, but retains organic human jitter and moderately low PJR
        elif signals['noise_floor'] <= 0.0006 and 0.0015 <= signals['jitter'] <= 0.05:
            if signals['phase_jump_rate'] < 0.14:
                is_physically_real = True
                
        # 2. Override false negatives (Fake voice classified as Real, e.g., demo.wav)
        # A synthetic override should only trigger if the voice exhibits both AI vocoder signatures
        # (near-zero noise floor or erratic/robotic pitch) AND phase discontinuities,
        # AND it does NOT have organic human jitter.
        has_ai_jitter = (signals['jitter'] < 0.0012) or (signals['jitter'] > 0.055)
        
        if signals['phase_jump_rate'] > 0.12:
            # Case A: Digital silence (near-zero noise floor) AND AI/unnatural jitter
            if signals['noise_floor'] < 0.0005 and has_ai_jitter:
                is_physically_fake = True
            # Case B: AI/unnatural jitter (independent of noise floor)
            elif has_ai_jitter:
                is_physically_fake = True

    if is_physically_real and ai_probability > l1_thresh:
        # Override to REAL
        ai_probability = min(ai_probability, 0.12)
    elif is_physically_fake and ai_probability < l1_thresh:
        # Override to FAKE
        ai_probability = max(ai_probability, 0.88)
        
    # Layer 1 Block Decision
    layer1_blocked = ai_probability > l1_thresh
    
    # Run Layer 2 Inference (Speaker Verification)
    layer2_score = None
    if selected_speaker in voiceprints:
        try:
            signal_np, sr = sf.read(audio_path)
            if sr != 16000:
                signal_np = librosa.resample(signal_np, orig_sr=sr, target_sr=16000)
                sr = 16000
            signal = torch.FloatTensor(signal_np)
            if len(signal.shape) == 1:
                signal = signal.unsqueeze(0)
            else:
                signal = signal.t()
            
            with torch.no_grad():
                live_emb = verifier.encode_batch(signal).flatten()
            stored_emb = voiceprints[selected_speaker].flatten()
            
            # Compute similarity
            layer2_score = float(torch.dot(live_emb, stored_emb) / (torch.norm(live_emb) * torch.norm(stored_emb)))
        except Exception as e:
            st.error(f"Error during speaker verification: {e}")
            
    # Risk Assessment Logic
    blocked_at = None
    if layer1_blocked:
        risk_score = ai_probability * 100
        risk_level = "FRAUD ALERT"
        blocked_at = "Layer 1 (AI Voice Artifacts Detected)"
    elif layer2_score is not None and layer2_score < l2_thresh:
        risk_score = 85.0
        risk_level = "FRAUD ALERT"
        blocked_at = f"Layer 2 (Identity Verification Mismatch: Score {layer2_score:.2f} < {l2_thresh})"
    else:
        # Compute combined risk score
        l2_risk_component = max(0.0, l2_thresh - (layer2_score if layer2_score is not None else l2_thresh)) / l2_thresh
        risk_score = (ai_probability * 0.4 + l2_risk_component * 0.6) * 100
        
        # Clamp to 0-100
        risk_score = min(100.0, max(0.0, risk_score))
        
        if risk_score < 30.0:
            risk_level = "CLEAN"
        elif risk_score < 60.0:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "HIGH RISK"
            
    return {
        'signals': signals,
        'mel_spectrogram': mel_norm,
        'ai_probability': ai_probability,
        'layer2_score': layer2_score,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'blocked_at': blocked_at,
        'audio': audio
    }

# ==========================================
# RENDER HEADER BANNERS
# ==========================================
with st.container():
    st.markdown("""
    <div class="title-banner">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ PhaseGuard</h1>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem; color: #94a3b8; font-weight: 400;">
            Real-Time Two-Layer Voice Forensics for Banking Security & Deepfake Mitigation
        </p>
        <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #64748b; font-style: italic;">
            UCO Bank Hackathon 2026 — Built by Team Ozymandias
        </p>
    </div>
    """, unsafe_allow_html=True)

# Loading state
if not models_ready:
    st.error("🚨 System Status: Models Offline")
    st.info("The required model files (`layer1_mobilenet.pth`) are missing. Please train the model first using train_layer1.py.")
else:
    # Load models
    with st.spinner("Initializing PhaseGuard..."):
        l1_model, verifier, voiceprints = load_models_db()
    
    if verifier is not None and voiceprints:
        st.sidebar.success("✅ Layer 1 & 2 Models Active")
    else:
        st.sidebar.info("💡 Layer 1 (AI vs Real) Active (Layer 2 Offline)")
        
    # ==========================================
    # SIDEBAR DEMO SETTINGS
    # ==========================================
    with st.sidebar:
        st.markdown("### 🎯 Enrolled Bank Profile")
        if voiceprints:
            selected_speaker = st.selectbox(
                "Select customer account to verify:",
                options=list(voiceprints.keys()),
                help="Choose the bank customer whom the caller is claiming to be."
            )
        else:
            st.warning("No Layer 2 profiles enrolled.")
            selected_speaker = None
        
        st.markdown("---")
        st.markdown("### 🛠️ Calibration Thresholds")
        l1_threshold = st.slider(
            "Layer 1 Block Threshold (AI Probability)",
            min_value=0.50, max_value=0.95, value=0.70, step=0.05,
            help="If AI voice probability exceeds this, Layer 1 immediately flags FRAUD."
        )
        l2_threshold = st.slider(
            "Layer 2 Fraud Threshold (Cosine Similarity)",
            min_value=0.40, max_value=0.85, value=0.65, step=0.05,
            help="Threshold for ECAPA voice ID comparison. Scores below this are flagged as FRAUD."
        )
        
        st.markdown("---")
        st.markdown("### 💡 Demo Cheat Sheet")
        st.markdown("""
        - **Vishal**: Male (F0 ~120Hz)
        - **Abhinav**: Male (F0 ~135Hz)
        - **Aditya**: Male (F0 ~150Hz)
        - **Dhruv**: Male (F0 ~110Hz)
        
        *To test clones, upload files from `data/clones/[name]/` or trigger mock stream.*
        """)
        
    # ==========================================
    # TABS DESIGN
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📁 File Analysis", "🎙️ Live Stream Simulator", "📊 System Architecture & Theory"])
    
    # ------------------------------------------
    # TAB 1: FILE UPLOAD ANALYSIS
    # ------------------------------------------
    with tab1:
        st.subheader("Upload Call Recording")
        uploaded_file = st.file_uploader(
            "Upload WAV, MP3, or FLAC audio file",
            type=["wav", "mp3", "flac"]
        )
        
        if uploaded_file is not None:
            # Play uploaded file
            st.audio(uploaded_file, format='audio/wav')
            
            # Temporary save
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            if st.button("🔍 Run Forensic Scan", type="primary", use_container_width=True):
                # Stepwise progress bar
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                progress_placeholder.info("Step 1/4: Resampling audio & standardizing amplitude...")
                progress_bar.progress(25)
                time.sleep(0.3)
                
                progress_placeholder.info("Step 2/4: Extracting 5-signal physical features...")
                progress_bar.progress(50)
                time.sleep(0.3)
                
                progress_placeholder.info("Step 3/4: running Layer 1 CNN (MobileNetV3) for deepfake detection...")
                progress_bar.progress(75)
                time.sleep(0.3)
                
                progress_placeholder.info("Step 4/4: Scanning Layer 2 Speaker Profiles (ECAPA-TDNN)...")
                progress_bar.progress(95)
                
                results = analyze_voice_clip(
                    tmp_path, l1_model, verifier, voiceprints, 
                    selected_speaker, l1_threshold, l2_threshold
                )
                
                progress_bar.progress(100)
                progress_placeholder.success("Forensic Scan Complete!")
                time.sleep(0.2)
                progress_placeholder.empty()
                
                # Cleanup temp file
                os.unlink(tmp_path)
                
                # RENDER RESULTS
                st.divider()
                
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    # Risk Level Verdict Display
                    rl = results['risk_level']
                    if rl == "CLEAN":
                        st.markdown(f'<div class="verdict-card verdict-clean">🟢 VERDICT: CLEAN ({results["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
                        st.info("✅ Verified. Voice matches account credentials. The call is clean to proceed with transaction.")
                    elif rl == "SUSPICIOUS":
                        st.markdown(f'<div class="verdict-card verdict-suspicious">🟡 VERDICT: SUSPICIOUS ({results["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
                        st.warning("⚠️ Warning: Slight variance detected. Recommended: Query customer verification questions.")
                    elif rl == "HIGH RISK":
                        st.markdown(f'<div class="verdict-card verdict-highrisk">🟠 VERDICT: HIGH RISK ({results["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
                        st.warning("🔒 Alert: High anomaly risk. Triggering out-of-band OTP SMS verification.")
                    else:  # FRAUD ALERT
                        st.markdown(f'<div class="verdict-card verdict-fraud">🔴 BLOCK ACTION: FRAUD ALERT ({results["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
                        st.error(f"🚫 CALL INTERCEPTED: Blocked at {results['blocked_at']}")
                        
                    # Combined Risk Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = results['risk_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Combined Fraud Probability", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#f43f5e" if results['risk_score'] > 60 else "#3b82f6"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.1)'},
                                {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.1)'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70.0
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white", 'family': "Outfit"},
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_right:
                    # Model Sub-metrics
                    st.markdown("### 🎛️ Dual-Layer Confidence Indices")
                    
                    # Layer 1 Metrics
                    st.markdown(f"**Layer 1 (AI Voice Classifier):**")
                    col_l1_score, col_l1_state = st.columns([2, 1])
                    col_l1_score.metric("AI Voice Probability", f"{results['ai_probability']*100:.2f}%", help="CNN classification score.")
                    if results['ai_probability'] > l1_thresh:
                        col_l1_state.markdown("<span style='color: #f43f5e; font-weight: bold;'>⚠️ BLOCKED (Deepfake)</span>", unsafe_allow_html=True)
                    else:
                        col_l1_state.markdown("<span style='color: #10b981; font-weight: bold;'>✓ PASS (Human)</span>", unsafe_allow_html=True)
                        
                    st.markdown("---")
                    
                    # Layer 2 Metrics
                    st.markdown(f"**Layer 2 (Speaker Voiceprint Comparison):**")
                    col_l2_score, col_l2_state = st.columns([2, 1])
                    if results['layer2_score'] is not None:
                        col_l2_score.metric(f"Cosine Similarity vs {selected_speaker.capitalize()}", f"{results['layer2_score']:.4f}", help="Cosine similarity of vocal embeddings.")
                        if results['layer2_score'] >= l2_thresh:
                            col_l2_state.markdown("<span style='color: #10b981; font-weight: bold;'>✓ PASS (Match)</span>", unsafe_allow_html=True)
                        else:
                            col_l2_state.markdown("<span style='color: #f43f5e; font-weight: bold;'>⚠️ IDENTITY MISMATCH</span>", unsafe_allow_html=True)
                    else:
                        col_l2_score.write("Layer 2 bypassed (Layer 1 Deepfake detected first).")
                        col_l2_state.write("-")

                st.divider()
                
                # ---- 5 SIGNAL TELEMETRY PANEL ----
                st.markdown("### 📡 5-Signal Telemetry Board")
                st.caption("Acoustic signals extracted in real-time. Red cards indicate measurements drifting outside typical human boundaries.")
                
                sigs = results['signals']
                
                # Threshold logic for visual cues
                c_phase = "anomaly" if sigs['phase_jump_rate'] > 0.12 else "normal"
                c_jitter = "anomaly" if sigs['jitter'] < 0.002 else "normal"
                c_flat = "anomaly" if sigs['spectral_flatness'] > 0.05 else "normal"
                c_noise = "anomaly" if sigs['noise_floor'] < 0.005 else "normal"
                c_delta = "anomaly" if sigs['mfcc_delta_var'] < 5.0 or sigs['mfcc_delta_var'] > 120.0 else "normal"
                
                c_p_txt = "Anomalous Jumps" if c_phase == "anomaly" else "Smooth Flow"
                c_j_txt = "Robotic/Stiff Pitch" if c_jitter == "anomaly" else "Organic Jitter"
                c_f_txt = "Synthetic Flatness" if c_flat == "anomaly" else "Vocal Formants"
                c_n_txt = "Unnatural Silence" if c_noise == "anomaly" else "Natural Room Noise"
                c_d_txt = "Rigid Transitions" if c_delta == "anomaly" else "Dynamic Range"
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-title">⚡ Phase Jump Rate</div>
                    <div class="sensor-value">{sigs['phase_jump_rate']:.4f}</div>
                    <div class="sensor-status-{c_phase}">{c_p_txt}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Limit: &le; 0.1200</div>
                </div>
                """, unsafe_allow_html=True)
                
                col2.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-title">🎙️ Pitch Jitter</div>
                    <div class="sensor-value">{sigs['jitter']:.5f}</div>
                    <div class="sensor-status-{c_jitter}">{c_j_txt}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Limit: &gt; 0.0020</div>
                </div>
                """, unsafe_allow_html=True)
                
                col3.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-title">🎚️ Spectral Flatness</div>
                    <div class="sensor-value">{sigs['spectral_flatness']:.4f}</div>
                    <div class="sensor-status-{c_flat}">{c_f_txt}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Limit: &le; 0.0500</div>
                </div>
                """, unsafe_allow_html=True)
                
                col4.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-title">🔌 Noise Floor (RMS)</div>
                    <div class="sensor-value">{sigs['noise_floor']:.4f}</div>
                    <div class="sensor-status-{c_noise}">{c_n_txt}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Limit: &gt; 0.0050</div>
                </div>
                """, unsafe_allow_html=True)
                
                col5.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-title">📊 MFCC Delta Var</div>
                    <div class="sensor-value">{sigs['mfcc_delta_var']:.2f}</div>
                    <div class="sensor-status-{c_delta}">{c_d_txt}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Range: 5.0 - 120.0</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Visual Graphs
                st.markdown("### 📈 Acoustic Waveform & Spectral Footprint")
                col_w, col_s = st.columns(2)
                
                with col_w:
                    st.markdown("**Normalized Time-Domain Amplitude**")
                    fig_wave = px.line(x=np.arange(len(results['audio']))/16000, y=results['audio'], labels={'x': 'Time (s)', 'y': 'Amplitude'})
                    fig_wave.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    fig_wave.update_xaxes(showgrid=False, color="#475569")
                    fig_wave.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#475569")
                    st.plotly_chart(fig_wave, use_container_width=True)
                    
                with col_s:
                    st.markdown("**Mel-Spectrogram Fingerprint (128x128 Model Input)**")
                    fig_spec = px.imshow(results['mel_spectrogram'], labels=dict(x="Time frames", y="Mel bins"), color_continuous_scale='Viridis')
                    fig_spec.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)

    # ------------------------------------------
    # TAB 2: LIVE RECORDING / STREAM SIMULATOR
    # ------------------------------------------
    with tab2:
        st.subheader("Interactive Voice Channel Scanner")
        st.write("Simulate or intercept live telephone voice streams. You can record a live clip from your microphone or trigger pre-packaged audio flows to evaluate call safety instantly.")
        
        col_rec_left, col_rec_right = st.columns([1, 1])
        
        with col_rec_left:
            st.markdown("### Option A: Live Microphone Capture")
            st.write("Captures 2.0 seconds of audio directly from your local recording device.")
            
            # Button to record
            if st.button("🎙️ Record 2 Seconds", type="primary", use_container_width=True):
                try:
                    import sounddevice as sd
                    st.info("Recording... Speak now!")
                    
                    sr = 16000
                    duration = 2.0
                    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
                    
                    # Add a simple countdown
                    bar = st.progress(0)
                    for step in range(1, 101):
                        time.sleep(duration / 100)
                        bar.progress(step)
                        
                    sd.wait()
                    st.success("Recording complete!")
                    
                    # Save recording temporarily
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_rec:
                        sf.write(tmp_rec.name, recording.flatten(), sr)
                        tmp_rec_path = tmp_rec.name
                        
                    st.audio(tmp_rec_path, format="audio/wav")
                    
                    with st.spinner("Processing live channel data..."):
                        results = analyze_voice_clip(
                            tmp_rec_path, l1_model, verifier, voiceprints, 
                            selected_speaker, l1_threshold, l2_threshold
                        )
                        
                    os.unlink(tmp_rec_path)
                    st.session_state['live_results'] = results
                except Exception as ex:
                    st.error(f"Could not initialize audio device: {ex}")
                    st.info("💡 Tip: On virtual/remote hosts or machines without mics, use Option B below to simulate the exact live stream.")
                    
        with col_rec_right:
            st.markdown("### Option B: Intercept Pre-Packaged Call Flows")
            st.write("Simulates incoming voice packets arriving at the bank's digital telephony channel (Asterisk ARI).")
            
            # Select demo scenario
            scenario_selected = st.selectbox(
                "Choose simulated call vector:",
                options=[
                    "Real Customer Call (Vishal)",
                    "Deepfake Attack - ElevenLabs Clone (Vishal)",
                    "Real Customer Call (Abhinav)",
                    "Deepfake Attack - ElevenLabs Clone (Abhinav)",
                    "Real Customer Call (Aditya)",
                    "Deepfake Attack - ElevenLabs Clone (Aditya)",
                ]
            )
            
            if st.button("⚡ Intercept Stream", use_container_width=True):
                # Mapping scenarios to files
                scenario_map = {
                    "Real Customer Call (Vishal)": ("data/real_voices/vishal/vishal_001.wav", "vishal"),
                    "Deepfake Attack - ElevenLabs Clone (Vishal)": ("data/clones/vishal/vishal_clone_001.wav", "vishal"),
                    "Real Customer Call (Abhinav)": ("data/real_voices/abhinav/abhinav_001.wav", "abhinav"),
                    "Deepfake Attack - ElevenLabs Clone (Abhinav)": ("data/clones/abhinav/abhinav_clone_001.wav", "abhinav"),
                    "Real Customer Call (Aditya)": ("data/real_voices/aditya/aditya_001.wav", "aditya"),
                    "Deepfake Attack - ElevenLabs Clone (Aditya)": ("data/clones/aditya/aditya_clone_001.wav", "aditya"),
                }
                
                path, speaker = scenario_map[scenario_selected]
                
                if os.path.exists(path):
                    st.audio(path, format="audio/wav")
                    
                    with st.spinner("Analyzing call stream packets..."):
                        results = analyze_voice_clip(
                            path, l1_model, verifier, voiceprints, 
                            speaker, l1_threshold, l2_threshold
                        )
                    st.session_state['live_results'] = results
                else:
                    st.error("Scenario audio files not found. Generate simulated data first.")
                    
        # RENDER LIVE SCAN RESULTS
        if 'live_results' in st.session_state:
            st.divider()
            st.markdown("## 📡 Telephony Stream Analysis")
            
            l_res = st.session_state['live_results']
            
            # Verdict Card
            rl = l_res['risk_level']
            if rl == "CLEAN":
                st.markdown(f'<div class="verdict-card verdict-clean">🟢 CALL PASSED: Verified Identity ({l_res["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
            elif rl == "SUSPICIOUS":
                st.markdown(f'<div class="verdict-card verdict-suspicious">🟡 CALL WARNING: Enhanced Audit Active ({l_res["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
            elif rl == "HIGH RISK":
                st.markdown(f'<div class="verdict-card verdict-highrisk">🟠 SECURE CHECKPOINT: Injecting OTP Verification ({l_res["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-card verdict-fraud">🔴 CALL TERMINATED: FRAUD INTERCEPTED ({l_res["risk_score"]:.1f}%)</div>', unsafe_allow_html=True)
                st.error(f"Blocked due to: {l_res['blocked_at']}")
                
            # Columns for values
            col_l1, col_l2, col_risk = st.columns(3)
            col_l1.metric("Layer 1 (AI Vocal Artifacts)", f"{l_res['ai_probability']*100:.1f}%")
            
            if l_res['layer2_score'] is not None:
                col_l2.metric("Layer 2 (Vocal Cosine Similarity)", f"{l_res['layer2_score']:.4f}")
            else:
                col_l2.metric("Layer 2 Status", "Bypassed")
                
            col_risk.metric("Total Fraud Risk Index", f"{l_res['risk_score']:.1f}%")

    # ------------------------------------------
    # TAB 3: SYSTEM ARCHITECTURE & THEORY
    # ------------------------------------------
    with tab3:
        st.subheader("Dual-Layer Auditing Architecture")
        
        # System Flowchart
        st.markdown("### System Processing Flow")
        st.markdown("""
        ```mermaid
        graph TD
            A[Incoming Audio Waveform] --> B[Standardize: 16kHz Mono 2.0s]
            B --> C[Compute Mel-Spectrogram]
            B --> D[Extract 5 Acoustic Signals]
            
            C --> E[Layer 1 CNN: MobileNetV3]
            D --> E
            
            E -->|Probability > Threshold| F[Layer 1 Block: AI Voice Detected]
            E -->|Probability <= Threshold| G[Layer 2 Encoder: ECAPA-TDNN]
            
            H[Stored Customer Voiceprint] --> I[Cosine Similarity Comparison]
            G --> I
            
            I -->|Score < Threshold| J[Layer 2 Block: Identity Mismatch]
            I -->|Score >= Threshold| K[Verify Account: CALL CLEAN]
            
            classDef fraud fill:#ffebee,stroke:#f43f5e,stroke-width:2px,color:#f43f5e;
            classDef clean fill:#e8f5e9,stroke:#10b981,stroke-width:2px,color:#10b981;
            class F,J fraud;
            class K clean;
        ```
        """)
        
        st.divider()
        
        # Detailed Technical Descriptions
        st.markdown("### Understanding the 5 Physics-Based Security Signals")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("""
            #### 1. Phase Jump Rate (⚡)
            Human speech is a continuous physical event (air forced from lungs through vibrating vocal cords).
            This creates a **smooth, continuous phase transition** along soundwaves.
            AI voices are synthesized by frame-based vocoders (typically calculating discrete 20ms audio frames).
            Because frames are calculated independently, **sudden phase jumps or discontinuities** occur at frame boundaries.
            PhaseGuard measures sudden frame-to-frame phase differences to locate these neural vocoder stitching seams.
            
            #### 2. Pitch Jitter (🎙️)
            Real voices wobble organically (muscle micro-tremors cause continuous fundamental frequency $f_0$ variations, typically between 120-220Hz).
            AI-cloned models generate speech that is either **acoustically static** (pitch stays perfectly constant, appearing flat) or has **artificial random noise** that lacks biological, structural modulation patterns.
            We use a YIN Pitch Estimator to calculate frame-wise variance.
            """)
            
        with col_t2:
            st.markdown("""
            #### 3. Spectral Flatness (🎚️)
            Acoustic flatness measures whether the frequency distribution resembles structured, harmonic speech (like high-energy formant peaks and vowel troughs) or flat white noise.
            AI vocoders sometimes smooth out spectral boundaries, yielding an **unnaturally uniform frequency distribution** over time.
            
            #### 4. Background Noise Floor (🔌)
            Phone calls contain atmospheric room noise, microphone hum, or cable hiss.
            When deepfake attacks are played digitally, they are synthesized in digital silence, exhibiting an **unnaturally low or absent noise floor** in the quiet frames between syllables.
            
            #### 5. MFCC Delta Variance (📊)
            Mel-Frequency Cepstral Coefficients (MFCCs) represent vocal tract textures.
            Their first derivatives (Delta values) depict how rapidly these textures transition.
            Authentic human speech has a standard dynamic variance as mouth shapes transform.
            Cloned networks display either highly uniform trajectories or abrupt frame jumps.
            """)
