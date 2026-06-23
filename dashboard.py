import sys
from unittest.mock import MagicMock
sys.modules['k2'] = MagicMock()
sys.modules['flair'] = MagicMock()
sys.modules['flair.data'] = MagicMock()
sys.modules['flair.embeddings'] = MagicMock()
sys.modules['flair.models'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['spacy.tokens'] = MagicMock()

import streamlit as st
import torch
import torchaudio
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
    page_title="PhaseGuard - Layer 1 AI Voice Authenticity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
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
    
    .title-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
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
    
    .verdict-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-weight: 800;
        font-size: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .verdict-clean {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.25) 100%);
        color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .verdict-fraud {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(225, 29, 72, 0.28) 100%);
        color: #f43f5e;
        border-color: rgba(244, 63, 94, 0.4);
        animation: pulse 2.0s infinite alternate;
    }
    
    .explain-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .explain-card-real { border-color: #10b981; }
    .explain-card-fake { border-color: #f43f5e; }
    .explain-card-neutral { border-color: #3b82f6; }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(244, 63, 94, 0.2); }
        100% { box-shadow: 0 0 25px rgba(244, 63, 94, 0.5); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED MODEL LOADER
# ==========================================
@st.cache_resource
def load_models_db():
    l1_model = PhaseGuardL1()
    l1_model.load_state_dict(torch.load("models/layer1_mobilenet.pth", map_location='cpu'))
    l1_model.eval()
    return l1_model

models_ready = os.path.exists("models/layer1_mobilenet.pth")

# ==========================================
# AUDIO PREDICTION PIPELINE
# ==========================================
def analyze_voice_clip(audio_path, l1_model, l1_thresh, enable_overrides=True):
    # Load full audio for physical features
    try:
        y_full, _ = librosa.load(audio_path, sr=16000)
    except Exception:
        y_full = None

    audio = load_and_standardize(audio_path)
    signals = extract_5_signals(y_full if y_full is not None else audio)

    mel = audio_to_mel_spectrogram(audio)
    if mel.shape[1] != 128:
        mel_resized = zoom(mel, (1, 128 / mel.shape[1]))
    else:
        mel_resized = mel

    mel_min = mel_resized.min()
    mel_max = mel_resized.max()
    mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-10)

    mel_tensor = torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        raw_cnn = float(l1_model(mel_tensor)[0][0])

    ai_probability = raw_cnn
    is_physically_real = False
    is_physically_fake = False
    override_reason = None

    if signals:
        # Case A: Very clean / studio real voice (low phase jumps and organic jitter)
        # Expanded jitter ceiling to 0.075 to cover controlled reading voices like LJSpeech
        if signals['phase_jump_rate'] < 0.11 and 0.0015 <= signals['jitter'] <= 0.075:
            is_physically_real = True
        # Case B: Compressed/echo-cancelled real voice (e.g., WhatsApp audio)
        # Raised noise floor threshold to 0.002, expanded jitter ceiling to 0.075
        elif signals['noise_floor'] > 0.002 and 0.0015 <= signals['jitter'] <= 0.075:
            if signals['phase_jump_rate'] < 0.23:
                is_physically_real = True
        # Case C: Noise-gated/edited real voice (e.g., edited in Audacity)
        elif signals['noise_floor'] <= 0.0006 and 0.0015 <= signals['jitter'] <= 0.075:
            if signals['phase_jump_rate'] < 0.14:
                is_physically_real = True

        # Fake overrides — only the definitive signal: near-zero digital silence
        # Real recordings always have background energy. AI voices are born in digital silence.
        has_ai_jitter = (signals['jitter'] < 0.0012) or (signals['jitter'] > 0.055)
        if signals['phase_jump_rate'] > 0.08 and signals['noise_floor'] < 0.0005 and has_ai_jitter:
            is_physically_fake = True

    if enable_overrides:
        if is_physically_real and raw_cnn > l1_thresh:
            # Only override to REAL if CNN is NOT extremely confident about fake (< 85%)
            # High-quality GAN fakes (WaveFake) can copy acoustic properties of the original voice,
            # so if CNN is > 85% sure it's fake, we trust the CNN over the physics rules.
            if raw_cnn < 0.85:
                ai_probability = min(raw_cnn, 0.12)
                override_reason = "Physics → REAL (organic jitter + low phase jumps)"
            # else: CNN > 85% confident fake — trust CNN, no override
        elif is_physically_fake and raw_cnn < l1_thresh:
            ai_probability = max(raw_cnn, 0.88)
            override_reason = "Physics → FAKE (AI vocoder artifacts + digital silence)"

    layer1_blocked = ai_probability > l1_thresh
    risk_score = ai_probability * 100

    if layer1_blocked:
        risk_level = "FRAUD ALERT"
    else:
        if risk_score < 30.0:
            risk_level = "CLEAN"
        elif risk_score < 60.0:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "HIGH RISK"

    return {
        'signals': signals,
        'mel_spectrogram': mel_norm,
        'mel_raw': mel_resized,
        'raw_cnn': raw_cnn,
        'ai_probability': ai_probability,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'layer1_blocked': layer1_blocked,
        'is_physically_real': is_physically_real,
        'is_physically_fake': is_physically_fake,
        'override_reason': override_reason,
        'audio': audio,
        'y_full': y_full if y_full is not None else audio,
    }

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="title-banner">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛡️ PhaseGuard — Layer 1</h1>
    <p style="margin: 5px 0 0 0; font-size: 1.1rem; color: #94a3b8; font-weight: 400;">
        AI vs Real Voice Detection · MobileNetV3 CNN + Physics Override Engine
    </p>
    <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #64748b; font-style: italic;">
        UCO Bank Hackathon 2026 — Team Ozymandias
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOAD
# ==========================================
if not models_ready:
    st.error("🚨 Model not found: `models/layer1_mobilenet.pth`. Run `train_layer1.py` first.")
    st.stop()

with st.spinner("Loading PhaseGuard Layer 1 CNN..."):
    l1_model = load_models_db()
st.sidebar.success("✅ Layer 1 CNN Active (MobileNetV3 Small)")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    l1_threshold = st.slider(
        "Block Threshold (AI Probability)",
        min_value=0.40, max_value=0.95, value=0.50, step=0.05,
        help="If AI probability > threshold, call is flagged as FAKE."
    )
    enable_overrides = st.checkbox(
        "Enable Physics Overrides",
        value=True,
        help="Apply heuristic checks on Phase Jump Rate, Jitter, and Noise Floor to override CNN. Enabled by default."
    )
    st.markdown("---")
    st.markdown("### 📖 Signal Quick Reference")
    st.markdown("""
| Signal | Real Range | AI Range |
|---|---|---|
| Phase Jump Rate | < 0.11 | > 0.08 |
| Pitch Jitter | 0.0015–0.05 | < 0.0012 or > 0.055 |
| Noise Floor | > 0.002 | < 0.0005 |
| Spectral Flatness | < 0.05 | Variable |
| MFCC Delta Var | 5–120 | Rigid or Jumpy |
""")

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyse", "🎙️ Pre-loaded Samples", "📚 How It Works"])

# ------------------------------------------
# TAB 1: FILE UPLOAD
# ------------------------------------------
with tab1:
    st.subheader("Upload an Audio File")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse a WAV / MP3 / FLAC file",
        type=["wav", "mp3", "flac"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if st.button("🔍 Run PhaseGuard Scan", type="primary", use_container_width=True):
            prog = st.progress(0)
            stat = st.empty()

            stat.info("Step 1/3 — Resampling & standardizing audio...")
            prog.progress(30)
            time.sleep(0.15)

            stat.info("Step 2/3 — Extracting 5 acoustic physical signals...")
            prog.progress(65)
            time.sleep(0.15)

            stat.info("Step 3/3 — Running MobileNetV3 CNN inference...")
            prog.progress(90)

            results = analyze_voice_clip(tmp_path, l1_model, l1_threshold, enable_overrides)
            prog.progress(100)
            stat.success("Scan complete!")
            time.sleep(0.3)
            stat.empty()
            prog.empty()
            os.unlink(tmp_path)

            # ---- VERDICT BANNER ----
            rl = results['risk_level']
            cert = results['risk_score'] if results['layer1_blocked'] else 100 - results['risk_score']
            if rl == "CLEAN":
                st.markdown(f'<div class="verdict-card verdict-clean">✅ REAL HUMAN VOICE &nbsp;·&nbsp; {cert:.1f}% Certainty</div>', unsafe_allow_html=True)
            elif rl == "SUSPICIOUS":
                st.markdown(f'<div class="verdict-card verdict-suspicious">⚠️ SUSPICIOUS — VERIFY CALLER &nbsp;·&nbsp; {results["risk_score"]:.1f}% AI Probability</div>', unsafe_allow_html=True)
            elif rl == "HIGH RISK":
                st.markdown(f'<div class="verdict-card verdict-suspicious">🟠 HIGH RISK DEEPFAKE &nbsp;·&nbsp; {results["risk_score"]:.1f}% AI Probability</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-card verdict-fraud">🚨 FAKE / AI VOICE CLONE &nbsp;·&nbsp; {cert:.1f}% Certainty</div>', unsafe_allow_html=True)

            # ---- OVERRIDE BANNER ----
            if results['override_reason']:
                st.warning(f"⚡ Physics Override Applied: {results['override_reason']}")
                col_cnn, col_final = st.columns(2)
                col_cnn.metric("CNN Raw Output", f"{results['raw_cnn']*100:.2f}%", help="MobileNetV3 sigmoid raw score")
                col_final.metric("Final AI Probability (after override)", f"{results['ai_probability']*100:.2f}%")
            else:
                st.metric("CNN Output = Final AI Probability", f"{results['raw_cnn']*100:.2f}%")

            st.divider()

            # ---- GAUGE + RADAR ----
            col_gauge, col_radar = st.columns(2)

            with col_gauge:
                st.markdown("#### 🎯 AI Probability Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['ai_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "AI Voice Probability (%)", 'font': {'size': 16, 'color': 'white'}},
                    number={'font': {'color': '#f43f5e' if results['layer1_blocked'] else '#10b981', 'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#f43f5e" if results['layer1_blocked'] else "#10b981"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.12)'},
                            {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.10)'},
                            {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.12)'},
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': l1_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    height=280,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_radar:
                st.markdown("#### 🕸️ Signal Radar vs Real/AI Reference")
                sigs = results['signals']

                # Normalize signals 0-1 for radar (higher = more AI-like)
                def norm_pjr(v): return min(v / 0.20, 1.0)                  # 0→0, 0.20→1
                def norm_jitter_ai(v): return 1.0 - min(max((v - 0.0012) / (0.055 - 0.0012), 0), 1)  # low jitter = high AI
                def norm_noise_ai(v): return max(1.0 - v / 0.005, 0.0)      # near-zero noise = high AI
                def norm_flat(v): return min(v / 0.10, 1.0)
                def norm_mfcc_ai(v): return 1.0 - min(abs(v - 30) / 90, 1.0)  # rigid MFCC = high AI

                user_vals = [
                    norm_pjr(sigs['phase_jump_rate']),
                    norm_jitter_ai(sigs['jitter']),
                    norm_noise_ai(sigs['noise_floor']),
                    norm_flat(sigs['spectral_flatness']),
                    norm_mfcc_ai(sigs['mfcc_delta_var']),
                ]
                real_ref = [0.10, 0.15, 0.20, 0.20, 0.15]
                fake_ref = [0.85, 0.90, 0.90, 0.55, 0.75]
                categories = ['Phase Jump', 'Jitter (AI)', 'Noise Floor (AI)', 'Spectral Flat', 'MFCC Rigid']

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=real_ref + [real_ref[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Typical REAL',
                    line=dict(color='#10b981', width=2),
                    fillcolor='rgba(16, 185, 129, 0.08)'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=fake_ref + [fake_ref[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Typical AI FAKE',
                    line=dict(color='#f43f5e', width=2),
                    fillcolor='rgba(244, 63, 94, 0.08)'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=user_vals + [user_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='Your Audio',
                    line=dict(color='#a855f7', width=3),
                    fillcolor='rgba(168, 85, 247, 0.15)'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 1], color='#475569', gridcolor='rgba(255,255,255,0.06)'),
                        angularaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.06)')
                    ),
                    showlegend=True,
                    legend=dict(font=dict(color='white', size=11), bgcolor='rgba(0,0,0,0)'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    height=280,
                    margin=dict(l=50, r=50, t=20, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()

            # ---- 5 SIGNAL TELEMETRY ----
            st.markdown("### 📡 5-Signal Telemetry Board")
            st.caption("Each signal measured from your audio vs typical human / AI ranges.")

            sigs = results['signals']
            c_phase = "anomaly" if sigs['phase_jump_rate'] > 0.08 else "normal"
            c_jitter = "anomaly" if (sigs['jitter'] < 0.0012 or sigs['jitter'] > 0.055) else "normal"
            c_flat = "anomaly" if sigs['spectral_flatness'] > 0.05 else "normal"
            c_noise = "anomaly" if sigs['noise_floor'] < 0.0005 else "normal"
            c_delta = "anomaly" if sigs['mfcc_delta_var'] < 5.0 or sigs['mfcc_delta_var'] > 120.0 else "normal"

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.markdown(f"""<div class="sensor-card">
                <div class="sensor-title">⚡ Phase Jump Rate</div>
                <div class="sensor-value" style="color: {'#f43f5e' if c_phase=='anomaly' else '#10b981'}">{sigs['phase_jump_rate']:.4f}</div>
                <div class="sensor-status-{c_phase}">{'Vocoder Seams' if c_phase=='anomaly' else 'Smooth Flow'}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px">Real: &lt; 0.11</div>
            </div>""", unsafe_allow_html=True)

            col2.markdown(f"""<div class="sensor-card">
                <div class="sensor-title">🎙️ Pitch Jitter</div>
                <div class="sensor-value" style="color: {'#f43f5e' if c_jitter=='anomaly' else '#10b981'}">{sigs['jitter']:.5f}</div>
                <div class="sensor-status-{c_jitter}">{'AI Artifact' if c_jitter=='anomaly' else 'Organic Jitter'}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px">Real: 0.0015–0.050</div>
            </div>""", unsafe_allow_html=True)

            col3.markdown(f"""<div class="sensor-card">
                <div class="sensor-title">🎚️ Spectral Flatness</div>
                <div class="sensor-value" style="color: {'#f43f5e' if c_flat=='anomaly' else '#10b981'}">{sigs['spectral_flatness']:.4f}</div>
                <div class="sensor-status-{c_flat}">{'Synthetic' if c_flat=='anomaly' else 'Vocal Formants'}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px">Real: &lt; 0.05</div>
            </div>""", unsafe_allow_html=True)

            col4.markdown(f"""<div class="sensor-card">
                <div class="sensor-title">🔌 Noise Floor (RMS)</div>
                <div class="sensor-value" style="color: {'#f43f5e' if c_noise=='anomaly' else '#10b981'}">{sigs['noise_floor']:.6f}</div>
                <div class="sensor-status-{c_noise}">{'Digital Silence' if c_noise=='anomaly' else 'Natural Room Noise'}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px">Real: &gt; 0.002</div>
            </div>""", unsafe_allow_html=True)

            col5.markdown(f"""<div class="sensor-card">
                <div class="sensor-title">📊 MFCC Delta Var</div>
                <div class="sensor-value" style="color: {'#f43f5e' if c_delta=='anomaly' else '#10b981'}">{sigs['mfcc_delta_var']:.2f}</div>
                <div class="sensor-status-{c_delta}">{'Rigid/Abrupt' if c_delta=='anomaly' else 'Dynamic Range'}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px">Real: 5 – 120</div>
            </div>""", unsafe_allow_html=True)

            st.divider()

            # ---- WHY REAL / WHY FAKE EXPLANATION CARDS ----
            st.markdown("### 🔬 Signal-by-Signal Explanation")
            is_fake = results['layer1_blocked']

            def explain_card(title, value_str, verdict, explanation, card_type):
                color = "#f43f5e" if card_type == "fake" else ("#10b981" if card_type == "real" else "#3b82f6")
                icon = "🔴" if card_type == "fake" else ("🟢" if card_type == "real" else "🔵")
                return f"""<div class="explain-card explain-card-{card_type}">
                    <strong style="color:{color}">{icon} {title}</strong><br>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.9rem">{value_str}</span><br>
                    <span style="color:#94a3b8;font-size:0.88rem">{verdict} — {explanation}</span>
                </div>"""

            # Phase Jump Rate explanation
            pjr = sigs['phase_jump_rate']
            if pjr > 0.08:
                pjr_html = explain_card("Phase Jump Rate", f"{pjr:.4f}",
                    "⚠️ ELEVATED",
                    "AI vocoders generate speech frame-by-frame (every 20ms). Each frame boundary causes a sudden phase discontinuity — a \"seam\" invisible to ears but measurable mathematically. Real vocal cords produce continuous phase flow.",
                    "fake")
            else:
                pjr_html = explain_card("Phase Jump Rate", f"{pjr:.4f}",
                    "✅ NORMAL",
                    "Phase transitions are smooth and continuous — consistent with a real human vocal tract producing uninterrupted airflow through vibrating cords.",
                    "real")

            # Jitter explanation
            jitter = sigs['jitter']
            if jitter < 0.0012:
                jit_html = explain_card("Pitch Jitter", f"{jitter:.6f}",
                    "⚠️ TOO FLAT (Robotic)",
                    "Jitter is near-zero, meaning pitch is unnaturally constant. Human vocal cords have micro-tremors that cause slight frequency wobble. AI TTS systems tend to produce perfectly flat pitch — a telltale digital artifact.",
                    "fake")
            elif jitter > 0.055:
                jit_html = explain_card("Pitch Jitter", f"{jitter:.6f}",
                    "⚠️ TOO ERRATIC (Synthetic noise)",
                    "Jitter exceeds normal human range. This level of pitch variation is characteristic of poorly calibrated TTS vocoders or voice conversion artifacts that introduce random noise into pitch modulation.",
                    "fake")
            else:
                jit_html = explain_card("Pitch Jitter", f"{jitter:.6f}",
                    "✅ ORGANIC",
                    "Pitch varies naturally within the human biological range. This is consistent with muscle micro-tremors in real vocal cords — a difficult-to-fake biometric.",
                    "real")

            # Noise floor explanation
            nf = sigs['noise_floor']
            if nf < 0.0005:
                nf_html = explain_card("Noise Floor (RMS)", f"{nf:.6f}",
                    "⚠️ DIGITAL SILENCE",
                    "Near-zero energy in the quietest frames. Real recordings always contain room noise, breath sounds, or mic hiss. This near-perfect silence is a strong indicator of a digitally synthesized voice played from a device.",
                    "fake")
            else:
                nf_html = explain_card("Noise Floor (RMS)", f"{nf:.6f}",
                    "✅ NATURAL",
                    "Background energy is present, consistent with real-world recording conditions (room acoustics, microphone hiss, or environmental noise). AI-generated voices typically have near-zero background energy.",
                    "real")

            # Spectral flatness explanation
            sf_val = sigs['spectral_flatness']
            if sf_val > 0.05:
                sf_html = explain_card("Spectral Flatness", f"{sf_val:.4f}",
                    "⚠️ FLAT SPECTRUM",
                    "Frequency energy is unusually spread (noise-like). Real speech has strong formant peaks (F1, F2, F3) giving it a non-flat spectrum. Excessive flatness suggests synthetic smoothing by a TTS model.",
                    "fake")
            else:
                sf_html = explain_card("Spectral Flatness", f"{sf_val:.4f}",
                    "✅ HARMONIC",
                    "Strong harmonic content with clear formant structure — consistent with real speech resonating in a human vocal tract.",
                    "real")

            # MFCC explanation
            mdv = sigs['mfcc_delta_var']
            if mdv < 5.0:
                mfcc_html = explain_card("MFCC Delta Variance", f"{mdv:.2f}",
                    "⚠️ RIGID",
                    "Spectral texture transitions too slowly. This indicates a voice that doesn't change its mouth/throat shape naturally — a sign of rigid AI generation.",
                    "fake")
            elif mdv > 120.0:
                mfcc_html = explain_card("MFCC Delta Variance", f"{mdv:.2f}",
                    "⚠️ ABRUPT JUMPS",
                    "Spectral texture changes violently between frames — consistent with vocoder frame boundary artifacts creating sudden discontinuities.",
                    "fake")
            else:
                mfcc_html = explain_card("MFCC Delta Variance", f"{mdv:.2f}",
                    "✅ DYNAMIC",
                    "Spectral texture transitions vary naturally — consistent with continuous articulation of a real human speaker.",
                    "real")

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown(pjr_html, unsafe_allow_html=True)
                st.markdown(jit_html, unsafe_allow_html=True)
                st.markdown(nf_html, unsafe_allow_html=True)
            with col_exp2:
                st.markdown(sf_html, unsafe_allow_html=True)
                st.markdown(mfcc_html, unsafe_allow_html=True)

            st.divider()

            # ---- VISUAL CHARTS ----
            st.markdown("### 📈 Acoustic Visualizations")
            col_w, col_s = st.columns(2)

            with col_w:
                st.markdown("**Waveform + RMS Energy (10th Pct = Noise Floor)**")
                y_disp = results['audio']
                t = np.arange(len(y_disp)) / 16000
                rms_frames = librosa.feature.rms(y=y_disp, frame_length=512, hop_length=160)[0]
                t_rms = librosa.frames_to_time(np.arange(len(rms_frames)), sr=16000, hop_length=160)

                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(
                    x=t, y=y_disp, name="Waveform",
                    line=dict(color='rgba(59, 130, 246, 0.5)', width=0.8)
                ))
                fig_wave.add_trace(go.Scatter(
                    x=t_rms, y=rms_frames, name="RMS Energy",
                    line=dict(color='#a855f7', width=2)
                ))
                nf_val = float(np.percentile(rms_frames, 10))
                fig_wave.add_hline(
                    y=nf_val, line_dash="dot", line_color="#f43f5e",
                    annotation_text=f"Noise Floor: {nf_val:.6f}",
                    annotation_font_color="#f43f5e"
                )
                fig_wave.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    height=260, margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11))
                )
                fig_wave.update_xaxes(title="Time (s)", color="#475569", showgrid=False)
                fig_wave.update_yaxes(title="Amplitude", color="#475569", gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig_wave, use_container_width=True)

            with col_s:
                st.markdown("**Mel-Spectrogram (128×128 CNN Input)**")
                fig_spec = px.imshow(
                    results['mel_spectrogram'],
                    labels=dict(x="Time Frames", y="Mel Frequency Bins"),
                    color_continuous_scale='Viridis',
                    aspect='auto'
                )
                fig_spec.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    height=260, margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_spec, use_container_width=True)

            # ---- PHASE JUMP HEATMAP ----
            st.markdown("**Phase Acceleration Heatmap (2nd-order Phase Difference — AI Seams Visible as Bright Bands)**")
            y_phase = results['y_full']
            stft = librosa.stft(y_phase, n_fft=512, hop_length=160)
            phase = np.angle(stft)
            pd1 = np.diff(phase, axis=1)
            pd1w = np.arctan2(np.sin(pd1), np.cos(pd1))
            pd2 = np.diff(pd1w, axis=1)
            pd2w = np.arctan2(np.sin(pd2), np.cos(pd2))
            heatmap_data = np.abs(pd2w[:96, :])   # voiced range only

            fig_phase = px.imshow(
                heatmap_data,
                labels=dict(x="Time Frames", y="Frequency Bins (0–3kHz)"),
                color_continuous_scale=[
                    [0, "#0d0d15"], [0.4, "#2563eb"], [0.7, "#7c3aed"], [1.0, "#f43f5e"]
                ],
                aspect='auto',
                zmin=0, zmax=np.pi
            )
            fig_phase.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Outfit'},
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(title="Phase Accel (rad)", tickfont=dict(color='white'))
            )
            st.plotly_chart(fig_phase, use_container_width=True)
            st.caption("🔴 Bright vertical bands = sudden phase jumps at vocoder frame boundaries. Real human voices show uniform low-level noise. AI voices show structured red bands.")

            # ---- SIGNAL BAR CHART ----
            st.markdown("**Signal Comparison: Your Audio vs Typical Real / AI Ranges**")
            bar_signals = ['Phase Jump Rate', 'Pitch Jitter ×100', 'Noise Floor ×1000', 'Spectral Flatness', 'MFCC δVar ÷10']
            bar_user = [
                sigs['phase_jump_rate'],
                sigs['jitter'] * 100,
                sigs['noise_floor'] * 1000,
                sigs['spectral_flatness'],
                sigs['mfcc_delta_var'] / 10
            ]
            bar_real_max = [0.11, 5.0, 5.0, 0.05, 12.0]
            bar_fake_min = [0.08, 0.0, 0.0, 0.05, 0.0]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Your Audio",
                x=bar_signals, y=bar_user,
                marker_color=['#f43f5e' if results['layer1_blocked'] else '#10b981'] * 5,
                opacity=0.85
            ))
            fig_bar.add_trace(go.Bar(
                name="Typical Real Upper Limit",
                x=bar_signals, y=bar_real_max,
                marker_color='rgba(16,185,129,0.2)',
                marker_line=dict(color='#10b981', width=2)
            ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Outfit'},
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
                xaxis=dict(color='#475569', showgrid=False),
                yaxis=dict(color='#475569', gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 2: PRE-LOADED SAMPLES
# ------------------------------------------
with tab2:
    st.subheader("Pre-loaded Voice Samples")
    st.write("Test with known real and cloned voices from the team.")

    scenario_map = {
        "Real — Vishal (Sample 1)": "data/real_voices/vishal/vishal_001.wav",
        "AI Clone — Vishal (Sample 1)": "data/clones/vishal/vishal_clone_001.wav",
        "Real — Abhinav (Sample 1)": "data/real_voices/abhinav/abhinav_001.wav",
        "AI Clone — Abhinav (Sample 1)": "data/clones/abhinav/abhinav_clone_001.wav",
        "Real — Aditya (Sample 1)": "data/real_voices/aditya/aditya_001.wav",
        "AI Clone — Aditya (Sample 1)": "data/clones/aditya/aditya_clone_001.wav",
        "Real — Dhruv (Sample 1)": "data/real_voices/dhruv/dhruv_001.wav",
        "AI Clone — Dhruv (Sample 1)": "data/clones/dhruv/dhruv_clone_001.wav",
    }

    scenario_selected = st.selectbox("Choose a sample:", list(scenario_map.keys()))
    path = scenario_map[scenario_selected]

    if st.button("⚡ Run Scan on Sample", type="primary", use_container_width=True):
        if os.path.exists(path):
            st.audio(path, format="audio/wav")
            with st.spinner("Analyzing..."):
                r2 = analyze_voice_clip(path, l1_model, l1_threshold, enable_overrides)

            rl2 = r2['risk_level']
            cert2 = r2['risk_score'] if r2['layer1_blocked'] else 100 - r2['risk_score']
            if rl2 == "CLEAN":
                st.markdown(f'<div class="verdict-card verdict-clean">✅ REAL HUMAN VOICE · {cert2:.1f}% Certainty</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-card verdict-fraud">🚨 FAKE / AI VOICE CLONE · {cert2:.1f}% Certainty</div>', unsafe_allow_html=True)

            if r2['override_reason']:
                st.warning(f"⚡ Physics Override: {r2['override_reason']}")

            s2 = r2['signals']
            st.markdown("#### Signal Readings")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Phase Jump Rate", f"{s2['phase_jump_rate']:.4f}", delta="↑ AI" if s2['phase_jump_rate'] > 0.08 else "✓ OK")
            sc2.metric("Pitch Jitter", f"{s2['jitter']:.5f}", delta="⚠ AI" if (s2['jitter'] < 0.0012 or s2['jitter'] > 0.055) else "✓ OK")
            sc3.metric("Spectral Flatness", f"{s2['spectral_flatness']:.4f}")
            sc4.metric("Noise Floor", f"{s2['noise_floor']:.6f}", delta="⚠ Low" if s2['noise_floor'] < 0.0005 else "✓ OK")
            sc5.metric("MFCC Delta Var", f"{s2['mfcc_delta_var']:.2f}")
        else:
            st.error(f"Audio file not found: `{path}`")

# ------------------------------------------
# TAB 3: HOW IT WORKS
# ------------------------------------------
with tab3:
    st.subheader("How PhaseGuard Layer 1 Works")

    st.markdown("""
    ### Two-Engine Architecture

    PhaseGuard Layer 1 runs **two independent engines** on every audio file and combines their outputs:
    """)

    c_arch1, c_arch2 = st.columns(2)
    with c_arch1:
        st.markdown("""
        #### 🧠 Engine 1: CNN (MobileNetV3 Small)
        - Converts audio → **128×128 log Mel-spectrogram** (a "photograph" of the voice's frequency content over time)
        - Feeds it through MobileNetV3 Small (a compact image classifier)
        - Outputs a **sigmoid probability** (0.0 = Real, 1.0 = Fake)
        - Trained on **3,966 samples** (Common Voice, ASVspoof 2019, WaveFake, team recordings)
        - Achieved **86.8% test accuracy**
        - **Limitation:** Can be fooled by out-of-distribution high-quality TTS (e.g. HuggingFace)
        """)

    with c_arch2:
        st.markdown("""
        #### ⚡ Engine 2: Physics Override (Rule-Based)
        - Extracts **5 acoustic signals** directly from the waveform mathematics
        - Applies **bidirectional override rules**:
            - If Physics says REAL but CNN says FAKE → override to REAL
            - If Physics says FAKE but CNN says REAL → override to FAKE
        - Catches CNN blind spots (e.g. HuggingFace TTS demo.wav)
        - **Enabled by default** (disable via sidebar checkbox to see raw CNN only)
        """)

    st.markdown("---")
    st.markdown("### 🔬 The 5 Physical Signals")

    signals_info = [
        ("⚡ Phase Jump Rate", "< 0.11 (real)", "> 0.08 (AI)",
         "Human speech is a continuous physical process — air through vibrating vocal cords creates smooth phase flow. AI vocoders generate speech in 20ms frames independently. At each frame boundary, there is a sudden phase discontinuity — an invisible 'seam'. PhaseGuard measures these seams using 2nd-order wrapped phase differences."),
        ("🎙️ Pitch Jitter", "0.0015–0.050 (real)", "< 0.0012 or > 0.055 (AI)",
         "Human vocal cords vibrate with natural micro-tremors — muscles shake slightly, causing small F0 variations. AI TTS voices are either perfectly flat (zero variation, robotic) or unnaturally noisy (random pitch artifacts). Jitter in the organic human range is a difficult-to-forge biometric."),
        ("🔌 Noise Floor (RMS)", "> 0.002 (real)", "< 0.0005 (AI)",
         "Real recordings always contain ambient environment sounds: room reverberation, microphone self-noise, HVAC hum, breathing. AI-generated voices are created in perfect digital silence. Even when played through a speaker, the quiet frames between syllables are near-zero energy — unlike any natural recording environment."),
        ("🎚️ Spectral Flatness", "< 0.05 (real)", "> 0.05 (AI)",
         "Human speech has a strong harmonic structure — vocal tract resonances create peaks (formants F1, F2, F3) in the frequency spectrum. This gives a non-flat, structured spectrum. Some AI vocoders smooth these boundaries, creating a more noise-like uniform frequency distribution."),
        ("📊 MFCC Delta Variance", "5–120 (real)", "< 5 or > 120 (AI)",
         "MFCCs capture vocal tract texture. Their first-order derivatives capture how fast the texture changes. Real speech has natural variation in transition speed as mouth shapes form consonants and vowels. AI voices can be either too rigid (transitions too slow) or too jumpy (frame boundary artifacts creating abrupt changes)."),
    ]

    for name, real_r, ai_r, desc in signals_info:
        with st.expander(f"{name} — Real: {real_r} | AI: {ai_r}"):
            st.write(desc)

    st.markdown("---")
    st.markdown("### 📊 Dataset Used for Training")
    st.markdown("""
| Category | Dataset | Files |
|---|---|---|
| 🟢 Real | Mozilla Common Voice (diverse speakers) | 1,266 |
| 🟢 Real | ASVspoof 2019 LA Real (studio recordings) | 400 |
| 🟢 Real | ASVspoof 2019 PA Real (room acoustics) | 300 |
| 🟢 Real | Team own recordings (Abhinav, Aditya, Dhruv, Vishal) | ~200 |
| 🔴 Fake | WaveFake (HiFi-GAN, MelGAN, WaveGlow vocoders) | 600 |
| 🔴 Fake | ASVspoof 2019 LA Fake (19 TTS/VC attack systems) | 600 |
| 🔴 Fake | ASVspoof 2019 PA Fake (replay attacks) | 400 |
| 🔴 Fake | Team voice clones (same 4 speakers cloned via TTS) | ~200 |
| **Total** | | **~3,966 spectrograms** |
""")
