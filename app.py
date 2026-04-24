import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import os
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import shap
import noisereduce as nr
from datetime import datetime

# ================================================
# PAGE CONFIG — must be first Streamlit call
# ================================================
st.set_page_config(
    page_title="DeepShield — Audio Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# GLOBAL CSS — Dark Cyber Forensics Theme
# ================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #020812;
    --surface:   #060f1e;
    --card:      #0a1628;
    --border:    #0d2444;
    --accent:    #00c8ff;
    --accent2:   #00ff9d;
    --danger:    #ff3b5c;
    --warn:      #ffb020;
    --text:      #c8dff5;
    --muted:     #4a6480;
    --glow: none;
}

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide all Streamlit chrome completely ── */
#MainMenu           { display: none !important; }
footer              { display: none !important; }
header              { display: none !important; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
[data-testid="stStatusWidget"]  { display: none !important; }

/* ── Hide all sidebar elements — navigation is via top tabs ── */
[data-testid="stSidebar"]               { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"]{ display: none !important; }
section[data-testid="stSidebar"]        { display: none !important; }

/* ── Top navigation tabs ── */
[data-testid="stTabsHeader"] {
    background: transparent !important;
    border-bottom: 1px solid #0d2444 !important;
}
button[data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    color: #4a6480 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.6rem !important;
    border-radius: 0 !important;
    transition: color 0.2s !important;
}
button[data-baseweb="tab"]:hover {
    color: #00c8ff !important;
    background: rgba(0,200,255,0.05) !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: #00c8ff !important;
    border-bottom: 2px solid #00c8ff !important;
    background: rgba(0,200,255,0.06) !important;
}
[data-testid="stTabContent"] {
    padding-top: 1.2rem !important;
}

/* ── Plotly chart container alignment ── */
[data-testid="stPlotlyChart"] {
    background: #0a1628 !important;
    border: 1px solid #0d2444 !important;
    border-radius: 6px !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
    overflow: hidden !important;
}
[data-testid="stPlotlyChart"] > div {
    width: 100% !important;
}

/* ── Matplotlib figure alignment ── */
[data-testid="stImage"],
.stPlotlyChart,
.element-container figure,
.element-container img {
    width: 100% !important;
    display: block !important;
    margin: 0 auto 1rem auto !important;
}
figure {
    margin: 0 !important;
    width: 100% !important;
}

/* ── Column gap uniform ── */
[data-testid="column"] {
    padding: 0 0.5rem !important;
}
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* ── Main content top padding (no header means no gap needed) ── */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}


/* ── Hide sidebar completely — navigation is via top tabs ── */
[data-testid="stSidebar"]            { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]     { display: none !important; }

/* ── Top navigation tabs ── */
[data-testid="stTabs"] {
    margin-bottom: 0 !important;
}
[data-testid="stTabsHeader"] {
    background: transparent !important;
    border-bottom: 1px solid #0d2444 !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    color: #4a6480 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover {
    color: #00c8ff !important;
    background: rgba(0,200,255,0.05) !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: #00c8ff !important;
    border-bottom: 2px solid #00c8ff !important;
    background: rgba(0,200,255,0.06) !important;
}
[data-testid="stTabContent"] {
    padding-top: 1.2rem !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: var(--accent) !important;
    letter-spacing: 0.06em;
}
h1 { font-size: 1.6rem !important; font-weight: 900 !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; }

/* ── Subheaders → styled differently ── */
[data-testid="stMarkdownContainer"] p {
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    font-size: 0.92rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: rgba(0,200,255,0.05) !important;
    box-shadow: none !important;
}

/* ── Inputs ── */
input[type="text"], input[type="password"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(0,200,255,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* ── Success / Error / Warning ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    border-left-width: 3px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent2), var(--accent)) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--accent) !important;
}

/* ── Expander — minimal clean styling ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] > details > summary {
    padding: 0.8rem 1rem !important;
    color: var(--accent) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    list-style: none !important;
}

/* ── Audio player ── */
audio {
    filter: invert(0.85) hue-rotate(175deg) !important;
    width: 100% !important;
    height: 35px !important;
    border-radius: 4px !important;
    outline: none !important;
}

/* ── Plots ── */
[data-testid="stImage"] img,
canvas {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Custom card HTML ── */
.ds-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.ds-card-accent {
    border-left: 3px solid var(--accent);
}
.ds-card-danger {
    border-left: 3px solid var(--danger);
}
.ds-card-success {
    border-left: 3px solid var(--accent2);
}
.ds-card-warn {
    border-left: 3px solid var(--warn);
}
.ds-tag {
    display: inline-block;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.25);
    color: var(--accent);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin: 2px;
}
.ds-tag-danger {
    background: rgba(255,59,92,0.08);
    border-color: rgba(255,59,92,0.3);
    color: var(--danger);
}
.ds-tag-success {
    background: rgba(0,255,157,0.08);
    border-color: rgba(0,255,157,0.3);
    color: var(--accent2);
}
.mono {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-real {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--accent2);
    letter-spacing: 0.08em;
}
.result-fake {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--danger);
    letter-spacing: 0.08em;
}
.law-article {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--warn);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
}
.law-article h4 {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.78rem !important;
    color: var(--warn) !important;
    margin: 0 0 0.4rem 0 !important;
    letter-spacing: 0.08em;
}
.law-article p {
    font-size: 0.85rem !important;
    color: var(--text) !important;
    margin: 0 0 0.3rem 0 !important;
    line-height: 1.6 !important;
}
.law-article .penalty {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--danger);
}
.scan-line {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ================================================
# DEVICE SETUP
# Device 1 = Intel Smart Sound Array (MME) — input
# Device 3 = Realtek Speakers (MME)        — output
# ================================================
sd.default.device = (1, 3)

# ================================================
# LOGIN
# ================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-family:Orbitron,monospace; font-size:2.2rem;
                        font-weight:900; color:#00c8ff; letter-spacing:0.1em;'>
                🛡️ DEEPSHIELD
            </div>
            <div style='font-family:Share Tech Mono,monospace; font-size:0.8rem;
                        color:#4a6480; letter-spacing:0.2em; margin-top:0.3rem;'>
                AUDIO FORENSICS PLATFORM
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="ds-card ds-card-accent">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Secure Access</div>', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("AUTHENTICATE", use_container_width=True):
                with st.spinner("Authenticating..."):
                    time.sleep(0.6)
                if username == "admin" and password == "1234":
                    st.session_state.login = True
                    st.success("Access granted.")
                    st.rerun()
                else:
                    st.error("Authentication failed — invalid credentials.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class='mono' style='text-align:center; margin-top:1rem;'>
            Unauthorized access is prohibited under IT Act 2000 S.43
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ================================================
# LOAD MODEL
# ================================================
@st.cache_resource
def load_model():
    model  = joblib.load("deepfake_audio_xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ================================================
# SESSION STATE
# ================================================
for key in ["result", "confidence", "filename", "features_scaled", "audio_type", "lang_note", "fake_pct"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ================================================
# FEATURE NAMES
# ================================================
FEATURE_NAMES = (
    [f"MFCC_{i+1}"   for i in range(40)] +
    [f"dMFCC_{i+1}"  for i in range(40)] +
    [f"d2MFCC_{i+1}" for i in range(40)] +
    ["SpectralCentroid", "ZCR", "SpectralBandwidth"]
)


# ================================================
# FEATURE EXTRACTION
# ================================================
# ================================================
# IMPROVEMENT 1: AUDIO TYPE CLASSIFIER
# Detects: human speech, music, animal, cough,
# threat/scream, silence, environmental
# Uses rule-based acoustic analysis — no extra model needed
# ================================================
# ─── SPEECH TYPES: deepfake model is valid for these ───
SPEECH_TYPES = {"Human Speech", "Cartoon / Synthetic Voice", "Whisper / Soft Speech"}

# ─── NON-SPEECH TYPES: model results unreliable ───
NON_SPEECH_TYPES = {
    "Music / Singing", "Animal Sound", "Cough / Breath",
    "Scream / Threat", "Environmental / Noise", "Silence"
}


def classify_audio_type(audio, sr):
    """
    Returns (type_label, confidence_pct, note, is_speech)
    FINAL FIXED VERSION — balanced & robust
    """

    peak = np.max(np.abs(audio)) + 1e-8
    audio_n = audio / peak

    rms = float(np.sqrt(np.mean(audio_n ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_n)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_n, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio_n, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio_n)))

    harmonic, percussive = librosa.effects.hpss(audio_n)
    harm_ratio = float(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-8))

    mfcc = librosa.feature.mfcc(y=audio_n, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

    onset_frames = librosa.onset.onset_detect(y=audio_n, sr=sr)
    onset_rate = len(onset_frames) / max(len(audio_n)/sr, 0.1)

    # RULES

    if rms < 0.015:
        return "Silence", 95, "No audio detected", False

   # MUSIC
    if harm_ratio > 2.5 and bandwidth > 2500:
         return "Music / Singing", 80, "Model not applicable to music", False

# ANIMAL SOUND (FIXED — stricter)
    if (harm_ratio < 0.5 and onset_rate > 5 and zcr > 0.13
        and 1000 < centroid < 6000 and flatness > 0.05):
        return "Animal Sound", 75, "Model not applicable to animal sounds", False

    if flatness > 0.30 and harm_ratio < 0.5:
        return "Environmental / Noise", 70, "Model not applicable to noise", False

    if onset_rate > 5 and zcr > 0.1:
        return "Cough / Breath", 78, "Model not applicable", False

    if mfcc_var < 6 and harm_ratio > 0.5:
        return "Synthetic / AI Voice", 70, "Possible synthetic speech", True

    # ✅ MAIN FIX — relaxed speech detection
    if (
        zcr > 0.02 and zcr < 0.20 and
        centroid > 500 and centroid < 5000 and
        harm_ratio > 0.4
    ):
        return "Human Speech", 85, "Speech detected", True

    # ✅ FINAL FALLBACK (VERY IMPORTANT)
    return "Uncertain (Speech-like)", 60, "Proceeding with analysis", True

# ================================================
# IMPROVEMENT 2: LONG AUDIO SEGMENTATION
# Slices audio into 3s chunks, runs model on each,
# returns majority vote + per-segment timeline
# ================================================
def analyse_long_audio(file_path, scaler, model):
    """
    Handles audio of any length up to ~1 hour.
    Splits into 3s segments, classifies each, returns:
      - overall verdict (majority vote)
      - overall confidence (mean)
      - segment_results list [(start_sec, end_sec, label, confidence)]
    """
    audio_full, sr = librosa.load(file_path, sr=16000, mono=True)
    total_duration  = len(audio_full) / sr
    segment_len     = 3 * sr   # 3 seconds in samples
    step            = segment_len  # no overlap for speed; use segment_len//2 for overlap

    segment_results = []
    real_votes      = 0
    fake_votes      = 0
    confidences     = []

    num_segments = max(1, int(np.ceil(len(audio_full) / segment_len)))

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for i in range(num_segments):
        start = i * step
        end   = min(start + segment_len, len(audio_full))
        seg   = audio_full[start:end]

        # Pad last segment if short
        if len(seg) < segment_len:
            seg = np.pad(seg, (0, segment_len - len(seg)))

        # Skip near-silent segments
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < 0.008:
            segment_results.append((start/sr, end/sr, "Silent", 0))
            continue

        # Noise reduction + normalize
        seg = nr.reduce_noise(y=seg, sr=sr)
        seg = seg / (np.max(np.abs(seg)) + 1e-6)

        # Extract features
        mfcc        = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        feat = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(mfcc_delta.T, axis=0),
            np.mean(mfcc_delta2.T, axis=0),
            np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(seg)),
            np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))
        ]).reshape(1, -1)

        feat_scaled  = scaler.transform(feat)
        pred         = model.predict(feat_scaled)[0]
        prob         = np.max(model.predict_proba(feat_scaled)) * 100
        label        = "Real" if pred == 0 else "Deepfake"

        if pred == 0:
            real_votes += 1
        else:
            fake_votes += 1

        confidences.append(prob)
        segment_results.append((round(start/sr, 1), round(end/sr, 1), label, round(prob, 1)))

        # Update progress
        progress_bar.progress(min(int((i+1)/num_segments * 100), 100))
        status_text.markdown(
            f"<div class='mono'>Analysing segment {i+1}/{num_segments} "
            f"({start/sr:.0f}s – {end/sr:.0f}s)...</div>",
            unsafe_allow_html=True
        )

    progress_bar.empty()
    status_text.empty()

    total_votes   = real_votes + fake_votes
    overall_label = "Deepfake Audio" if fake_votes > real_votes else "Real Audio (Bonafide)"
    overall_conf  = round(np.mean(confidences), 2) if confidences else 0
    fake_pct      = round(fake_votes / total_votes * 100, 1) if total_votes > 0 else 0

    return overall_label, overall_conf, segment_results, fake_pct, total_duration


def show_segment_timeline(segment_results, total_duration):
    """Render a colour-coded segment timeline chart."""
    if not segment_results:
        return

    fig, ax = plt.subplots(figsize=(12, 1.6))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#060f1e')

    for (start, end, label, conf) in segment_results:
        color = '#00ff9d' if label == "Real" else '#ff3b5c' if label == "Deepfake" else '#2a3a50'
        ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.9)

    ax.set_xlim(0, total_duration)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=8, color='#4a6480')
    ax.set_title("SEGMENT TIMELINE  —  Green = Real  |  Red = Deepfake",
                 fontsize=8, color='#4a6480', fontfamily='monospace')
    ax.tick_params(colors='#4a6480', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#0d2444')
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def extract_features(file_path):
    # ── Pre-processing log ──
    log = st.empty()
    def log_step(msg):
        log.markdown(
            f"<div class='mono' style='color:#4a6480; font-size:0.75rem;'>"
            f"⟳ {msg}</div>",
            unsafe_allow_html=True
        )

    log_step("Loading audio file...")
    raw_audio, raw_sr = librosa.load(file_path, sr=None, mono=False)
    orig_sr      = raw_sr
    orig_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[0]

    log_step(f"Original: {orig_sr} Hz, {orig_channels} channel(s) — resampling to 16000 Hz mono...")
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    log_step("Applying noise reduction...")
    audio = nr.reduce_noise(y=audio, sr=sr)

    log_step("Checking signal energy...")
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        log.empty()
        return None

    log_step("Generating visualizations...")
    log.empty()  # clear the log once done

    # ── Interactive Plotly waveform (zoomable) ──
    st.markdown("<div class='section-label'>Audio Waveform — scroll to zoom, drag to pan</div>",
                unsafe_allow_html=True)
    times = np.linspace(0, len(audio)/sr, num=len(audio))
    # Downsample for display only (keep every 4th sample for speed)
    ds = 4
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(
        x=times[::ds], y=audio[::ds],
        mode='lines',
        line=dict(color='#00c8ff', width=0.6),
        name='Amplitude'
    ))
    fig_wave.update_layout(
        paper_bgcolor='#0a1628', plot_bgcolor='#060f1e',
        font=dict(color='#4a6480', size=10),
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        xaxis=dict(title='Time (s)', gridcolor='#0d2444', showgrid=True),
        yaxis=dict(title='Amplitude', gridcolor='#0d2444', showgrid=True),
        hovermode='x unified'
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # ── Spectrogram ──
    st.markdown("<div class='section-label'>Spectrogram — frequency content over time</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#060f1e')
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img  = librosa.display.specshow(spec, sr=sr, x_axis="time",
                                    y_axis="hz", ax=ax, cmap="plasma")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors='#4a6480', labelsize=6)
    ax.set_title("SPECTROGRAM", fontsize=8, color='#4a6480', fontfamily='monospace')
    ax.tick_params(colors='#4a6480', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#0d2444')
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Normalize + pad
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    max_len = 3 * sr
    audio = audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)))

    # Features
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.hstack([
        np.mean(mfcc.T,        axis=0),
        np.mean(mfcc_delta.T,  axis=0),
        np.mean(mfcc_delta2.T, axis=0),
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(audio)),
        np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    ])

    return features.reshape(1, -1)


# ================================================
# SHAP CHART
# ================================================
def show_shap(features_scaled):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)
    shap_vals   = shap_values[0]

    top_idx   = np.argsort(np.abs(shap_vals))[-15:]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_vals  = shap_vals[top_idx]
    bar_colors = ["#ff3b5c" if v > 0 else "#00ff9d" for v in top_vals]

    # Plotly interactive bar chart — aligns perfectly with waveform
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_vals,
        y=top_names,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='#0d2444', width=0.5)
        ),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line=dict(color='#0d2444', width=1.5))
    # Shorten y-axis labels to prevent overlap
    short_names = []
    for n in top_names:
        if n.startswith("d2MFCC_"):
            short_names.append("d2_" + n.split("_")[1])
        elif n.startswith("dMFCC_"):
            short_names.append("d_" + n.split("_")[1])
        elif n.startswith("MFCC_"):
            short_names.append("M_" + n.split("_")[1])
        else:
            short_names.append(n[:12])
    fig.data[0].y = short_names

    fig.update_layout(
        paper_bgcolor='#0a1628',
        plot_bgcolor='#060f1e',
        font=dict(color='#4a6480', size=11, family='Share Tech Mono'),
        margin=dict(l=90, r=30, t=30, b=60),
        height=420,
        xaxis=dict(
            title='SHAP Value — positive pushes toward Deepfake, negative toward Real',
            gridcolor='#0d2444', showgrid=True,
            zerolinecolor='#4a6480', zerolinewidth=1.5,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='#0d2444', showgrid=False,
            tickfont=dict(size=11),
            automargin=True
        ),
        bargap=0.3,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='ds-card' style='margin-top:0.8rem; padding:1rem;'>
        <div class='section-label'>How to read this chart</div>
        <div class='mono' style='line-height:1.9; font-size:0.78rem;'>
            <span class='ds-tag-danger'>■ RED bars</span>
            push the prediction toward <strong>Deepfake</strong> &nbsp;|&nbsp;
            <span class='ds-tag-success'>■ GREEN bars</span>
            push toward <strong>Real</strong><br><br>
            <strong style='color:#00c8ff;'>MFCC 1–40</strong>
            — Mel-frequency cepstral coefficients: capture the overall vocal
            timbre and shape of speech. Real voices have richer, more varied
            MFCC patterns. Synthetic voices often appear unnaturally smooth.<br><br>
            <strong style='color:#00c8ff;'>dMFCC 1–40 (Delta)</strong>
            — How fast the vocal quality changes over time. Natural speech has
            high variation (breathing, stress, emotion). Robotic/AI voices show
            very flat deltas — this is one of the strongest deepfake indicators.<br><br>
            <strong style='color:#00c8ff;'>d2MFCC 1–40 (Delta-Delta)</strong>
            — The acceleration of vocal change. Real human speech accelerates
            and decelerates naturally (like handwriting). Synthetic speech is
            mechanically consistent — an unusually low d2MFCC is suspicious.<br><br>
            <strong style='color:#00c8ff;'>SpectralCentroid</strong>
            — The "brightness" of the sound. Synthetic voices often have an
            unnaturally centred spectral brightness without the natural drift
            found in human speech.<br><br>
            <strong style='color:#00c8ff;'>ZCR (Zero Crossing Rate)</strong>
            — How often the audio signal crosses zero. Real speech has
            characteristic ZCR patterns tied to voicing and consonants.
            AI voices can deviate from these patterns.<br><br>
            <strong style='color:#00c8ff;'>SpectralBandwidth</strong>
            — The spread of frequencies present. Real voices are wider and
            more variable. Narrowly focused frequency bands can indicate
            synthetic generation.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ================================================
# PDF REPORT
# ================================================
def safe(text):
    """Replace unicode chars that latin-1 cannot encode."""
    return (text
        .replace("\u2014", "-").replace("\u2013", "-")
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u00a7", "S.").replace("\u20b9", "Rs.")
        .replace("\u00a0", " ")
    )

def generate_pdf(filename, result, confidence):
    is_deepfake = result != "Real Audio (Bonafide)"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "DeepShield - Audio Forensic Investigation Report",
             ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, safe(f"Investigator : Admin"),                                          ln=True)
    pdf.cell(200, 8, safe(f"Audio File   : {filename}"),                                     ln=True)
    pdf.cell(200, 8, safe(f"Detection    : {result}"),                                       ln=True)
    pdf.cell(200, 8, safe(f"Confidence   : {confidence:.2f}%"),                              ln=True)
    pdf.cell(200, 8, safe("Model Used   : XGBoost (Explainable AI)"),                        ln=True)
    pdf.cell(200, 8, safe("Features     : MFCC + Delta + Delta2 + ZCR + Spectral"),          ln=True)
    pdf.cell(200, 8, safe(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)
    pdf.ln(8)

    if is_deepfake:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 8, "Applicable Cyber Laws (India)", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(180, 60, 60)
        pdf.cell(200, 7,
            safe("WARNING: Deepfake audio detected. The following laws may apply."),
            ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        laws = [
            ("IT Act 2000 - S.66D", "Cheating by personation using computer resources - up to 3 years + fine."),
            ("IT Act 2000 - S.66E", "Violation of privacy - up to 3 years imprisonment or Rs. 2 lakh fine."),
            ("IT Act 2000 - S.43",  "Unauthorized access / data tampering - compensation up to Rs. 1 crore."),
            ("IPC S.468",           "Forgery for the purpose of cheating - up to 7 years imprisonment."),
            ("IPC S.471",           "Using forged documents as genuine - up to 2 years imprisonment."),
            ("IPC S.420",           "Cheating and dishonest inducement - up to 7 years imprisonment."),
        ]
        for section, desc in laws:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(200, 7, safe(section), ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 7, safe(desc), ln=True)
            pdf.ln(2)
        pdf.ln(4)
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(200, 7,
            safe("To file a complaint: cybercrime.gov.in  |  Helpline: 1930"),
            ln=True)
        pdf.set_text_color(0, 0, 0)

    else:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 140, 80)
        pdf.cell(200, 8, "Audio Verified - No Deepfake Indicators Found", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 7,
            safe("This audio sample passed forensic analysis as genuine (bonafide) speech."),
            ln=True)
        pdf.cell(200, 7,
            safe("No synthetic voice artifacts were detected by the XGBoost model."),
            ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(200, 7, "Generated by DeepShield Audio Forensics Platform", ln=True)
    pdf.output("forensic_report.pdf")


# ================================================
# TOP NAVIGATION — replaces sidebar
# Always visible, no collapse issues
# ================================================

# Brand header
st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;
            padding:0.6rem 0 0.8rem 0; border-bottom:1px solid #0d2444;
            margin-bottom:1rem;'>
    <div style='font-family:Orbitron,monospace; font-size:1.3rem;
                font-weight:900; color:#00c8ff; letter-spacing:0.1em;'>
        🛡️ DEEPSHIELD
        <span style='font-size:0.6rem; color:#4a6480; margin-left:0.6rem;
                     vertical-align:middle; letter-spacing:0.2em;'>
            AUDIO FORENSICS v2.0
        </span>
    </div>
    <div class='mono' style='font-size:0.72rem; color:#4a6480;'>
        <span style='color:#00ff9d;'>●</span> Model Loaded &nbsp;
        <span style='color:#00ff9d;'>●</span> XGBoost · 123-dim &nbsp;
        <span style='color:#00ff9d;'>●</span> Real-time Ready
    </div>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_laws, tab_about = st.tabs(
    ["⚡  Detection Lab", "⚖️  Cyber Laws", "📋  About"]
)
page = None  # not used — tabs handle routing directly






# ================================================
# TAB 1: DETECTION LAB
# ================================================
with tab_lab:

    # ── Two-column layout: Live Recording (left) | File Upload (right) ──
    lab_col1, lab_col2 = st.columns([1, 1], gap="large")

    # ════════════════════════════════
    # LEFT COLUMN — LIVE RECORDING
    # ════════════════════════════════
    with lab_col1:
        st.markdown("""
        <div class='section-label' style='margin-bottom:0.6rem;'>
            Live Recording
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ds-card ds-card-accent">', unsafe_allow_html=True)

        duration_options = {
            "3 sec  (quick test)": 3,
            "10 seconds":         10,
            "30 seconds":         30,
            "1 minute":           60,
            "2 minutes":         120,
            "5 minutes (max)":   300,
        }
        selected_label = st.selectbox(
            "Duration",
            list(duration_options.keys()),
            index=0,
            help="Max 5 min for live recording. Use File Upload for longer audio."
        )
        chosen_duration = duration_options[selected_label]

        st.markdown(f"""
        <div class='mono' style='font-size:0.72rem; margin:0.4rem 0 0.8rem 0;'>
            Device 1 · Intel MME · auto SR → 16 kHz
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            rec_clicked = st.button("▶ RECORD", use_container_width=True, key="rec_btn")
        with col_b:
            analyse_clicked = st.button("🔍 ANALYSE", use_container_width=True, key="analyse_btn")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Recording logic — countdown timer ──
        if rec_clicked:
            recorded_sr = None
            mins  = chosen_duration // 60
            secs  = chosen_duration % 60
            label = f"{mins}m {secs}s" if mins else f"{secs}s"

            # Find working sample rate first
            for try_sr in [16000, 48000, 44100]:
                try:
                    recording = sd.rec(
                        int(chosen_duration * try_sr),
                        samplerate=try_sr, channels=1, dtype="float32"
                    )
                    recorded_sr = try_sr
                    break
                except Exception:
                    continue

            if recorded_sr is None:
                st.error("Recording failed. Check microphone connection.")
            else:
                # Live countdown display
                countdown_box  = st.empty()
                waveform_box   = st.empty()
                total_samples  = int(chosen_duration * recorded_sr)

                for elapsed in range(chosen_duration):
                    remaining = chosen_duration - elapsed
                    r_min = remaining // 60
                    r_sec = remaining % 60
                    bar_filled = int((elapsed / chosen_duration) * 20)
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    countdown_box.markdown(f"""
                    <div class='ds-card ds-card-accent'>
                        <div class='section-label'>Recording in progress</div>
                        <div style='font-family:Orbitron,monospace; font-size:1.2rem;
                                    color:#ff3b5c; letter-spacing:0.1em;'>
                            ● REC &nbsp; {r_min:02d}:{r_sec:02d} remaining
                        </div>
                        <div style='font-family:Share Tech Mono,monospace;
                                    color:#00c8ff; font-size:0.9rem; margin-top:0.4rem;'>
                            {bar} {elapsed+1}/{chosen_duration}s
                        </div>
                        <div class='mono' style='margin-top:0.3rem; font-size:0.72rem;'>
                            Do not close this tab · SR: {recorded_sr} Hz
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)

                sd.wait()
                sf.write("recorded_audio.wav", recording, recorded_sr)
                countdown_box.empty()
                waveform_box.empty()

                file_mb = round(os.path.getsize("recorded_audio.wav") / 1e6, 1)
                st.success(f"Captured {label} at {recorded_sr} Hz  ({file_mb} MB)")
                if chosen_duration <= 60:
                    st.audio("recorded_audio.wav")
                else:
                    st.info("Recording saved. Click ANALYSE RECORDING to run segmented analysis.")

        # ── Analysis logic (full width, below buttons) ──
        if analyse_clicked:
            if not os.path.exists("recorded_audio.wav"):
                st.warning("Record audio first.")
            else:
                raw, raw_sr = librosa.load("recorded_audio.wav", sr=16000)
                if np.sqrt(np.mean(raw ** 2)) < 0.01:
                    st.error("Signal too weak — speak clearly.")
                else:
                    # Audio type — 4 values (type, conf, note, is_speech)
                    audio_type, type_conf, lang_note, is_speech = classify_audio_type(raw, raw_sr)

                    if not is_speech:
                        speech_warn = f"<span style='color:#ffb020;'>⚠ Model not applicable to {audio_type}</span>"
                    else:
                        speech_warn = "<span style='color:#00ff9d;'>✓ Speech detected — model fully applicable</span>"

                    st.markdown(f"""
                    <div class='ds-card ds-card-accent' style='margin-bottom:0.8rem;'>
                        <div class='section-label'>Audio Profile</div>
                        <div class='mono' style='line-height:1.9;'>
                            Type: <strong style='color:#00c8ff;'>{audio_type}</strong>
                            &nbsp;·&nbsp; Confidence: {type_conf}%<br>
                            {speech_warn}<br>
                            <span style='color:#4a6480; font-size:0.72rem;'>
                                Model trained on ASVspoof 2019 LA — human speech only
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if not is_speech:
                        st.markdown(f"""
                        <div class='ds-card' style='border-left:3px solid #ffb020;'>
                            <div class='section-label'>Analysis Blocked</div>
                            <div class='mono'>
                                Detected: <strong style='color:#ffb020;'>{audio_type}</strong><br>
                                The deepfake model is trained only on human speech (ASVspoof 2019).
                                Applying it to <strong>{audio_type}</strong> would produce
                                meaningless results.<br><br>
                                <span style='color:#4a6480;'>
                                Record a voice speaking clearly to use the deepfake detector.
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.spinner("Extracting features..."):
                            features = extract_features("recorded_audio.wav")

                        if features is None:
                            st.error("Audio rejected by energy gate.")
                        else:
                            features_scaled = scaler.transform(features)
                            prediction      = model.predict(features_scaled)
                            probability     = model.predict_proba(features_scaled)
                            confidence      = np.max(probability) * 100

                            st.session_state.result          = "Real Audio (Bonafide)" if prediction[0] == 0 else "Deepfake Audio"
                            st.session_state.confidence      = confidence
                            st.session_state.filename        = "Recorded Audio"
                            st.session_state.features_scaled = features_scaled
                            st.session_state.audio_type      = audio_type
                            st.session_state.lang_note       = lang_note

# ════════════════════════════════
# RIGHT COLUMN — FILE UPLOAD
# ════════════════════════════════
with lab_col2:
    st.markdown("""
    <div class='section-label' style='margin-bottom:0.6rem;'>
        File Upload
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ds-card ds-card-accent">', unsafe_allow_html=True)

    long_mode = st.checkbox(
        "Long audio mode (segmented analysis)",
        value=False,
        help="Splits audio into 3s segments. Auto-activates for files over 4 seconds."
    )

    uploaded_file = st.file_uploader(
        "wav / mp3 / flac", type=["wav", "mp3", "flac"],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio(uploaded_file)

        raw, raw_sr = librosa.load("temp_audio.wav", sr=16000)
        duration_sec = len(raw) / raw_sr

        # Silence check
        if np.sqrt(np.mean(raw ** 2)) < 0.01:
            st.error("File audio is too quiet.")
        else:
            # ===== AUDIO TYPE CLASSIFICATION =====
            audio_type, type_conf, lang_note, is_speech = classify_audio_type(raw, raw_sr)

            # 🔧 FIX: Prevent crash
            safe_lang = (lang_note or "Unknown")

            if not is_speech:
                speech_warn = f"<span style='color:#ffb020;'>⚠ Model not applicable to {audio_type}</span>"
            else:
                speech_warn = "<span style='color:#00ff9d;'>✓ Speech detected — model fully applicable</span>"

            st.markdown(f"""
            <div class='ds-card ds-card-accent' style='margin-bottom:0.8rem;'>
                <div class='section-label'>Audio Profile</div>
                <div class='mono' style='line-height:1.9;'>
                    Type: <strong style='color:#00c8ff;'>{audio_type}</strong>
                    &nbsp;·&nbsp; Confidence: {type_conf}%<br>
                    Duration: <strong style='color:#00ff9d;'>{duration_sec:.1f}s</strong><br>
                    {speech_warn}<br>
                    <span style='color:#4a6480; font-size:0.72rem;'>
                        Model trained on ASVspoof 2019 LA — human speech only
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 🚨 HARD STOP — MOST IMPORTANT FIX
            if not is_speech:
                st.markdown(f"""
                <div class='ds-card' style='border-left:3px solid #ff3b5c;'>
                    <div class='section-label'>Analysis Blocked</div>
                    <div class='mono'>
                        Detected: <strong>{audio_type}</strong><br>
                        The deepfake model works only on human speech.<br><br>
                        Please upload a speech audio file.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.stop()

            # ===== ONLY SPEECH BELOW =====

            if long_mode or duration_sec > 4:
                st.markdown('<div class="section-label">Running segmented analysis...</div>',
                            unsafe_allow_html=True)

                overall_label, overall_conf, seg_results, fake_pct, total_dur = \
                    analyse_long_audio("temp_audio.wav", scaler, model)

                show_segment_timeline(seg_results, total_dur)

                st.markdown('<div class="section-label">Segment breakdown</div>',
                            unsafe_allow_html=True)

                filter_col1, filter_col2 = st.columns([2, 1])

                with filter_col1:
                    seg_filter = st.radio(
                        "Show segments",
                        ["All", "Deepfake only", "Real only"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )

                with filter_col2:
                    st.markdown(
                        f"<div class='mono' style='text-align:right;'>"
                        f"Total: {len(seg_results)} segments</div>",
                        unsafe_allow_html=True
                    )

                # Apply filter
                if seg_filter == "Deepfake only":
                    filtered = [s for s in seg_results if s[2] == "Deepfake"]
                elif seg_filter == "Real only":
                    filtered = [s for s in seg_results if s[2] == "Real"]
                else:
                    filtered = seg_results

                if not filtered:
                    st.markdown(
                        "<div class='mono' style='color:#4a6480;'>No segments match this filter.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    hdr = st.columns([1, 1, 2, 1])
                    for h, t in zip(hdr, ["Start", "End", "Verdict", "Confidence"]):
                        h.markdown(f"<div class='mono'><strong>{t}</strong></div>",
                                   unsafe_allow_html=True)

                    parts = ["<div style='max-height:300px;overflow-y:auto;border:1px solid #0d2444;border-radius:4px;'>"]

                    for (s, e, lbl, conf) in filtered:
                        color = "#00ff9d" if lbl == "Real" else "#ff3b5c"
                        parts.append(
                            f"<div style='display:grid;grid-template-columns:1fr 1fr 2fr 1fr;"
                            f"padding:5px 10px;border-bottom:1px solid #0d2444;'>"
                            f"<span class='mono'>{s}s</span>"
                            f"<span class='mono'>{e}s</span>"
                            f"<span class='mono' style='color:{color};font-weight:600;'>{lbl}</span>"
                            f"<span class='mono'>{conf}%</span></div>"
                        )

                    parts.append("</div>")
                    st.markdown("".join(parts), unsafe_allow_html=True)

                # Store results
                st.session_state.result = overall_label
                st.session_state.confidence = overall_conf
                st.session_state.filename = uploaded_file.name
                st.session_state.audio_type = audio_type
                st.session_state.lang_note = safe_lang
                st.session_state.fake_pct = fake_pct

            else:
                with st.spinner("Running AI forensic analysis..."):
                    features = extract_features("temp_audio.wav")

                if features is None:
                    st.error("Audio rejected — near-silent file.")
                else:
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)
                    probability = model.predict_proba(features_scaled)

                    confidence = np.max(probability) * 100

                    st.session_state.result = "Real Audio (Bonafide)" if prediction[0] == 0 else "Deepfake Audio"
                    st.session_state.confidence = confidence
                    st.session_state.filename = uploaded_file.name
                    st.session_state.audio_type = audio_type
                    st.session_state.lang_note = safe_lang

    st.markdown('</div>', unsafe_allow_html=True)

    # ── RESULT PANEL ──
    if st.session_state.result is not None:
        st.markdown("<br>", unsafe_allow_html=True)

        is_real      = st.session_state.result == "Real Audio (Bonafide)"
        result_class = "result-real" if is_real else "result-fake"
        card_class   = "ds-card-success" if is_real else "ds-card-danger"
        verdict_icon = "✅" if is_real else "⚠️"
        audio_type   = st.session_state.get("audio_type", "Unknown")
        lang_note    = st.session_state.get("lang_note", "")
        fake_pct     = st.session_state.get("fake_pct", None)

        # ── Top row: Verdict (left) | Audio Profile (right) ──
        r_col1, r_col2 = st.columns([3, 2], gap="medium")

        # Reliability flag — shown if audio type is uncertain or non-speech
        audio_type_stored = st.session_state.get("audio_type", "Unknown")
        is_reliable = audio_type_stored in SPEECH_TYPES or audio_type_stored == "Unknown Audio Type"
        reliability_note = (
            "" if is_reliable else
            f"<div class='mono' style='color:#ffb020; font-size:0.72rem; margin-top:0.5rem;'>"
            f"⚠ Audio type: {audio_type_stored} — model trained on speech only. "
            f"This result may not be reliable.</div>"
        )

        with r_col1:
            st.markdown(f"""
            <div class="ds-card {card_class}" style="height:100%; min-height:130px;">
                <div class="section-label">Forensic Verdict</div>
                <div class="{result_class}" style="margin-bottom:0.5rem;">
                    {verdict_icon}&nbsp; {st.session_state.result.upper()}
                </div>
                <div class='mono' style='color:#4a6480; font-size:0.7rem;'>
                    XGBoost · Speech deepfake detection
                </div>
                {f"<div class='mono' style='color:#ff3b5c; margin-top:0.3rem;'>Deepfake in {fake_pct}% of segments</div>" if fake_pct is not None else ""}
                {reliability_note}
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div class="ds-card ds-card-accent" style="height:100%; min-height:130px;">
                <div class="section-label">Audio Profile</div>
                <div class='mono' style='line-height:2.1;'>
                    <span style='color:#4a6480;'>Type</span>&nbsp;&nbsp;&nbsp;
                    <strong style='color:#00c8ff;'>{audio_type}</strong><br>
                    <span style='color:#4a6480;'>Language</span>&nbsp;
                   <span style='color:#00ff9d;'>{(lang_note or "Unknown").split("—")[0].strip()}</span><br>
                    <span style='color:#4a6480;'>Features</span>&nbsp;
                    <span style='color:#c8dff5;'>123-dimensional</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Metrics row ──
        st.markdown("<div style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="small")
        m1.metric("Confidence",  f"{st.session_state.confidence:.1f}%")
        m2.metric("Audio Type",  audio_type.split("/")[0].strip())
        m3.metric("Model",       "XGBoost")
        m4.metric("Features",    "123-dim")
        st.markdown("</div>", unsafe_allow_html=True)
        st.progress(int(st.session_state.confidence))

        # ── Legal warning if deepfake ──
        if not is_real:
            st.markdown("""
            <div class="ds-card ds-card-warn" style="margin-top:0.6rem;">
                <div class="section-label">Legal Notice</div>
                <div class='mono'>
                    Deepfake audio detected. Click the
                    <strong style='color:#ffb020;'>⚖️ Cyber Laws</strong>
                    tab above for applicable legal sections before filing a complaint.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP — full width, inline (no expander to avoid overlap) ──
        if st.session_state.features_scaled is not None:
            st.markdown("""
            <div style='border-top:1px solid #0d2444; margin:1.2rem 0 0.8rem 0;
                        padding-top:0.8rem;'>
                <span style='font-family:Share Tech Mono,monospace; font-size:0.7rem;
                             color:#4a6480; letter-spacing:0.2em; text-transform:uppercase;'>
                    Explainability Analysis (SHAP)
                </span>
            </div>
            """, unsafe_allow_html=True)
            show_shap(st.session_state.features_scaled)

        # ── PDF button ──
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_col, _ = st.columns([2, 1])
        with pdf_col:
            if st.button("📄 GENERATE FORENSIC REPORT", use_container_width=True):
                generate_pdf(
                    st.session_state.filename,
                    st.session_state.result,
                    st.session_state.confidence
                )
                with open("forensic_report.pdf", "rb") as f:
                    st.download_button(
                        label="⬇ DOWNLOAD REPORT (PDF)",
                        data=f.read(),
                        file_name="DeepShield_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


# ================================================
# TAB 2: CYBER LAWS
# ================================================
with tab_laws:

    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.2rem;
                    font-weight:900; color:#00c8ff; letter-spacing:0.08em;'>
            ⚖️ CYBER LAWS
        </div>
        <div class='mono'>Applicable legal framework for deepfake audio offences in India</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ds-card ds-card-warn" style="margin-bottom:1.5rem;">
        <div class='mono'>
        If you have detected deepfake audio, you may have grounds to file a complaint.
        The laws below are the most relevant sections under Indian law.
        Always consult a legal professional before proceeding.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── IT Act 2000 ──
    st.markdown('<div class="section-label">Information Technology Act, 2000</div>', unsafe_allow_html=True)

    laws_it = [
        {
            "section": "Section 43 — Unauthorized Access & Data Tampering",
            "desc": "Whoever accesses or manipulates a computer system without permission is liable to pay compensation to the affected party.",
            "penalty": "Compensation up to ₹1 crore to the victim.",
            "relevance": "Applies when deepfake audio is created by illegally accessing someone's voice recordings."
        },
        {
            "section": "Section 66 — Computer Related Offences",
            "desc": "Dishonestly or fraudulently committing any act referred to in Section 43 is a criminal offence.",
            "penalty": "Up to 3 years imprisonment and/or fine up to ₹5 lakh.",
            "relevance": "Applies to fraudulent creation or distribution of deepfake audio."
        },
        {
            "section": "Section 66C — Identity Theft",
            "desc": "Fraudulently or dishonestly making use of another person's electronic signature, password, or any other unique identification feature.",
            "penalty": "Up to 3 years imprisonment and fine up to ₹1 lakh.",
            "relevance": "Applies when a deepfake voice is used to impersonate someone's identity."
        },
        {
            "section": "Section 66D — Cheating by Personation",
            "desc": "Cheating someone by pretending to be another person using a computer resource or communication device.",
            "penalty": "Up to 3 years imprisonment and fine up to ₹1 lakh.",
            "relevance": "Directly applicable — deepfake voice used to impersonate and deceive."
        },
        {
            "section": "Section 66E — Violation of Privacy",
            "desc": "Knowingly or intentionally capturing, publishing, or transmitting the image or voice of a private area of any person without consent.",
            "penalty": "Up to 3 years imprisonment or fine up to ₹2 lakh, or both.",
            "relevance": "Applies when deepfake audio is created using private voice recordings."
        },
        {
            "section": "Section 67 — Publishing Obscene Material",
            "desc": "Publishing or transmitting obscene material in electronic form.",
            "penalty": "First conviction: up to 3 years + ₹5 lakh fine. Repeat: up to 5 years + ₹10 lakh fine.",
            "relevance": "Applies if deepfake audio is of an obscene or sexual nature."
        },
    ]

    # 3-column grid for IT Act laws
    it_rows = [laws_it[i:i+3] for i in range(0, len(laws_it), 3)]
    for row in it_rows:
        cols = st.columns(len(row), gap="medium")
        for col, law in zip(cols, row):
            col.markdown(f"""
            <div style='background:#0a1628; border:1px solid #0d2444;
                        border-top:3px solid #ffb020; border-radius:6px;
                        padding:1rem 1.1rem; height:100%;'>
                <div style='font-family:Orbitron,monospace; font-size:0.68rem;
                            color:#ffb020; letter-spacing:0.08em; margin-bottom:0.5rem;'>
                    {law['section'].split(' — ')[0]}
                </div>
                <div style='font-size:0.78rem; font-weight:600; color:#c8dff5;
                            margin-bottom:0.5rem; line-height:1.4;'>
                    {law['section'].split(' — ')[1] if ' — ' in law['section'] else ''}
                </div>
                <div style='font-size:0.78rem; color:#8aa4be; line-height:1.5;
                            margin-bottom:0.6rem;'>
                    {law['desc']}
                </div>
                <div style='font-family:Share Tech Mono,monospace; font-size:0.72rem;
                            color:#ff3b5c; margin-bottom:0.4rem;'>
                    ⚖ {law['penalty']}
                </div>
                <div style='font-size:0.72rem; color:#4a6480; line-height:1.4;'>
                    {law['relevance']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── IPC ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Indian Penal Code (IPC)</div>', unsafe_allow_html=True)

    laws_ipc = [
        {
            "section": "IPC Section 419 — Punishment for Cheating by Personation",
            "desc": "Whoever cheats by pretending to be some other person, or by knowingly substituting one person for another.",
            "penalty": "Up to 3 years imprisonment, or fine, or both.",
            "relevance": "Applies when deepfake voice is used to deceive victims directly."
        },
        {
            "section": "IPC Section 420 — Cheating & Dishonest Inducement",
            "desc": "Cheating and thereby dishonestly inducing delivery of property or causing alteration of valuable security.",
            "penalty": "Up to 7 years imprisonment and fine.",
            "relevance": "Applies in financial fraud cases using deepfake audio (e.g. fake CEO voice calls)."
        },
        {
            "section": "IPC Section 468 — Forgery for Purpose of Cheating",
            "desc": "Committing forgery with the intention that the forged document or record be used for cheating.",
            "penalty": "Up to 7 years imprisonment and fine.",
            "relevance": "Deepfake audio used as fabricated evidence or to forge communications."
        },
        {
            "section": "IPC Section 469 — Forgery for Reputation Damage",
            "desc": "Committing forgery intending to harm the reputation of any party, or knowing it is likely to be used for that purpose.",
            "penalty": "Up to 3 years imprisonment and fine.",
            "relevance": "Deepfake audio created to defame or harm someone's social/professional reputation."
        },
        {
            "section": "IPC Section 471 — Using Forged Documents as Genuine",
            "desc": "Fraudulently or dishonestly using as genuine any document or electronic record which is known to be forged.",
            "penalty": "Same as for forgery of that document — up to life imprisonment in severe cases.",
            "relevance": "Using deepfake audio as genuine evidence in disputes, courts, or negotiations."
        },
        {
            "section": "IPC Section 500 — Defamation",
            "desc": "Making or publishing any imputation concerning a person with intent to harm reputation.",
            "penalty": "Up to 2 years imprisonment, or fine, or both.",
            "relevance": "Deepfake audio spread publicly to damage someone's character."
        },
    ]

    # 3-column grid for IPC laws
    ipc_rows = [laws_ipc[i:i+3] for i in range(0, len(laws_ipc), 3)]
    for row in ipc_rows:
        cols = st.columns(len(row), gap="medium")
        for col, law in zip(cols, row):
            col.markdown(f"""
            <div style='background:#0a1628; border:1px solid #0d2444;
                        border-top:3px solid #00c8ff; border-radius:6px;
                        padding:1rem 1.1rem; height:100%;'>
                <div style='font-family:Orbitron,monospace; font-size:0.68rem;
                            color:#00c8ff; letter-spacing:0.08em; margin-bottom:0.5rem;'>
                    {law['section'].split(' — ')[0]}
                </div>
                <div style='font-size:0.78rem; font-weight:600; color:#c8dff5;
                            margin-bottom:0.5rem; line-height:1.4;'>
                    {law['section'].split(' — ')[1] if ' — ' in law['section'] else ''}
                </div>
                <div style='font-size:0.78rem; color:#8aa4be; line-height:1.5;
                            margin-bottom:0.6rem;'>
                    {law['desc']}
                </div>
                <div style='font-family:Share Tech Mono,monospace; font-size:0.72rem;
                            color:#ff3b5c; margin-bottom:0.4rem;'>
                    ⚖ {law['penalty']}
                </div>
                <div style='font-size:0.72rem; color:#4a6480; line-height:1.4;'>
                    {law['relevance']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── How to File ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How to File a Complaint</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Generate Report", "Use the Detection Lab to generate your forensic PDF report as evidence."),
        ("02", "Cybercrime Portal", "File at cybercrime.gov.in — select 'Report Cyber Crime' → 'Other Cyber Crime'."),
        ("03", "Local Police", "Visit the nearest cyber cell or police station with your forensic report."),
        ("04", "Legal Counsel", "Consult a cyber law attorney — they can file under the most applicable sections."),
        ("05", "Preserve Evidence", "Keep the original audio file, forensic report, and any messages related to the incident."),
    ]

    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="ds-card" style="text-align:center; height:100%;">
            <div style='font-family:Orbitron,monospace; font-size:1.4rem;
                        color:#00c8ff; font-weight:900;'>{num}</div>
            <div style='font-family:Orbitron,monospace; font-size:0.65rem;
                        color:#ffb020; margin:0.4rem 0;'>{title}</div>
            <div class='mono' style='font-size:0.72rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div class='ds-card ds-card-accent'>
        <div class='mono' style='font-size:0.8rem;'>
        📞 <strong>National Cybercrime Helpline:</strong> 1930<br>
        🌐 <strong>Portal:</strong> <a href="https://cybercrime.gov.in" style="color:#00c8ff;">cybercrime.gov.in</a><br>
        📧 <strong>Email:</strong> cybercrime@gov.in
        </div>
    </div>
    """, unsafe_allow_html=True)


# ================================================
# TAB 3: ABOUT
# ================================================
with tab_about:

    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.2rem;
                    font-weight:900; color:#00c8ff; letter-spacing:0.08em;'>
            📋 ABOUT
        </div>
        <div class='mono'>Explainable Deepfake Audio Detection Framework for Digital Forensics</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="ds-card ds-card-accent">
            <div class="section-label">System Overview</div>
            <div class='mono' style='line-height:1.8;'>
            DeepShield uses machine learning to detect synthetic (deepfake) audio.
            It extracts 123 acoustic features from each audio sample and classifies
            them using a trained XGBoost model. SHAP values explain why each
            decision was made — making it suitable for forensic use.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ds-card ds-card-accent">
            <div class="section-label">Feature Pipeline</div>
            <div class='mono' style='line-height:2;'>
            <span class='ds-tag'>MFCC ×40</span>
            <span class='ds-tag'>Δ-MFCC ×40</span>
            <span class='ds-tag'>Δ²-MFCC ×40</span><br>
            <span class='ds-tag'>Spectral Centroid</span>
            <span class='ds-tag'>ZCR</span>
            <span class='ds-tag'>Spectral Bandwidth</span><br><br>
            Total: 123-dimensional feature vector<br>
            Preprocessing: Noise reduction → Normalize → 3s fixed length
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="ds-card ds-card-accent">
            <div class="section-label">Training Dataset</div>
            <div class='mono' style='line-height:1.8;'>
            Dataset: ASVspoof 2019 — Logical Access (LA)<br>
            Bonafide samples : 2,000<br>
            Spoof samples    : 2,000<br>
            Own-voice samples: added for mic generalisation<br>
            Split            : 80% train / 20% test
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ds-card ds-card-accent">
            <div class="section-label">Model Architecture</div>
            <div class='mono' style='line-height:1.8;'>
            Primary : XGBoost (n=200, depth=6, lr=0.05)<br>
            Trained : Random Forest + Gradient Boosting<br>
            Scaler  : StandardScaler (fitted on training set)<br>
            XAI     : SHAP TreeExplainer<br>
            Report  : Auto-generated PDF with legal references
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model metadata card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Model Metadata & Versioning</div>",
                unsafe_allow_html=True)

    try:
        import os as _os
        model_path  = "deepfake_audio_xgboost_model.pkl"
        scaler_path = "scaler.pkl"
        model_mtime  = datetime.fromtimestamp(_os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M") if _os.path.exists(model_path) else "N/A"
        scaler_mtime = datetime.fromtimestamp(_os.path.getmtime(scaler_path)).strftime("%Y-%m-%d %H:%M") if _os.path.exists(scaler_path) else "N/A"
        model_size   = f"{_os.path.getsize(model_path)/1e6:.1f} MB" if _os.path.exists(model_path) else "N/A"
    except Exception:
        model_mtime = scaler_mtime = model_size = "N/A"

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.markdown(f"""
    <div class='ds-card ds-card-accent'>
        <div class='section-label'>Model File</div>
        <div class='mono' style='line-height:1.8;'>
        Name: XGBoost deepfake detector<br>
        Last trained: {model_mtime}<br>
        File size: {model_size}<br>
        Version: v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    meta_col2.markdown(f"""
    <div class='ds-card ds-card-accent'>
        <div class='section-label'>Scaler</div>
        <div class='mono' style='line-height:1.8;'>
        Type: StandardScaler<br>
        Last fitted: {scaler_mtime}<br>
        Dimensions: 123<br>
        Fit on: ASVspoof2019 + own-voice
        </div>
    </div>
    """, unsafe_allow_html=True)

    meta_col3.markdown("""
    <div class='ds-card ds-card-accent'>
        <div class='section-label'>Performance Targets</div>
        <div class='mono' style='line-height:1.8;'>
        Accuracy: check terminal after training<br>
        EER target: &lt; 10%<br>
        AUC target: &gt; 0.90<br>
        Run extract_features.py to see actuals
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='ds-card' style='margin-top:1rem; text-align:center;'>
        <div class='mono' style='font-size:0.75rem; color:#4a6480;'>
        DeepShield · Explainable Audio Forensics · Built with Streamlit + XGBoost + SHAP<br>
        For academic and forensic research use only · Always consult a legal professional<br>
        Cyber Laws information is for reference only — not legal advice
        </div>
    </div>
    """, unsafe_allow_html=True)
