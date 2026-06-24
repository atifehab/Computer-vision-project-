import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import zipfile

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Sets the browser tab title, icon, and layout mode.
# "wide" layout gives more horizontal space for the UI.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COTS Detector — Starfish Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — PREMIUM DARK THEME
# Injects CSS to override Streamlit's default styling and create a modern,
# visually stunning dark interface with:
#   • Deep gradient background (navy → dark purple)
#   • Glassmorphism cards (semi-transparent with backdrop blur)
#   • Custom Google Font (Inter) for clean typography
#   • Smooth hover animations on interactive elements
#   • Gradient text for the main title
#   • Custom styled file uploader with dashed border
#   • Animated progress bars with color-coded gradients
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1040 40%, #0d1b2a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Main Title ── */
    .main-title {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }
    .main-title h1 {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff6bcc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .main-title p {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(123, 47, 247, 0.3);
        box-shadow: 0 8px 32px rgba(123, 47, 247, 0.15);
    }

    /* ── Result Card ── */
    .result-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-card.cots-found {
        border-color: rgba(255, 82, 82, 0.4);
        box-shadow: 0 0 40px rgba(255, 82, 82, 0.1);
    }
    .result-card.no-cots {
        border-color: rgba(0, 210, 255, 0.4);
        box-shadow: 0 0 40px rgba(0, 210, 255, 0.1);
    }

    /* ── Prediction Label ── */
    .prediction-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .prediction-label.danger {
        color: #ff5252;
    }
    .prediction-label.safe {
        color: #00d2ff;
    }

    /* ── Confidence Display ── */
    .confidence-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        letter-spacing: -2px;
    }
    .confidence-value.danger {
        background: linear-gradient(135deg, #ff5252, #ff793f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .confidence-value.safe {
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Confidence Bar ── */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        height: 14px;
        margin: 1.5rem auto;
        max-width: 400px;
        overflow: hidden;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease;
    }
    .confidence-bar.danger {
        background: linear-gradient(90deg, #ff5252, #ff793f);
    }
    .confidence-bar.safe {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    }

    /* ── Probability Breakdown ── */
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .prob-label {
        color: #ccd6f6;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .prob-value {
        font-weight: 700;
        font-size: 1.1rem;
    }
    .prob-value.highlight {
        color: #7b2ff7;
    }
    .prob-value.muted {
        color: #8892b0;
    }

    /* ── Section Headers ── */
    .section-header {
        color: #ccd6f6;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Info Badge ── */
    .info-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .info-badge.model {
        background: rgba(123, 47, 247, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(123, 47, 247, 0.3);
    }

    /* ── Upload Area ── */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(123, 47, 247, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    .stFileUploader > div > div:hover {
        border-color: rgba(123, 47, 247, 0.6) !important;
        background: rgba(123, 47, 247, 0.05) !important;
    }

    /* ── Tabs Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(123, 47, 247, 0.15) !important;
        color: #a78bfa !important;
    }

    /* ── Feature Cards ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .feature-item:hover {
        transform: translateY(-2px);
        border-color: rgba(123, 47, 247, 0.3);
    }
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .feature-name {
        color: #ccd6f6;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .feature-desc {
        color: #8892b0;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(123, 47, 247, 0.3), transparent);
        margin: 2rem 0;
    }

    /* ── Button Override ── */
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7 0%, #00d2ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(123, 47, 247, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource ensures the model is loaded only ONCE and cached in memory
# across all user sessions. This prevents reloading the ~20MB model on every
# page refresh or interaction.
#
# Steps:
# 1. Create EfficientNet-B0 backbone (without pretrained ImageNet weights,
#    since we're loading our own fine-tuned weights)
# 2. Replace the default 1000-class classifier with a 2-class linear layer
#    that matches our saved state_dict structure:
#      classifier = Sequential(Dropout, Linear(1280, 2))
# 3. Re-archive the extracted model directory into a .zip file that
#    torch.load() can read (PyTorch's save format is actually a zip)
# 4. Load the state_dict into the model
# 5. Set model.eval() to freeze BatchNorm and Dropout layers for inference
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the fine-tuned EfficientNet-B0 model for COTS classification."""

    # Build the EfficientNet-B0 architecture (no pretrained weights needed)
    model = models.efficientnet_b0(weights=None)

    # Replace classifier head to match our 2-class fine-tuned model
    # Original: classifier = Sequential(Dropout(0.2), Linear(1280, 1000))
    # Ours:     classifier = Sequential(Dropout(0.2), Linear(1280, 2))
    num_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(num_features, 2)

    # Determine paths
    zip_path = os.path.join(os.path.dirname(__file__), "cots_model_archive.zip")
    new_model_path = os.path.join(os.path.dirname(__file__), "models", "cots_model.pth")

    # Load saved state_dict (OrderedDict of parameter tensors)
    if os.path.exists(new_model_path):
        state_dict = torch.load(new_model_path, map_location="cpu", weights_only=False)
    else:
        # Re-create zip archive from the extracted model directory if needed
        if not os.path.exists(zip_path):
            shutil.make_archive(zip_path.replace(".zip", ""), "zip", ".", "cots_model")
        state_dict = torch.load(zip_path, map_location="cpu", weights_only=False)
    
    # Handle state_dict key matching for modified classifier structures
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try loading with the new Sequential head architecture (used in v2 training)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 2),
        )
        model.load_state_dict(state_dict)

    # Set to evaluation mode: disables dropout, uses running mean/var for BN
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# This transform chain prepares any input image for the EfficientNet-B0 model:
#
# 1. Resize(256): Scale the shorter edge to 256 pixels (aspect-preserving)
# 2. CenterCrop(224): Crop a 224×224 region from the center
#    → EfficientNet-B0 expects exactly 224×224 input
# 3. ToTensor(): Convert PIL Image → float32 tensor with values in [0, 1]
#    and reorder dimensions from (H, W, C) → (C, H, W)
# 4. Normalize(): Apply ImageNet normalization statistics
#    → Each channel: (pixel - mean) / std
#    → This matches the preprocessing used during training
# ─────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet channel means
        std=[0.229, 0.224, 0.225],     # ImageNet channel stds
    ),
])

# Class labels for the binary classifier output
# Class 0 = COTS Found, Class 1 = No COTS
CLASS_NAMES = ["COTS Found!", "No COTS Detected"]


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# Takes a PIL Image and returns the predicted class index, confidence,
# and full probability distribution.
#
# Pipeline:
# 1. Convert to RGB (handles RGBA/grayscale inputs)
# 2. Apply preprocessing transforms
# 3. Add batch dimension: (C, H, W) → (1, C, H, W)
# 4. Forward pass through the model (no gradient computation needed)
# 5. Apply softmax to convert logits → probabilities
# 6. Extract the predicted class (argmax) and its confidence
# ─────────────────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model):
    """
    Run inference on a single image.

    Args:
        image: PIL Image to classify
        model: Loaded EfficientNet-B0 model

    Returns:
        predicted_class (int): 0 = COTS Found, 1 = No COTS
        confidence (float): Probability of the predicted class (0.0 to 1.0)
        probabilities (list[float]): [prob_cots, prob_no_cots]
    """
    # Ensure 3-channel RGB input
    image = image.convert("RGB")

    # Preprocess: resize → crop → tensor → normalize
    input_tensor = transform(image)

    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    input_batch = input_tensor.unsqueeze(0)

    # Inference (no gradient tracking for efficiency)
    with torch.no_grad():
        outputs = model(input_batch)                     # Raw logits: (1, 2)
        probabilities = torch.softmax(outputs, dim=1)    # Probabilities: (1, 2)
        confidence, predicted = torch.max(probabilities, 1)  # Best class

    return (
        predicted.item(),
        confidence.item(),
        probabilities[0].tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── Title Section ──
st.markdown("""
<div class="main-title">
    <h1>COTS Detector</h1>
    <p>AI-Powered Crown-of-Thorns Starfish Classification</p>
</div>
""", unsafe_allow_html=True)

# ── Feature Highlights ──
st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <div class="feature-icon" style="font-size:1.4rem; color:#a78bfa; font-weight:800;">AI</div>
        <div class="feature-name">EfficientNet-B0</div>
        <div class="feature-desc">State-of-the-art backbone</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon" style="font-size:1.4rem; color:#00d2ff; font-weight:800;">2-Class</div>
        <div class="feature-name">Binary Classifier</div>
        <div class="feature-desc">COTS vs No-COTS</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon" style="font-size:1.4rem; color:#ff6bcc; font-weight:800;">Fast</div>
        <div class="feature-name">Real-Time</div>
        <div class="feature-desc">Instant predictions</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Load Model ──
with st.spinner("Loading model..."):
    model = load_model()

st.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <span class="info-badge model">Model Loaded — EfficientNet-B0 (2 classes)</span>
</div>
""", unsafe_allow_html=True)

# ── Input Section ──
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">Input Image</div>', unsafe_allow_html=True)

    # Tabs for different input methods
    tab_upload, tab_camera = st.tabs(["Upload Image", "Camera"])

    input_image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

    with tab_camera:
        camera_input = st.camera_input("Take a photo")
        if camera_input is not None:
            input_image = Image.open(camera_input)

    # Display uploaded image
    if input_image is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(input_image, caption="Input Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Classify button
        classify_clicked = st.button("Classify Image", use_container_width=True)
    else:
        classify_clicked = False
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 1.6rem; margin-bottom: 1rem; color: #8892b0; font-weight: 700;">[ IMAGE ]</div>
            <p style="color: #8892b0; font-size: 1.1rem;">
                Upload an underwater image or take a photo to detect COTS
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Results Section ──
with col_right:
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

    if input_image is not None and classify_clicked:
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, probs = predict(input_image, model)

        # Determine styling based on prediction
        # Class 0 = COTS detected, Class 1 = No COTS (safe)
        is_cots = predicted_class == 0
        card_class = "cots-found" if is_cots else "no-cots"
        style_class = "danger" if is_cots else "safe"
        icon = "[!]" if is_cots else "[OK]"
        label = "COTS Detected!" if is_cots else "No COTS — Safe"

        # Result Card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div style="font-size: 2rem; margin-bottom: 0.5rem; font-weight: 900; color: {'#ff5252' if is_cots else '#00d2ff'};">{icon}</div>
            <div class="prediction-label {style_class}">{label}</div>
            <div class="confidence-value {style_class}">{confidence * 100:.1f}%</div>
            <p style="color: #8892b0; font-size: 0.9rem; margin-bottom: 0.5rem;">Confidence Score</p>
            <div class="confidence-bar-container">
                <div class="confidence-bar {style_class}" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Breakdown
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)

        # Probability highlights (Class 0 = COTS, Class 1 = No COTS)
        cots_highlight = "highlight" if predicted_class == 0 else "muted"
        no_cots_highlight = "highlight" if predicted_class == 1 else "muted"

        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label">COTS Found</span>
            <span class="prob-value {cots_highlight}">{probs[0] * 100:.2f}%</span>
        </div>
        <div class="prob-row">
            <span class="prob-label">No COTS</span>
            <span class="prob-value {no_cots_highlight}">{probs[1] * 100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Additional info
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1.2rem;">
            <div style="color: #8892b0; font-size: 0.85rem;">
                <strong style="color: #ccd6f6;">About this detection:</strong><br>
                The Crown-of-Thorns Starfish (<em>Acanthaster planci</em>) is one of the
                largest and most destructive predators of coral reefs. Early detection
                is critical for reef conservation efforts. This model was trained on
                underwater imagery from the Great Barrier Reef.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="result-card" style="padding: 4rem 2rem;">
            <div style="font-size: 1.4rem; margin-bottom: 1rem; color: #8892b0; font-weight: 700;">[ ? ]</div>
            <p style="color: #8892b0; font-size: 1.1rem; margin-bottom: 0.5rem;">
                Waiting for Analysis
            </p>
            <p style="color: #64748b; font-size: 0.9rem;">
                Upload an image and click "Classify" to see results
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <p style="color: #64748b; font-size: 0.8rem;">
        COTS Detector | Built with Streamlit and PyTorch | EfficientNet-B0
    </p>
</div>
""", unsafe_allow_html=True)
