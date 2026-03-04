import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import json
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Classifier",
    page_icon="🏅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ────────────────────────────────────────────────────
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MODEL_PATH    = "sports_efficientnet_b0_pytorch.pth"
CLASS_JSON    = "class_indices.json"
NUM_CLASSES   = 100
DROPOUT       = 0.3

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f172a; }
    .block-container { padding-top: 2rem; max-width: 780px; }
    .title-block { text-align: center; margin-bottom: 1.5rem; }
    .title-block h1 { color: #f97316; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .title-block p  { color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem; }
    .pred-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .pred-top {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f97316;
        margin-bottom: 0.2rem;
    }
    .pred-conf { color: #94a3b8; font-size: 0.85rem; }
    .bar-row { display: flex; align-items: center; margin: 0.35rem 0; gap: 10px; }
    .bar-label { color: #cbd5e1; font-size: 0.82rem; width: 160px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-bg { flex: 1; background: #334155; border-radius: 4px; height: 10px; }
    .bar-fill { height: 10px; border-radius: 4px; background: linear-gradient(90deg, #f97316, #ea580c); }
    .bar-pct { color: #94a3b8; font-size: 0.78rem; width: 42px; }
    .upload-hint { color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 0.5rem; }
    div[data-testid="stFileUploader"] label { color: #cbd5e1 !important; }
    div[data-testid="stFileUploader"] section {
        background: #1e293b !important;
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #f97316 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load EfficientNet-B0 with saved weights and class mapping."""
    # Build model architecture
    weights    = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_model = models.efficientnet_b0(weights=None)
    in_features = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT / 2),
        nn.Linear(512, NUM_CLASSES)
    )

    # Load checkpoint
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Handle both raw state_dict and full checkpoint dict
    if "model_state_dict" in checkpoint:
        base_model.load_state_dict(checkpoint["model_state_dict"])
        idx_to_class = checkpoint.get("idx_to_class", None)
    else:
        base_model.load_state_dict(checkpoint)
        idx_to_class = None

    # Fallback: load class indices from JSON
    if idx_to_class is None:
        with open(CLASS_JSON, "r") as f:
            idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    base_model.to(device).eval()
    return base_model, idx_to_class, device


# ── Inference transform ──────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(img: Image.Image, model, idx_to_class, device, top_k=5):
    """Return top-k (label, confidence) predictions for a PIL image."""
    tensor = infer_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs              = torch.softmax(model(tensor), dim=1)[0]
        top_probs, top_idx = probs.topk(top_k)
    return [
        (idx_to_class[i.item()].replace("_", " ").title(), p.item())
        for i, p in zip(top_idx, top_probs)
    ]


def confidence_bars(predictions):
    """Render top-5 confidence bars as HTML."""
    html = ""
    for label, conf in predictions:
        pct  = conf * 100
        fill = f"width:{pct:.1f}%"
        html += f"""
        <div class="bar-row">
            <div class="bar-label" title="{label}">{label}</div>
            <div class="bar-bg"><div class="bar-fill" style="{fill}"></div></div>
            <div class="bar-pct">{pct:.1f}%</div>
        </div>"""
    return html


# ── UI ───────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏅 Sports Classifier</h1>
    <p>EfficientNet-B0 · 100 sports · Transfer Learning · PyTorch</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    try:
        model, idx_to_class, device = load_model()
        st.success(f"Model ready — running on **{device}**", icon="✅")
    except FileNotFoundError:
        st.error(
            f"Model file `{MODEL_PATH}` not found. "
            "Make sure it is in the same directory as app.py.",
            icon="🚨"
        )
        st.stop()

st.divider()

# Upload
uploaded_file = st.file_uploader(
    "Upload a sports image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG, WEBP",
)
st.markdown('<p class="upload-hint">Drop any sports photo — the model will predict which of 100 sports it belongs to</p>', unsafe_allow_html=True)

if uploaded_file:
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(img, caption=uploaded_file.name, use_column_width=True)

    with col2:
        with st.spinner("Classifying..."):
            predictions = predict(img, model, idx_to_class, device, top_k=5)

        top_label, top_conf = predictions[0]

        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-top">{top_label}</div>
            <div class="pred-conf">Top prediction · {top_conf*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Top 5 predictions**")
        st.markdown(
            f'<div style="margin-top:0.5rem">{confidence_bars(predictions)}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Full probability table (expander)
    with st.expander("Show all top-10 predictions"):
        full_preds = predict(img, model, idx_to_class, device, top_k=10)
        for rank, (label, conf) in enumerate(full_preds, 1):
            st.markdown(f"**{rank}.** {label} — `{conf*100:.2f}%`")

# ── Footer ───────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e293b; margin-top:2rem"/>
<p style="text-align:center; color:#334155; font-size:0.75rem">
    EfficientNet-B0 · ImageNet pretrained · Fine-tuned on 100 Sports Dataset
</p>
""", unsafe_allow_html=True)
