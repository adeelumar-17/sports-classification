import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# Model Config — .pth file should be in the same repo directory
# ============================================================
MODEL_FILENAME = "sports_efficientnet_b0_pytorch.pth"
CLASS_INDICES_FILE = "class_indices.json"     # optional local fallback

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# Model Architecture (matches updated notebook — 3-layer head)
# ============================================================

def build_classifier_head(in_features, num_classes, dropout):
    """
    Upgraded 3-layer classifier head with BatchNorm.
    Architecture: in_features → 1024 → 512 → num_classes
    Must match the notebook's build_classifier_head exactly.
    """
    return nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(512, num_classes)
    )


def create_model(num_classes, dropout=0.3):
    """Build EfficientNet-B0 with the upgraded classifier head."""
    base = models.efficientnet_b0(weights=None)  # no pretrained needed at inference
    in_features = base.classifier[1].in_features
    base.classifier = build_classifier_head(in_features, num_classes, dropout)
    return base


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(
    page_title="🏅 Sports Classifier — EfficientNet-B0",
    page_icon="🏅",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f97316 0%, #ef4444 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    .metric-card h3 {
        color: #f97316;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card p {
        color: #fff;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }

    .pred-bar {
        background: linear-gradient(90deg, #f97316, #ef4444);
        border-radius: 6px;
        height: 28px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 6px;
        transition: width 0.6s ease;
    }

    .pred-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .pred-label {
        width: 180px;
        text-align: right;
        padding-right: 12px;
        font-weight: 500;
        font-size: 0.9rem;
        color: #ccc;
    }
    .pred-value {
        font-weight: 600;
        font-size: 0.85rem;
        padding-left: 8px;
        color: #f97316;
        min-width: 55px;
    }

    .image-label {
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem 0;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏅 Sports Image Classifier</h1>
    <p>EfficientNet-B0 fine-tuned on 100 sports classes — upload an image and get predictions</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model():
    """Load the EfficientNet model from local .pth file (cached)."""
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"❌ Model file `{MODEL_FILENAME}` not found. Make sure it's in the repo root.")
        st.stop()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_FILENAME, map_location=device, weights_only=False)

    num_classes = checkpoint.get('num_classes', 100)
    img_size    = checkpoint.get('img_size', 224)
    idx_to_class = checkpoint.get('idx_to_class', {})

    # Convert string keys back to int if needed
    if idx_to_class and isinstance(list(idx_to_class.keys())[0], str):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, device, idx_to_class, img_size, num_classes


def get_transform(img_size):
    """Build inference transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def predict(model, image, transform, device, idx_to_class, top_k=5,
            temperature=1.0, use_tta=False, img_size=224):
    """
    Run inference with optional adjustments.
    
    Args:
        temperature: softmax temperature scaling (>1 = softer, <1 = sharper)
        use_tta: whether to use Test-Time Augmentation (horizontal flip)
        top_k: number of top predictions to return
    """
    model.eval()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

        if use_tta:
            # Horizontal flip augmentation
            flip_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            tensor_flip = flip_transform(image).unsqueeze(0).to(device)
            logits_flip = model(tensor_flip)
            logits = (logits + logits_flip) / 2.0

        # Temperature scaling
        logits = logits / temperature
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(top_k)
    results = []
    for idx, prob in zip(top_idxs, top_probs):
        label = idx_to_class.get(idx.item(), f"Class {idx.item()}")
        results.append((label, prob.item()))

    return results, probs


# ============================================================
# Sidebar — Inference Controls
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Inference Settings")

    top_k = st.slider(
        "🔝 Top-K Predictions",
        min_value=1, max_value=20, value=5, step=1,
        help="Number of top predictions to display"
    )

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Softmax temperature: <1 = sharper/more confident, >1 = softer/more spread out"
    )

    use_tta = st.checkbox(
        "🔄 Test-Time Augmentation",
        value=False,
        help="Average predictions with horizontally-flipped image for potentially better accuracy"
    )

    confidence_threshold = st.slider(
        "📊 Confidence Threshold (%)",
        min_value=0, max_value=100, value=0, step=5,
        help="Only show predictions above this confidence. Set to 0 to show all Top-K."
    )

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
    **Model:** EfficientNet-B0  
    **Head:** 3-layer + BatchNorm  
    **Classes:** 100 sports  
    **Input:** 224×224 RGB  
    
    Trained with hyperparameter ablation  
    on the Sports Classification dataset.
    """)


# ============================================================
# Main Content
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a sports image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload any image — it will be resized to 224×224 for classification"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    try:
        model, device, idx_to_class, img_size, num_classes = load_model()
        device_name = "GPU 🟢" if device.type == 'cuda' else "CPU 🟡"
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    transform = get_transform(img_size)

    # Run prediction
    with st.spinner("🔄 Classifying..."):
        results, all_probs = predict(
            model, image, transform, device, idx_to_class,
            top_k=top_k, temperature=temperature,
            use_tta=use_tta, img_size=img_size
        )

    # Filter by confidence threshold
    if confidence_threshold > 0:
        results = [(label, prob) for label, prob in results if prob * 100 >= confidence_threshold]

    # Layout: image left, predictions right
    col_img, col_pred = st.columns([1, 1.5])

    with col_img:
        st.markdown('<div class="image-label">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_pred:
        if results:
            st.markdown('<div class="image-label">🏆 Predictions</div>', unsafe_allow_html=True)
            top_label = results[0][0].replace('_', ' ').title()
            top_conf  = results[0][1] * 100

            # Top prediction highlight
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Predicted Sport</h3>
                <p>{top_label}</p>
                <p style="font-size: 1rem; color: #94a3b8; margin-top: 4px;">{top_conf:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

            # All predictions as bars
            for label, prob in results:
                nice_label = label.replace('_', ' ').title()
                pct = prob * 100
                bar_width = max(pct, 2)
                is_top = (label == results[0][0])
                bar_color = "linear-gradient(90deg, #f97316, #ef4444)" if is_top else "linear-gradient(90deg, #475569, #64748b)"
                st.markdown(f"""
                <div class="pred-row">
                    <div class="pred-label">{nice_label}</div>
                    <div style="flex: 1;">
                        <div class="pred-bar" style="width: {bar_width}%; background: {bar_color};">
                        </div>
                    </div>
                    <div class="pred-value">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No predictions above {confidence_threshold}% confidence threshold.")

    # Metrics row
    st.divider()
    m1, m2, m3, m4 = st.columns(4)

    # Entropy of prediction distribution (measures uncertainty)
    entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-9)).item()
    max_entropy = np.log(num_classes)
    uncertainty_pct = (entropy / max_entropy) * 100

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Device</h3>
            <p>{device_name}</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Temperature</h3>
            <p>{temperature:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Uncertainty</h3>
            <p>{uncertainty_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>TTA</h3>
            <p>{"On ✓" if use_tta else "Off"}</p>
        </div>
        """, unsafe_allow_html=True)

    # Show full distribution in expander
    with st.expander("📊 Full Probability Distribution (all classes)"):
        class_probs = []
        for i in range(num_classes):
            lbl = idx_to_class.get(i, f"Class {i}").replace('_', ' ').title()
            class_probs.append({"Sport": lbl, "Confidence (%)": round(all_probs[i].item() * 100, 3)})
        class_probs.sort(key=lambda x: x["Confidence (%)"], reverse=True)

        import pandas as pd
        df = pd.DataFrame(class_probs)
        st.dataframe(df, use_container_width=True, height=400)

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #666;">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">🏅</p>
        <p style="font-size: 1.2rem;">Upload a sports image to get started</p>
        <p style="font-size: 0.9rem; max-width: 500px; margin: 0 auto;">
            The model will classify it into one of 100 sports categories using an
            EfficientNet-B0 backbone with an upgraded 3-layer classifier head.
        </p>
    </div>
    """, unsafe_allow_html=True)
