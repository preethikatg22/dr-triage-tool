"""
Diabetic Retinopathy Triage Tool - Streamlit version.

Self-contained, same as app.py (Gradio version): no dependency on the Kaggle
notebook's other scripts. Deployment needs just this file + requirements.txt
+ the two model files below, pushed to a GitHub repo and connected at
share.streamlit.io (Streamlit Community Cloud, free).

To deploy:
  1. Create a GitHub repo containing: streamlit_app.py, requirements.txt,
     best_model.pt, calibration.json (the last two from your Kaggle run1/).
  2. Go to share.streamlit.io, sign in with GitHub, "New app", pick the repo
     and this file as the entry point.
  3. It builds automatically and gives you a permanent
     https://<something>.streamlit.app URL.

Note: best_model.pt is typically 40-150MB depending on backbone. GitHub's
default per-file limit is 100MB - if your file is larger, use Git LFS
(https://git-lfs.com) when pushing it, or host the weights elsewhere and
download them at startup (ask if you want that version instead).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import streamlit as st

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MODEL_DIR = Path(__file__).parent


# ----------------------------------------------------------------------------
# Preprocessing (identical to the training pipeline)
# ----------------------------------------------------------------------------
def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pad_to_square(img):
    h, w = img.shape[:2]
    d = abs(h - w)
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, d // 2, d - d // 2, cv2.BORDER_CONSTANT, value=0)
    if w > h:
        return cv2.copyMakeBorder(img, d // 2, d - d // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


def retina_mask(img, tol=7, erode_frac=0.04):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = (gray > tol).astype(np.uint8)
    k = max(3, int(0.05 * min(h, w)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    disc = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disc, (w // 2, h // 2), int(min(h, w) / 2 * 0.98), 1, -1)
    m = m & disc
    e = max(3, int(erode_frac * min(h, w)) | 1)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))


def ben_graham(img, mask, sigma_frac=1 / 30):
    sigma = img.shape[0] * sigma_frac
    m = mask.astype(np.float32)
    imgf = img.astype(np.float32)
    num = cv2.GaussianBlur(imgf * m[..., None], (0, 0), sigma)
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-3)[..., None]
    out = np.clip(4.0 * imgf - 4.0 * (num / den) + 128.0, 0, 255).astype(np.uint8)
    out[mask == 0] = 128
    return out


def preprocess_fundus(img_rgb, size=300):
    img = crop_black_borders(img_rgb)
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = retina_mask(img)
    return ben_graham(img, mask)


def to_tensor(processed_uint8, device):
    x = processed_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


# ----------------------------------------------------------------------------
# Model: EfficientNet-B3 + CORAL ordinal head (identical to training)
# ----------------------------------------------------------------------------
class CoralHead(nn.Module):
    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias


class DRModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.head = CoralHead(self.backbone.num_features)

    def forward(self, x):
        return self.head(self.backbone(x))


def coral_predict(logits):
    return int((torch.sigmoid(logits) > 0.5).sum(dim=1).item())


def coral_probs(logits):
    p_gt = torch.cummin(torch.sigmoid(logits), dim=1).values
    ones = torch.ones_like(p_gt[:, :1])
    zeros = torch.zeros_like(ones)
    p_ge = torch.cat([ones, p_gt], dim=1)
    p_next = torch.cat([p_gt, zeros], dim=1)
    return (p_ge - p_next).clamp_min(0).cpu().numpy()[0]


# ----------------------------------------------------------------------------
# Grad-CAM (same auto-resolution layer picker as the notebook version)
# ----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations, self.gradients = None, None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def find_target_layer(model, size, device, min_res=14):
    blocks = model.backbone.blocks
    shapes = {}

    def make_hook(i):
        def hook(module, inp, out):
            shapes[i] = out.shape[-2:]
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=device))
    for h in handles:
        h.remove()
    candidates = [i for i, (h, w) in shapes.items() if min(h, w) >= min_res]
    chosen = max(candidates) if candidates else max(shapes, key=lambda i: min(shapes[i]))
    return blocks[chosen]


def overlay_heatmap(img_rgb_uint8, cam, alpha=0.4):
    h, w = img_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(img_rgb_uint8 * (1 - alpha) + heat * alpha)


# ----------------------------------------------------------------------------
# Load model + calibration once, cached across Streamlit reruns
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    size = ckpt.get("size", 300)
    model = DRModel(ckpt.get("backbone", "efficientnet_b3")).to(device).eval()
    model.load_state_dict(ckpt["model"])
    gradcam = GradCAM(model, find_target_layer(model, size, device))

    calib_path = MODEL_DIR / "calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        temperature = calib["temperature"]
        flag_threshold = calib["flags"]["cutoff_2"]["threshold"]
    else:
        temperature, flag_threshold = 1.0, 0.5

    return {"model": model, "gradcam": gradcam, "device": device, "size": size,
            "temperature": temperature, "flag_threshold": flag_threshold}


def predict(assets, img_rgb):
    device, size, T = assets["device"], assets["size"], assets["temperature"]
    processed = preprocess_fundus(img_rgb, size=size)
    x = to_tensor(processed, device)

    cam, logits = assets["gradcam"](x)
    scaled = logits / T
    grade = coral_predict(scaled)
    probs = coral_probs(scaled)
    confidence = float(probs[grade])

    p_ge2 = float(torch.sigmoid(logits[0, 1] / T))  # P(grade >= 2) = Moderate or worse
    urgent = p_ge2 >= assets["flag_threshold"]

    overlay = overlay_heatmap(processed, cam)
    return processed, overlay, grade, confidence, probs, urgent, p_ge2


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Diabetic Retinopathy Triage Tool", layout="wide")
st.title("Diabetic Retinopathy Triage Tool")
st.warning(
    "**This is a research prototype, not a diagnostic device.** It has not "
    "been clinically validated and must not be used to make or withhold a "
    "clinical decision without review by a qualified ophthalmologist."
)

assets = load_assets()

uploaded = st.file_uploader("Upload a fundus photo", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Analyzing..."):
        processed, overlay, grade, confidence, probs, urgent, p_ge2 = predict(assets, img_rgb)

    col1, col2, col3 = st.columns(3)
    col1.image(img_rgb, caption="Uploaded image", use_container_width=True)
    col2.image(processed, caption="Preprocessed", use_container_width=True)
    col3.image(overlay, caption="Grad-CAM (where the model is looking)", use_container_width=True)

    st.subheader(f"Predicted severity: {CLASS_NAMES[grade]} (grade {grade})")
    st.write(f"**Confidence:** {confidence:.1%} (calibrated)")

    if urgent:
        st.error(f"🔴 **URGENT — flag for specialist review**  "
                 f"(P[Moderate or worse] = {p_ge2:.1%}, threshold {assets['flag_threshold']:.0%})")
    else:
        st.success(f"🟢 **Routine**  "
                   f"(P[Moderate or worse] = {p_ge2:.1%}, threshold {assets['flag_threshold']:.0%})")

    st.subheader("Full probability breakdown")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}")
        st.progress(float(p), text=f"{p:.1%}")
else:
    st.info("Upload a fundus photo above to get a severity grade, confidence score, "
            "urgent-review flag, and Grad-CAM heatmap.")
