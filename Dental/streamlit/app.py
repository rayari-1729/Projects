import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="OralSense", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size:28px;
    font-weight:700;
    color:#0b5394;
}
.metric-card {
    background-color:#f7f9fc;
    padding:15px;
    border-radius:10px;
    border:1px solid #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦷 OralSense — Dental Lesion AI </div>', unsafe_allow_html=True)
st.write("AI-assisted dental lesion classification with visual explanation")

st.markdown("---")

# -----------------------
# Model definition
# -----------------------
class CMLS(nn.Module):
    def __init__(self, num_classes, text_dim=384, r=16):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.vision = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_dim = 1280

        self.low_rank = nn.Sequential(
            nn.Linear(self.vision_dim, r),
            nn.ReLU(),
            nn.Linear(r, text_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, class_text_embeds):
        B = img.size(0)
        v = self.avgpool(self.vision(img)).view(B, -1)
        q = self.low_rank(v)

        qn = F.normalize(q, dim=1)
        en = F.normalize(class_text_embeds, dim=1)
        attn_logits = qn @ en.T
        w = F.softmax(attn_logits, dim=1)
        prompt = w @ class_text_embeds

        fused = torch.cat([v, prompt], dim=1)
        logits = self.classifier(fused)
        return logits, attn_logits


# -----------------------
# GradCAM
# -----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_text_embeds, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(x, class_text_embeds)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze(1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()[0]


def overlay_cam_on_image(pil_img, cam, alpha=0.45):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Model Settings")

ckpt_path = st.sidebar.text_input(
    "Checkpoint path",
    value=r"\oral_sense.pt"
)

show_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True)
topk = st.sidebar.slider("Top-K predictions", 2, 5, 3)

# -----------------------
# Transforms
# -----------------------
val_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model_and_assets(path):
    device = "cpu"
    ckpt = torch.load(path, map_location=device)

    classes = ckpt["classes"]
    model = CMLS(len(classes), ckpt.get("text_dim", 384), ckpt.get("r", 16))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_text_embeds = ckpt["class_text_embeds"].to(device).float()
    return model, class_text_embeds, classes, device


model, class_text_embeds, classes, device = load_model_and_assets(ckpt_path)

# -----------------------
# Upload section
# -----------------------
uploaded = st.file_uploader("Upload oral image", type=["jpg", "png", "jpeg"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    x = val_tfms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x, class_text_embeds)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_name = classes[pred_idx]
    pred_prob = float(probs[pred_idx])

    overlay = None
    if show_gradcam:
        cam_engine = GradCAM(model, model.vision[-1])
        cam = cam_engine(x, class_text_embeds, pred_idx)
        cam_engine.remove()
        overlay = overlay_cam_on_image(pil_img, cam)

    # -----------------------
    # Dashboard layout
    # -----------------------
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.image(pil_img, caption="Input Image", width=380)

    with col_img2:
        if overlay is not None:
            st.image(overlay, caption="Model Attention (Grad-CAM)", width=380)

    st.markdown("---")

    colA, colB, colC = st.columns(3)

    colA.metric("Prediction", pred_name)
    colB.metric("Confidence", f"{pred_prob:.3f}")
    colC.metric("Classes", len(classes))

    st.markdown("---")
    st.subheader("Top Predictions")

    top_idxs = np.argsort(-probs)[:topk]
    for i in top_idxs:
        st.progress(float(probs[int(i)]), text=f"{classes[int(i)]} — {probs[int(i)]:.3f}")
