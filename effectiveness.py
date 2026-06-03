"""
COVID-19 X-Ray — Model Explainability & Evaluation
====================================================
Explainability Methods
----------------------
  CNN  : GradCAM  (Attribution Map via gradient-weighted activations)
         Integrated Gradients  (pixel-level attribution via Captum)
         Gradient × Input      (fast saliency baseline)

  ViT  : Attention Rollout     (propagates attention across all layers)
         Raw Attention         (last-layer [CLS] attention tokens)
         Integrated Gradients  (same Captum method, model-agnostic)

Quantitative Metrics
--------------------
  • Insertion  — AUC when pixels are revealed high→low importance
  • Deletion   — AUC when pixels are removed high→low importance
  • Entropy    — Shannon entropy of normalised saliency map
  • AOPC       — Area Over Perturbation Curve (MoRF order)

Usage
-----
  python explainability.py          # uses synthetic dataset + trained models
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from captum.attr import IntegratedGradients, GradientShap, Saliency

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

DATA_ROOT  = "COVID_19_dataset"
MODEL_DIR  = "outputs"
OUT_DIR    = "outputs/xai"
IMG_SIZE   = 224         # must match training
N_SAMPLES  = 6           # images per class to explain
AOPC_STEPS = 20          # perturbation steps for AOPC / Insertion / Deletion
PATCH_FRAC = 0.05        # fraction of pixels perturbed per step

os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES  = ["COVID", "Normal", "Viral Pneumonia"]
CLASS_COLORS = {"COVID": "#E63946", "Normal": "#2A9D8F", "Viral Pneumonia": "#E9C46A"}

SALIENCY_CMAP = LinearSegmentedColormap.from_list(
    "saliency", ["#000000", "#FF4500", "#FFD700"], N=256
)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.skip(x))

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock(32,  64),     ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64,  128, 2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop),
        )
        # store last attention weights for rollout
        self.last_attn_weights = None

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_w = self.attn(normed, normed, normed, need_weights=True,
                                      average_attn_weights=False)
        self.last_attn_weights = attn_w.detach()   # (B, heads, N+1, N+1)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))

class ViT(nn.Module):
    def __init__(self, num_classes=3, img_size=224, patch_size=16,
                 embed_dim=256, depth=8, heads=8):
        super().__init__()
        n_patches        = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks      = nn.ModuleList(
            [TransformerBlock(embed_dim, heads) for _ in range(depth)]
        )
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.patch_size  = patch_size
        self.n_patches_side = img_size // patch_size

    def forward(self, x):
        B   = x.size(0)
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm(t):
    """Denormalise a (3,H,W) tensor to [0,1]."""
    return (t.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

def load_test_samples(root, n_per_class=N_SAMPLES):
    """Returns list of (img_tensor, label_int, class_name)."""
    test_ds = datasets.ImageFolder(os.path.join(root, "test"), eval_tf)
    samples = []
    counts  = {i: 0 for i in range(len(test_ds.classes))}
    loader  = DataLoader(test_ds, batch_size=1, shuffle=True)
    for img, label in loader:
        lbl = label.item()
        if counts[lbl] < n_per_class:
            samples.append((img.squeeze(0), lbl, test_ds.classes[lbl]))
            counts[lbl] += 1
        if all(v >= n_per_class for v in counts.values()):
            break
    print(f"[Data] Loaded {len(samples)} test samples for XAI")
    return samples

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_models():
    cnn = ResNetCNN(3)
    vit = ViT(3, img_size=IMG_SIZE, patch_size=16, embed_dim=256, depth=8, heads=8)

    cnn_path = os.path.join(MODEL_DIR, "ResNet_CNN_weights.pth")
    vit_path = os.path.join(MODEL_DIR, "ViT_weights.pth")

    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        print(f"[Model] ResNet-CNN weights loaded from {cnn_path}")
    else:
        print(f"[Model] ResNet-CNN: no weights found, using random init")

    if os.path.exists(vit_path):
        vit.load_state_dict(torch.load(vit_path, map_location="cpu"))
        print(f"[Model] ViT weights loaded from {vit_path}")
    else:
        print(f"[Model] ViT: no weights found, using random init")

    cnn.eval().to(DEVICE)
    vit.eval().to(DEVICE)
    return cnn, vit

####### METHODS USED: 

# 1. GradCAM (CNN)
class GradCAM:
    """
    Selects the last convolutional layer, registers hooks to capture
    activations and gradients, then forms a weighted sum of feature maps.
    """
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._hooks.append(
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            )
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            )
        )

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        x = x.to(DEVICE).unsqueeze(0).requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

    def remove(self):
        for h in self._hooks:
            h.remove()

# 2. Integrated Gradients
def integrated_gradients_map(model, img_tensor, class_idx):
    """
    Approximates the integral of gradients along a straight path from
    a black baseline to the input. Returns (H,W) saliency map.
    """
    ig      = IntegratedGradients(model)
    inp     = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    baseline= torch.zeros_like(inp)
    attrs   = ig.attribute(inp, baseline, target=class_idx,
                           n_steps=50, internal_batch_size=10)
    sal     = attrs.squeeze(0).abs().mean(0).cpu().detach().numpy()
    if sal.max() > sal.min():
        sal = (sal - sal.min()) / (sal.max() - sal.min())
    return sal

# 3. Gradient × Input
def gradient_x_input_map(model, img_tensor, class_idx):
    """Element-wise product of input and gradient — cheap but informative."""
    model.zero_grad()
    inp = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    out = model(inp)
    out[0, class_idx].backward()
    sal = (inp.grad.squeeze(0).abs() * inp.squeeze(0).abs()).mean(0)
    sal = sal.cpu().detach().numpy()
    if sal.max() > sal.min():
        sal = (sal - sal.min()) / (sal.max() - sal.min())
    return sal

# 4. Attention Rollout (ViT)
def attention_rollout(vit_model, img_tensor):
    """
    Crismann et al. 2020: multiply attention matrices across all layers,
    adding the identity (residual) at each step to account for skip connections.
    Returns (H, W) spatial map.
    """
    with torch.no_grad():
        vit_model(img_tensor.unsqueeze(0).to(DEVICE))

    rollout = None
    for blk in vit_model.blocks:
        # attn_w: (B, heads, N+1, N+1)
        attn_w = blk.last_attn_weights[0]           # (heads, N+1, N+1)
        attn_w = attn_w.mean(0).cpu().float().numpy()  # average over heads

        # Add identity (skip connection) and renormalise rows
        n      = attn_w.shape[0]
        attn_w = attn_w + np.eye(n)
        attn_w = attn_w / attn_w.sum(axis=-1, keepdims=True)

        rollout = attn_w if rollout is None else attn_w @ rollout

    cls_attn = rollout[0, 1:]     
    side     = vit_model.n_patches_side
    cls_attn = cls_attn.reshape(side, side)
    sal      = np.array(
        matplotlib.image.AxesImage(None).to_rgba(
            cls_attn, cmap="gray"
        )
    ) if False else cls_attn  
    from PIL import Image as PILImage
    sal_img  = PILImage.fromarray(
        (cls_attn / (cls_attn.max() + 1e-8) * 255).astype(np.uint8)
    ).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    sal      = np.array(sal_img).astype(float) / 255.0
    return sal

# 5. Raw Last-Layer Attention (ViT)
def raw_attention_map(vit_model, img_tensor):
    """
    Uses the CLS-token attention from the LAST transformer block only.
    Faster and simpler than rollout; sometimes noisier.
    """
    with torch.no_grad():
        vit_model(img_tensor.unsqueeze(0).to(DEVICE))

    last_blk = vit_model.blocks[-1]
    attn_w   = last_blk.last_attn_weights[0]          
    cls_attn = attn_w.mean(0)[0, 1:].cpu().numpy()    
    side     = vit_model.n_patches_side
    cls_attn = cls_attn.reshape(side, side)

    from PIL import Image as PILImage
    sal_img  = PILImage.fromarray(
        (cls_attn / (cls_attn.max() + 1e-8) * 255).astype(np.uint8)
    ).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    sal      = np.array(sal_img).astype(float) / 255.0
    return sal

### METRICS

def _sorted_pixel_indices(saliency_map):
    """Return flat pixel indices sorted from most to least important (MoRF order)."""
    flat = saliency_map.flatten()
    return np.argsort(flat)[::-1]   # descending importance

def _perturb_image(img_tensor, mask_indices, mode="delete"):
    """
    Delete  → set important pixels to mean (baseline).
    Insert  → start from mean baseline, reveal pixels at mask_indices.
    Returns (3, H, W) tensor.
    """
    img  = img_tensor.clone()            # (3, H, W)
    mean = img.mean(dim=(1, 2), keepdim=True).expand_as(img).clone()
    flat_img  = img.view(3, -1)
    flat_mean = mean.view(3, -1)

    if mode == "delete":
        out = flat_img.clone()
        out[:, mask_indices] = flat_mean[:, mask_indices]
    else:  # insert
        out = flat_mean.clone()
        out[:, mask_indices] = flat_img[:, mask_indices]

    return out.view_as(img)

@torch.no_grad()
def _predict_prob(model, img_tensor, class_idx):
    """Return softmax probability for class_idx on a single (3,H,W) tensor."""
    logits = model(img_tensor.unsqueeze(0).to(DEVICE))
    return F.softmax(logits, dim=1)[0, class_idx].item()

def insertion_deletion(model, img_tensor, saliency_map, class_idx, steps=AOPC_STEPS):
    """
    Sweep pixels from most→least important.
    Deletion : remove them   → expect confidence to drop quickly for good saliency.
    Insertion: reveal them   → expect confidence to rise quickly for good saliency.
    Returns dict with AUC for both curves and the curves themselves.
    """
    sorted_idx  = _sorted_pixel_indices(saliency_map)
    n_pixels    = len(sorted_idx)
    step_size   = max(1, n_pixels // steps)

    del_probs, ins_probs = [], []
    del_fracs,  ins_fracs = [], []

    for k in range(0, n_pixels, step_size):
        frac       = k / n_pixels
        top_k      = sorted_idx[:k] if k > 0 else np.array([], dtype=int)

        del_img    = _perturb_image(img_tensor, top_k, mode="delete")
        ins_img    = _perturb_image(img_tensor, top_k, mode="insert")

        del_probs.append(_predict_prob(model, del_img, class_idx))
        ins_probs.append(_predict_prob(model, ins_img, class_idx))
        del_fracs.append(frac)
        ins_fracs.append(frac)

    # Normalise fracs to [0,1] and compute AUC via trapezoid rule
    del_auc = float(np.trapz(del_probs, del_fracs) / (del_fracs[-1] - del_fracs[0] + 1e-8))
    ins_auc = float(np.trapz(ins_probs, ins_fracs) / (ins_fracs[-1] - ins_fracs[0] + 1e-8))

    return {
        "deletion_auc" : del_auc,
        "insertion_auc": ins_auc,
        "deletion_curve" : (del_fracs, del_probs),
        "insertion_curve": (ins_fracs, ins_probs),
    }

def saliency_entropy(saliency_map):
    """
    Shannon entropy of the normalised saliency distribution.
    Lower entropy → more focused / peaked saliency (better localisation).
    Higher entropy → diffuse / noisy saliency.
    """
    flat = saliency_map.flatten().astype(np.float64)
    flat = flat - flat.min()
    s    = flat.sum()
    if s < 1e-12:
        return 0.0
    p    = flat / s
    p    = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))

def aopc(model, img_tensor, saliency_map, class_idx, steps=AOPC_STEPS):
    """
    AOPC measures how fast the model's confidence drops as the most
    important pixels are removed one batch at a time (MoRF order).

    AOPC = (1/K) * Σ_k [ f(x) − f(x with top-k pixels removed) ]

    Higher AOPC → saliency correctly identifies the most decisive pixels.
    """
    sorted_idx = _sorted_pixel_indices(saliency_map)
    n_pixels   = len(sorted_idx)
    step_size  = max(1, n_pixels // steps)

    f0  = _predict_prob(model, img_tensor, class_idx)   # original confidence
    drops, fracs = [], []

    for k in range(step_size, n_pixels, step_size):
        top_k   = sorted_idx[:k]
        del_img = _perturb_image(img_tensor, top_k, mode="delete")
        fk      = _predict_prob(model, del_img, class_idx)
        drops.append(f0 - fk)
        fracs.append(k / n_pixels)

    aopc_score = float(np.mean(drops)) if drops else 0.0
    return {"aopc": aopc_score, "drops": drops, "fracs": fracs, "f0": f0}



def overlay(img_np, sal_map, alpha=0.55):
    """Blend saliency (H,W) over RGB image (H,W,3)."""
    colored = plt.cm.jet(sal_map)[..., :3]
    return (1 - alpha) * img_np + alpha * colored

def plot_qualitative_cnn(samples, cnn_model, save_dir):
    """Grid: original | GradCAM | IntGrad | Grad×Input  for each sample."""
    print("[Plot] CNN qualitative maps ...")
    cam_extractor = GradCAM(cnn_model, cnn_model.layer4[1].body[0])

    n   = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1: axes = axes[np.newaxis]

    col_titles = ["Original", "GradCAM", "Integrated Gradients", "Gradient × Input"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=11, fontweight="bold", pad=8)

    for row, (img_t, lbl, cls_name) in enumerate(samples):
        img_np = denorm(img_t).permute(1, 2, 0).numpy()

        cam, pred_idx = cam_extractor(img_t)
        ig_map = integrated_gradients_map(cnn_model, img_t, pred_idx)
        gxi    = gradient_x_input_map(cnn_model, img_t, pred_idx)

        axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"{cls_name}\n(pred={CLASS_NAMES[pred_idx]})",
                                fontsize=9, rotation=0, labelpad=60, va="center")

        for col, sal in enumerate([cam, ig_map, gxi], start=1):
            axes[row, col].imshow(overlay(img_np, sal))
            axes[row, col].axis("off")

    plt.suptitle("CNN Attribution Maps — COVID-19 X-Ray", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "cnn_qualitative.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    cam_extractor.remove()
    print(f"  Saved: {path}")

def plot_qualitative_vit(samples, vit_model, save_dir):
    """Grid: original | Attention Rollout | Raw Attention | IntGrad."""
    print("[Plot] ViT qualitative maps ...")
    n   = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1: axes = axes[np.newaxis]

    col_titles = ["Original", "Attention Rollout", "Raw Attention (last)", "Integrated Gradients"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=11, fontweight="bold", pad=8)

    for row, (img_t, lbl, cls_name) in enumerate(samples):
        img_np = denorm(img_t).permute(1, 2, 0).numpy()

        with torch.no_grad():
            logits   = vit_model(img_t.unsqueeze(0).to(DEVICE))
            pred_idx = logits.argmax(1).item()

        rollout  = attention_rollout(vit_model, img_t)
        raw_att  = raw_attention_map(vit_model, img_t)
        ig_map   = integrated_gradients_map(vit_model, img_t, pred_idx)

        axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"{cls_name}\n(pred={CLASS_NAMES[pred_idx]})",
                                fontsize=9, rotation=0, labelpad=60, va="center")

        for col, sal in enumerate([rollout, raw_att, ig_map], start=1):
            axes[row, col].imshow(overlay(img_np, sal))
            axes[row, col].axis("off")

    plt.suptitle("ViT Attention Maps — COVID-19 X-Ray", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "vit_qualitative.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
#  ── QUANTITATIVE EVALUATION PIPELINE ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_methods(samples, cnn_model, vit_model):
    """
    For each sample, compute all saliency methods, then score them with
    Insertion/Deletion, Entropy, and AOPC.
    Returns nested dict: results[model_name][method_name] = list of metric dicts.
    """
    print("\n[Eval] Computing quantitative metrics ...")
    cam_extractor = GradCAM(cnn_model, cnn_model.layer4[1].body[0])

    results = {
        "ResNet-CNN": {m: [] for m in ["GradCAM", "IntGrad", "Grad×Input"]},
        "ViT":        {m: [] for m in ["Rollout", "RawAttn", "IntGrad"]},
    }

    for i, (img_t, lbl, cls_name) in enumerate(samples):
        print(f"  Sample {i+1}/{len(samples)}  ({cls_name})")

        # ── CNN ────────────────────────────────────────────────────────
        with torch.no_grad():
            cnn_pred = cnn_model(img_t.unsqueeze(0).to(DEVICE)).argmax(1).item()

        cam_map, _ = cam_extractor(img_t)
        ig_cnn     = integrated_gradients_map(cnn_model, img_t, cnn_pred)
        gxi_cnn    = gradient_x_input_map(cnn_model, img_t, cnn_pred)

        for method_name, sal in [("GradCAM", cam_map),
                                  ("IntGrad", ig_cnn),
                                  ("Grad×Input", gxi_cnn)]:
            ins_del = insertion_deletion(cnn_model, img_t, sal, cnn_pred)
            aop     = aopc(cnn_model, img_t, sal, cnn_pred)
            ent     = saliency_entropy(sal)
            results["ResNet-CNN"][method_name].append({
                "insertion_auc" : ins_del["insertion_auc"],
                "deletion_auc"  : ins_del["deletion_auc"],
                "entropy"       : ent,
                "aopc"          : aop["aopc"],
                "ins_curve"     : ins_del["insertion_curve"],
                "del_curve"     : ins_del["deletion_curve"],
                "aopc_fracs"    : aop["fracs"],
                "aopc_drops"    : aop["drops"],
            })

        # ── ViT ────────────────────────────────────────────────────────
        with torch.no_grad():
            vit_pred = vit_model(img_t.unsqueeze(0).to(DEVICE)).argmax(1).item()

        rollout  = attention_rollout(vit_model, img_t)
        raw_att  = raw_attention_map(vit_model, img_t)
        ig_vit   = integrated_gradients_map(vit_model, img_t, vit_pred)

        for method_name, sal in [("Rollout", rollout),
                                  ("RawAttn", raw_att),
                                  ("IntGrad", ig_vit)]:
            ins_del = insertion_deletion(vit_model, img_t, sal, vit_pred)
            aop     = aopc(vit_model, img_t, sal, vit_pred)
            ent     = saliency_entropy(sal)
            results["ViT"][method_name].append({
                "insertion_auc" : ins_del["insertion_auc"],
                "deletion_auc"  : ins_del["deletion_auc"],
                "entropy"       : ent,
                "aopc"          : aop["aopc"],
                "ins_curve"     : ins_del["insertion_curve"],
                "del_curve"     : ins_del["deletion_curve"],
                "aopc_fracs"    : aop["fracs"],
                "aopc_drops"    : aop["drops"],
            })

    cam_extractor.remove()
    return results


METHOD_COLORS = {
    # CNN
    "GradCAM":   "#E63946",
    "IntGrad":   "#457B9D",
    "Grad×Input":"#2A9D8F",
    # ViT
    "Rollout":   "#E9C46A",
    "RawAttn":   "#F4A261",
}

def _avg(lst, key):
    return np.mean([d[key] for d in lst])

def plot_insertion_deletion_curves(results, save_dir):
    """Average insertion & deletion curves per method per model."""
    print("[Plot] Insertion / Deletion curves ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = [
        ("ResNet-CNN", "Insertion"),
        ("ResNet-CNN", "Deletion"),
        ("ViT",        "Insertion"),
        ("ViT",        "Deletion"),
    ]

    for ax, (model_name, curve_type) in zip(axes.flat, titles):
        key = "ins_curve" if curve_type == "Insertion" else "del_curve"
        auc_key = "insertion_auc" if curve_type == "Insertion" else "deletion_auc"
        for method, data_list in results[model_name].items():
            fracs  = data_list[0][key][0]
            avg_p  = np.mean([d[key][1] for d in data_list], axis=0)
            auc    = _avg(data_list, auc_key)
            color  = METHOD_COLORS.get(method, "#888")
            ax.plot(fracs, avg_p, "-o", ms=3, color=color,
                    label=f"{method} (AUC={auc:.3f})", linewidth=2)

        ax.set_title(f"{model_name} — {curve_type}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fraction of pixels perturbed")
        ax.set_ylabel("Model confidence")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        if curve_type == "Insertion":
            ax.annotate("Higher AUC = better", xy=(0.5, 0.95), xycoords="axes fraction",
                        ha="center", fontsize=8, color="green",
                        arrowprops=None, style="italic")
        else:
            ax.annotate("Lower AUC = better", xy=(0.5, 0.95), xycoords="axes fraction",
                        ha="center", fontsize=8, color="red", style="italic")

    plt.suptitle("Insertion & Deletion Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "insertion_deletion_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def plot_aopc_curves(results, save_dir):
    """Average AOPC perturbation curves."""
    print("[Plot] AOPC curves ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_name in zip(axes, ["ResNet-CNN", "ViT"]):
        for method, data_list in results[model_name].items():
            fracs  = data_list[0]["aopc_fracs"]
            n      = min(len(fracs), min(len(d["aopc_drops"]) for d in data_list))
            fracs  = fracs[:n]
            avg_d  = np.mean([d["aopc_drops"][:n] for d in data_list], axis=0)
            cum_d  = np.cumsum(avg_d) / (np.arange(1, n + 1))   # running mean
            aoc    = _avg(data_list, "aopc")
            color  = METHOD_COLORS.get(method, "#888")
            ax.plot(fracs, cum_d, "-o", ms=3, color=color,
                    label=f"{method} (AOPC={aoc:.3f})", linewidth=2)

        ax.set_title(f"{model_name} — AOPC", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fraction of pixels removed")
        ax.set_ylabel("Running mean confidence drop")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    plt.suptitle("Area Over Perturbation Curve (AOPC) — MoRF Order",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "aopc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def plot_entropy_comparison(results, save_dir):
    """Box-and-strip plot of saliency entropy per method."""
    print("[Plot] Entropy comparison ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, model_name in zip(axes, ["ResNet-CNN", "ViT"]):
        method_names, entropies, colors = [], [], []
        for method, data_list in results[model_name].items():
            vals = [d["entropy"] for d in data_list]
            method_names.append(method)
            entropies.append(vals)
            colors.append(METHOD_COLORS.get(method, "#888"))

        bp = ax.boxplot(entropies, patch_artist=True, widths=0.45,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75)

        for k, (vals, c) in enumerate(zip(entropies, colors)):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter([k + 1 + j for j in jitter], vals, color=c,
                       edgecolor="white", zorder=3, s=40, alpha=0.85)

        ax.set_xticks(range(1, len(method_names) + 1))
        ax.set_xticklabels(method_names, fontsize=10)
        ax.set_title(f"{model_name} — Saliency Entropy", fontsize=12, fontweight="bold")
        ax.set_ylabel("Shannon Entropy  (lower = more focused)")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Saliency Map Entropy Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "entropy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def plot_summary_radar(results, save_dir):
    """
    Radar chart comparing all methods on 4 normalised axes:
      Insertion AUC (↑), Deletion AUC (↓→ invert), AOPC (↑), Focus (1-entropy↑)
    """
    print("[Plot] Summary radar chart ...")
    from matplotlib.patches import FancyArrowPatch

    categories = ["Insertion\nAUC (↑)", "1 − Deletion\nAUC (↑)",
                  "AOPC (↑)", "Focus\n1−H (↑)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             subplot_kw={"projection": "polar"})

    for ax, model_name in zip(axes, ["ResNet-CNN", "ViT"]):
        all_scores = {m: {} for m in results[model_name]}
        for method, data_list in results[model_name].items():
            all_scores[method]["ins"]  = _avg(data_list, "insertion_auc")
            all_scores[method]["del"]  = _avg(data_list, "deletion_auc")
            all_scores[method]["aopc"] = _avg(data_list, "aopc")
            avg_ent = _avg(data_list, "entropy")
            all_scores[method]["focus"] = avg_ent

        ins_vals   = [all_scores[m]["ins"]   for m in all_scores]
        del_vals   = [all_scores[m]["del"]   for m in all_scores]
        aopc_vals  = [all_scores[m]["aopc"]  for m in all_scores]
        focus_vals = [all_scores[m]["focus"] for m in all_scores]

        def norm01(vals):
            mn, mx = min(vals), max(vals)
            if mx == mn: return [0.5] * len(vals)
            return [(v - mn) / (mx - mn) for v in vals]

        ins_n   = norm01(ins_vals)
        del_n   = [1 - v for v in norm01(del_vals)]   
        aopc_n  = norm01(aopc_vals)
        focus_n = [1 - v for v in norm01(focus_vals)] 
        for k, method in enumerate(all_scores):
            vals   = [ins_n[k], del_n[k], aopc_n[k], focus_n[k]]
            vals  += vals[:1]
            color  = METHOD_COLORS.get(method, "#888")
            ax.plot(angles, vals, "-o", color=color, linewidth=2, label=method, ms=5)
            ax.fill(angles, vals, color=color, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.set_title(model_name, fontsize=12, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(alpha=0.4)

    plt.suptitle("Explainability Method Comparison — Radar Chart",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def plot_summary_bar(results, save_dir):
    """Side-by-side grouped bar chart of all metrics."""
    print("[Plot] Summary bar chart ...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metric_cfg = [
        ("insertion_auc",  "Insertion AUC (↑ better)", True),
        ("deletion_auc",   "Deletion AUC (↓ better)",  False),
        ("aopc",           "AOPC (↑ better)",           True),
        ("entropy",        "Entropy (↓ better = focused)", False),
    ]

    for ax, (key, title, higher_better) in zip(axes.flat, metric_cfg):
        for model_name, model_results in results.items():
            methods = list(model_results.keys())
            vals    = [_avg(model_results[m], key) for m in methods]
            errs    = [np.std([d[key] for d in model_results[m]]) for m in methods]
            x       = np.arange(len(methods))
            offset  = -0.2 if model_name == "ResNet-CNN" else 0.2
            color   = "#E63946" if model_name == "ResNet-CNN" else "#457B9D"
            bars = ax.bar(x + offset, vals, 0.38, yerr=errs, capsize=4,
                          label=model_name, color=color, alpha=0.8, edgecolor="white",
                          error_kw={"elinewidth": 1.5})
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errs) * 0.5 + 0.002,
                        f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(np.arange(len(list(results["ResNet-CNN"].keys()))))
        ax.set_xticklabels(list(results["ResNet-CNN"].keys()), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        arrow = "↑" if higher_better else "↓"
        ax.set_xlabel(f"Method   ({arrow} = better)", fontsize=9)

    plt.suptitle("Quantitative Explainability Evaluation — All Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def print_and_save_summary(results, save_dir):
    summary = {}
    print(f"\n{'═'*75}")
    print("  EXPLAINABILITY EVALUATION — SUMMARY")
    print(f"{'═'*75}")
    hdr = f"  {'Model':<12} {'Method':<14} {'Ins AUC':>10} {'Del AUC':>10} {'AOPC':>10} {'Entropy':>10}"
    print(hdr)
    print(f"  {'-'*71}")

    for model_name, model_results in results.items():
        summary[model_name] = {}
        for method, data_list in model_results.items():
            ins  = _avg(data_list, "insertion_auc")
            dlt  = _avg(data_list, "deletion_auc")
            aop  = _avg(data_list, "aopc")
            ent  = _avg(data_list, "entropy")
            print(f"  {model_name:<12} {method:<14} {ins:>10.4f} {dlt:>10.4f} {aop:>10.4f} {ent:>10.4f}")
            summary[model_name][method] = {
                "insertion_auc": round(ins, 4),
                "deletion_auc" : round(dlt, 4),
                "aopc"         : round(aop, 4),
                "entropy"      : round(ent, 4),
            }

    print(f"\n  Interpretation guide:")
    print(f"  • Insertion AUC  — higher is better (fast confidence rise on reveal)")
    print(f"  • Deletion AUC   — lower is better  (fast confidence drop on removal)")
    print(f"  • AOPC           — higher is better (sharp drop when important px removed)")
    print(f"  • Entropy        — lower is better  (focused, peaked saliency)")
    print(f"{'═'*75}\n")

    with open(os.path.join(save_dir, "xai_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {os.path.join(save_dir, 'xai_summary.json')}")

def main():
    cnn_model, vit_model = load_models()

    samples = load_test_samples(DATA_ROOT, n_per_class=2)

    plot_qualitative_cnn(samples, cnn_model, OUT_DIR)
    plot_qualitative_vit(samples, vit_model, OUT_DIR)

    results = evaluate_methods(samples, cnn_model, vit_model)

    plot_insertion_deletion_curves(results, OUT_DIR)
    plot_aopc_curves(results, OUT_DIR)
    plot_entropy_comparison(results, OUT_DIR)
    plot_summary_bar(results, OUT_DIR)
    plot_summary_radar(results, OUT_DIR)

    print_and_save_summary(results, OUT_DIR)

    print(f"\nAll XAI outputs saved to: {OUT_DIR}/")

if __name__ == "__main__":
    main()