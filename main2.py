"""
COVID-19 Chest X-Ray Image Classification
==========================================
Models : ResNet-CNN  +  Vision Transformer (ViT)
Classes: COVID | Normal | Viral Pneumonia

GPU/CPU strategy
----------------
- Automatically detects CUDA; falls back to CPU with zero code changes.
- pin_memory + non_blocking transfers for fast GPU data loading.
- Mixed-precision (AMP) enabled on CUDA, silently skipped on CPU.
- num_workers scales to 4 on GPU, 2 on CPU.
- DataParallel wraps the model when multiple GPUs are present.

Dataset layout expected
-----------------------
COVID_19_dataset/
  train/  COVID/  Normal/  Viral Pneumonia/
  val/    COVID/  Normal/  Viral Pneumonia/
  test/   COVID/  Normal/  Viral Pneumonia/
"""

import os, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def setup_device():
    """
    Detects and configures the best available device.
    Returns device, and flags for AMP and multi-GPU.
    """
    if torch.cuda.is_available():
        device      = torch.device("cuda")
        n_gpus      = torch.cuda.device_count()
        gpu_name    = torch.cuda.get_device_name(0)
        use_amp     = True          # mixed precision → faster GPU training
        num_workers = 4             # more workers with fast GPU I/O
        pin_memory  = True          # page-locked RAM → faster host→GPU copy
        print(f"[Device] GPU detected: {gpu_name}  ({n_gpus} GPU(s))")
        print(f"         Mixed-precision AMP : ENABLED")
        print(f"         DataParallel        : {'YES' if n_gpus > 1 else 'NO (single GPU)'}")
    else:
        device      = torch.device("cpu")
        n_gpus      = 0
        use_amp     = False         # AMP not supported on CPU
        num_workers = 2
        pin_memory  = False         # pin_memory has no benefit on CPU
        print(f"[Device] No GPU found — running on CPU")
        print(f"         Mixed-precision AMP : DISABLED")

    print(f"         num_workers         : {num_workers}")
    print(f"         pin_memory          : {pin_memory}\n")
    return device, n_gpus, use_amp, num_workers, pin_memory

DEVICE, N_GPUS, USE_AMP, NUM_WORKERS, PIN_MEMORY = setup_device()

#hyper parameters

DATA_ROOT    = "COVID_19_dataset"
IMG_SIZE     = 224          # full resolution for real X-rays
BATCH_SIZE   = 32 if DEVICE.type == "cpu" else 64   # larger batches on GPU
NUM_EPOCHS   = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
OUT_DIR      = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True   # auto-tune conv algorithms for speed

print(f"[Config] Image size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"         Batch size  : {BATCH_SIZE}")
print(f"         Epochs      : {NUM_EPOCHS}")
print(f"         Output dir  : {OUT_DIR}\n")

#data pipelines
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_data(root):
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  eval_tf)

    # pin_memory=True speeds up CPU→GPU transfer when using CUDA
    kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    print(f"[Data] Classes : {train_ds.classes}")
    print(f"       Train   : {len(train_ds)} images")
    print(f"       Val     : {len(val_ds)}   images")
    print(f"       Test    : {len(test_ds)}  images\n")
    return train_loader, val_loader, test_loader, train_ds.classes

# resnet cnn
class ResBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        self.layer1 = nn.Sequential(ResBlock(32,  64),        ResBlock(64,  64))
        self.layer2 = nn.Sequential(ResBlock(64,  128, 2),    ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2),    ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2),    ResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))

# vit
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)

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

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        return x + self.mlp(self.norm2(x))

class ViT(nn.Module):
    def __init__(self, num_classes=3, img_size=224, patch_size=16,
                 embed_dim=256, depth=8, heads=8):
        super().__init__()
        n_patches          = (img_size // patch_size) ** 2
        self.patch_embed   = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed     = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks        = nn.Sequential(
            *[TransformerBlock(embed_dim, heads) for _ in range(depth)]
        )
        self.norm          = nn.LayerNorm(embed_dim)
        self.head          = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        B   = x.size(0)
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.norm(self.blocks(x))
        return self.head(x[:, 0])   # classification token

# wrapper
def prepare_model(model):
    """
    Move model to device.
    Wrap with DataParallel automatically when multiple GPUs exist.
    """
    model = model.to(DEVICE)
    if N_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"  → DataParallel across {N_GPUS} GPUs")
    return model

# training
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train_model(model, name, train_loader, val_loader):
    model = prepare_model(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_wts = 0.0, None

    print(f"\n{'='*65}")
    print(f"  Training: {name}   |   Device: {DEVICE}")
    print(f"{'='*65}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"  Ep {epoch:02d}/{NUM_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {vl_loss:.4f}/{vl_acc:.4f} | "
              f"LR {lr_now:.2e} | {elapsed:.1f}s")

    print(f"\n  Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history

#m metrics
def compute_metrics(y_true, y_pred, class_names):
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall"   : recall_score(y_true,   y_pred, average="weighted", zero_division=0),
        "f1"       : f1_score(y_true,       y_pred, average="weighted", zero_division=0),
        "report"   : classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "cm"       : confusion_matrix(y_true, y_pred),
    }

# plots
def plot_training_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ResNet-CNN": "#E63946", "ViT": "#457B9D"}
    for name, h in histories.items():
        c = colors[name]; ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "--", color=c, alpha=0.55, label=f"{name} Train")
        axes[0].plot(ep, h["val_loss"],   "-",  color=c,             label=f"{name} Val")
        axes[1].plot(ep, h["train_acc"],  "--", color=c, alpha=0.55, label=f"{name} Train")
        axes[1].plot(ep, h["val_acc"],    "-",  color=c,             label=f"{name} Val")
    for ax, ttl in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(ttl, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ttl); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f"Training Curves  [{DEVICE.type.upper()}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_confusion_matrices(cms, names, class_names, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(7 * len(cms), 6))
    if len(cms) == 1: axes = [axes]
    for ax, cm, name in zip(axes, cms, names):
        norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{name} — Confusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_metrics_bar(results, path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x, width = np.arange(len(metrics)), 0.35
    colors   = ["#E63946", "#457B9D"]
    fig, ax  = plt.subplots(figsize=(10, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=name, color=colors[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12); ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=12)
    ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Model Comparison — Test Set  [{DEVICE.type.upper()}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_per_class_f1(results, class_names, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5), sharey=True)
    if len(results) == 1: axes = [axes]
    cmap = {"COVID": "#E63946", "Normal": "#2A9D8F", "Viral Pneumonia": "#E9C46A"}
    for ax, (name, res) in zip(axes, results.items()):
        lines = res["report"].strip().split("\n")
        f1s, lbls = [], []
        for line in lines[2:2+len(class_names)]:
            parts = line.split()
            if len(parts) >= 5:
                lbls.append(" ".join(parts[:-4]))
                f1s.append(float(parts[-3]))
        bars = ax.bar(lbls, f1s, color=[cmap.get(l, "#aaa") for l in lbls],
                      edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15); ax.set_title(f"{name}\nPer-Class F1", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

# main function
def main():
    if not os.path.isdir(DATA_ROOT):
        create_synthetic_dataset(DATA_ROOT)

    train_loader, val_loader, test_loader, class_names = load_data(DATA_ROOT)
    nc = len(class_names)

    model_defs = {
        "ResNet-CNN": ResNetCNN(nc),
        "ViT":        ViT(nc, img_size=IMG_SIZE),
    }

    histories, results, cms = {}, {}, []

    for name, raw_model in model_defs.items():
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"\n[Model] {name}  |  Trainable params: {n_params:,}")

        trained, history = train_model(raw_model, name, train_loader, val_loader)
        histories[name]  = history

        # ── Test 
        print(f"\n[Eval] {name} on test set ...")
        _, _, y_pred, y_true = evaluate(trained, test_loader, nn.CrossEntropyLoss())
        m = compute_metrics(y_true, y_pred, class_names)
        results[name] = m
        cms.append(m["cm"])

        print(f"\n  ── {name} Test Results ──")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1-Score : {m['f1']:.4f}")
        print(f"\n{m['report']}")

        save_path = os.path.join(OUT_DIR, f"{name.replace('-','_')}_weights.pth")
        state = trained.module.state_dict() if hasattr(trained, "module") else trained.state_dict()
        torch.save(state, save_path)
        print(f"  Weights saved → {save_path}")

    print("\n[Plots] Generating ...")
    plot_training_curves(histories,
        os.path.join(OUT_DIR, "training_curves.png"))
    plot_confusion_matrices(cms, list(results.keys()), class_names,
        os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_metrics_bar(results,
        os.path.join(OUT_DIR, "metrics_comparison.png"))
    plot_per_class_f1(results, class_names,
        os.path.join(OUT_DIR, "per_class_f1.png"))

    summary = {
        name: {k: (float(v) if isinstance(v, (float, np.floating)) else str(v))
               for k, v in res.items() if k != "cm"}
        for name, res in results.items()
    }
    summary["_meta"] = {"device": str(DEVICE), "amp": USE_AMP, "epochs": NUM_EPOCHS}
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS  [{DEVICE.type.upper()}]")
    print(f"{'='*65}")
    print(f"  {'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*57}")
    for name, res in results.items():
        print(f"  {name:<14} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>10.4f} {res['f1']:>10.4f}")
    print(f"{'='*65}")
    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()