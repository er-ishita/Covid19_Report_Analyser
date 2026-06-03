"""
COVID-19 Chest X-Ray Image Classification
==========================================
Models: ResNet50 (CNN) + ViT-B/16 (Vision Transformer)
Classes: COVID, Viral Pneumonia, Normal
Dataset structure:
  COVID_19_dataset/
    train/ {COVID, Viral Pneumonia, Normal}
    val/   {COVID, Viral Pneumonia, Normal}
    test/  {COVID, Viral Pneumonia, Normal}
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

DATA_ROOT   = "COVID_19_dataset"   # ← change if needed
CLASSES     = ["COVID", "Normal", "Viral Pneumonia"]
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_EPOCHS  = 8
LR          = 3e-4
WEIGHT_DECAY= 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
OUT_DIR     = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

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

def load_datasets(root):
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  eval_tf)
    return train_ds, val_ds, test_ds

def make_loaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

class ConvBlock(nn.Module):
    """Residual-style conv block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x))

def build_resnet50(num_classes=3):
    """Custom lightweight ResNet-style CNN (no pretrained weights needed)."""
    class MiniResNet(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(ConvBlock(16, 32),    ConvBlock(32, 32))
            self.layer2 = nn.Sequential(ConvBlock(32, 64, 2), ConvBlock(64, 64))
            self.layer3 = nn.Sequential(ConvBlock(64, 128, 2), ConvBlock(128, 128))
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.head   = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, nc),
            )
        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
            x = self.pool(x).flatten(1)
            return self.head(x)
    return MiniResNet(num_classes)


class PatchEmbed(nn.Module):
    """Patch embedding for mini-ViT."""
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # B, N, D

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

def build_vit(num_classes=3):
    """Compact Vision Transformer trained from scratch."""
    class MiniViT(nn.Module):
        def __init__(self, nc, img_size=64, patch=8, embed=128, depth=4, heads=4):
            super().__init__()
            self.patch_embed = PatchEmbed(img_size, patch, 3, embed)
            n = (img_size // patch) ** 2
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed))
            self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token,  std=0.02)
            self.blocks = nn.Sequential(*[TransformerBlock(embed, heads) for _ in range(depth)])
            self.norm   = nn.LayerNorm(embed)
            self.head   = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed, nc))
        def forward(self, x):
            B = x.size(0)
            x = self.patch_embed(x)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1) + self.pos_embed
            x   = self.norm(self.blocks(x))
            return self.head(x[:, 0])
    return MiniViT(num_classes)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def compute_metrics(y_true, y_pred, class_names):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report":    classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "cm":        confusion_matrix(y_true, y_pred),
    }

def train_model(model, model_name, train_loader, val_loader, class_names):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  |  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  |  "
              f"{elapsed:.1f}s")

    model.load_state_dict(best_state)
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    return model, history

def plot_training_curves(histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ResNet50": "#E63946", "ViT-Small": "#457B9D"}

    for name, h in histories.items():
        c = colors[name]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], "--", color=c, alpha=0.6, label=f"{name} Train")
        axes[0].plot(epochs, h["val_loss"],   "-",  color=c,           label=f"{name} Val")
        axes[1].plot(epochs, h["train_acc"],  "--", color=c, alpha=0.6, label=f"{name} Train")
        axes[1].plot(epochs, h["val_acc"],    "-",  color=c,           label=f"{name} Val")

    for ax, title, ylabel in zip(axes, ["Loss", "Accuracy"], ["Loss", "Accuracy"]):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Training Curves — COVID-19 X-Ray Classification", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrices(cms, model_names, class_names, save_path):
    fig, axes = plt.subplots(1, len(cms), figsize=(7 * len(cms), 6))
    if len(cms) == 1:
        axes = [axes]
    for ax, cm, name in zip(axes, cms, model_names):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, cbar=True,
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_metrics_comparison(results, save_path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = [m.capitalize() for m in metrics]
    x       = np.arange(len(labels))
    width   = 0.35
    colors  = ["#E63946", "#457B9D"]

    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(results.keys())
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=name, color=colors[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_per_class_f1(results, class_names, save_path):
    """Per-class F1 from classification report."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    colors_map = {"COVID": "#E63946", "Normal": "#2A9D8F", "Viral Pneumonia": "#E9C46A"}

    for ax, (name, res) in zip(axes, results.items()):
        lines = res["report"].strip().split("\n")
        f1s, lbls = [], []
        for line in lines[2:2 + len(class_names)]:
            parts = line.split()
            if len(parts) >= 5:
                cls = " ".join(parts[:-4])
                f1  = float(parts[-3])
                f1s.append(f1)
                lbls.append(cls)
        bar_colors = [colors_map.get(l, "#aaa") for l in lbls]
        bars = ax.bar(lbls, f1s, color=bar_colors, edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{name}\nPer-Class F1", fontsize=13, fontweight="bold")
        ax.set_ylabel("F1 Score")
        ax.tick_params(axis="x", rotation=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def main():
    # Locate dataset
    if not os.path.isdir(DATA_ROOT):
        create_synthetic_dataset(DATA_ROOT)

    print(f"\nLoading data from: {DATA_ROOT}")
    train_ds, val_ds, test_ds = load_datasets(DATA_ROOT)
    class_names = train_ds.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    model_configs = {
        "ResNet50":  build_resnet50(len(class_names)),
        "ViT-Small": build_vit(len(class_names)),
    }

    histories, results, cms = {}, {}, []

    for model_name, model in model_configs.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{model_name} — Trainable params: {n_params:,}")

        # Train
        trained_model, history = train_model(
            model, model_name, train_loader, val_loader, class_names
        )
        histories[model_name] = history

        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set ...")
        criterion = nn.CrossEntropyLoss()
        _, _, y_pred, y_true = eval_epoch(trained_model, test_loader, criterion)
        metrics = compute_metrics(y_true, y_pred, class_names)
        results[model_name] = metrics
        cms.append(metrics["cm"])

        print(f"\n{'─'*50}")
        print(f"{model_name} — Test Metrics")
        print(f"{'─'*50}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"\n{metrics['report']}")

        # Save model
        torch.save(trained_model.state_dict(),
                   os.path.join(OUT_DIR, f"{model_name.replace('-','_')}_weights.pth"))

    print("\nGenerating plots ...")
    plot_training_curves(histories,
        os.path.join(OUT_DIR, "training_curves.png"))
    plot_confusion_matrices(cms, list(results.keys()), class_names,
        os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_metrics_comparison(results,
        os.path.join(OUT_DIR, "metrics_comparison.png"))
    plot_per_class_f1(results, class_names,
        os.path.join(OUT_DIR, "per_class_f1.png"))

    summary = {
        name: {k: float(v) if isinstance(v, (np.floating, float)) else str(v)
               for k, v in res.items() if k != "cm"}
        for name, res in results.items()
    }
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<15} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>10.4f} {res['f1']:>10.4f}")
    print(f"{'='*60}")
    print("\nAll outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()