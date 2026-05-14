"""Generate supplementary figures for the LaTeX report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

# ── 1. UNet architecture diagram ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

def box(ax, x, y, w, h, label, sub="", color="#4C72B0", fontsize=8, sublabel_size=6.5):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor="black",
                           facecolor=color, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sub else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.22, sub,
                ha="center", va="center", fontsize=sublabel_size,
                color="white", style="italic", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="gray", lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=5)

def skip(ax, x1, y1, x2, y2):
    """Draw a skip connection as a curved arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#E87722", lw=1.6,
                                connectionstyle="arc3,rad=-0.35"), zorder=5)

COLORS = {
    "input": "#2ca02c",
    "enc": "#4C72B0",
    "bridge": "#d62728",
    "dec": "#9467bd",
    "output": "#2ca02c",
    "concat": "#8c564b",
}

# --- Input ---
box(ax, 0.2, 3.8, 1.3, 1.4, "Input", "3×256×256\n(Canny+Sobel\n+Laplacian)", color=COLORS["input"], fontsize=7.5, sublabel_size=6)

# --- Encoder stages ---
enc_data = [
    (2.0, "E1", "64\n128×128"),
    (3.6, "E2", "64\n128×128"),
    (5.2, "E3", "128\n64×64"),
    (6.8, "E4", "256\n32×32"),
    (8.4, "E5", "512\n16×16"),
]
for x, name, ch in enc_data:
    box(ax, x, 4.05, 1.2, 0.9, name, ch, color=COLORS["enc"], fontsize=8.5, sublabel_size=6.5)

# arrow: input → E1
arrow(ax, 1.5, 4.5, 2.0, 4.5)
# arrows enc → enc
for i in range(len(enc_data) - 1):
    x1 = enc_data[i][0] + 1.2
    x2 = enc_data[i+1][0]
    arrow(ax, x1, 4.5, x2, 4.5)

# --- Bridge ---
box(ax, 10.0, 3.8, 1.3, 1.4, "Bridge", "512\n16×16", color=COLORS["bridge"], fontsize=8, sublabel_size=6.5)
arrow(ax, enc_data[-1][0] + 1.2, 4.5, 10.0, 4.5)

# --- Decoder stages (right to left in x, but drawn going right) ---
dec_data = [
    (11.6, "D4", "256\n32×32"),
    (13.0, "D3", "128\n64×64"),
    # We'll place D2 and D1 below to simulate the architecture
]
# Actually, let's lay decoder below the encoder to create a U shape
# Encoder top row at y=4.05, decoder bottom row at y=2.2

dec_bottom = [
    (8.4, "D4", "256\n32×32"),
    (6.8, "D3", "128\n64×64"),
    (5.2, "D2", "64\n128×128"),
    (3.6, "D1", "32\n256×256"),
]
for x, name, ch in dec_bottom:
    box(ax, x, 2.05, 1.2, 0.9, name, ch, color=COLORS["dec"], fontsize=8.5, sublabel_size=6.5)

# Arrow: Bridge → D4 (down then left)
arrow(ax, 10.65, 3.8, 9.0, 2.55)

# Arrows D4 → D3 → D2 → D1
for i in range(len(dec_bottom) - 1):
    x1 = dec_bottom[i][0]
    x2 = dec_bottom[i+1][0] + 1.2
    arrow(ax, x1, 2.5, x2, 2.5)

# Skip connections (from encoder to decoder, same spatial resolution)
# E2→D1 (both 128×128), E3→D2 (64×64), E4→D3 (32×32), E5→D4 (16×16)
skip_pairs = [
    (enc_data[1][0]+0.6, 4.05, dec_bottom[3][0]+0.6, 2.95),  # E2→D1
    (enc_data[2][0]+0.6, 4.05, dec_bottom[2][0]+0.6, 2.95),  # E3→D2
    (enc_data[3][0]+0.6, 4.05, dec_bottom[1][0]+0.6, 2.95),  # E4→D3
    (enc_data[4][0]+0.6, 4.05, dec_bottom[0][0]+0.6, 2.95),  # E5→D4
]
for x1, y1, x2, y2 in skip_pairs:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#E87722", lw=1.5,
                                linestyle="dashed"), zorder=5)

# --- Head / Output ---
box(ax, 2.0, 2.05, 1.2, 0.9, "Head", "Conv 1×1\n+Sigmoid", color="#17becf", fontsize=7.5, sublabel_size=6.5)
arrow(ax, dec_bottom[3][0], 2.5, 3.2, 2.5)

box(ax, 0.2, 2.05, 1.3, 0.9, "Output", "1×256×256", color=COLORS["output"], fontsize=8, sublabel_size=6.5)
arrow(ax, 2.0, 2.5, 1.5, 2.5)

# Labels
ax.text(5.7, 5.25, "ResNet34 Encoder", ha="center", fontsize=10, fontweight="bold", color=COLORS["enc"])
ax.text(6.1, 1.65, "UNet Decoder", ha="center", fontsize=10, fontweight="bold", color=COLORS["dec"])
ax.text(10.65, 5.5, "Bottleneck", ha="center", fontsize=9, fontweight="bold", color=COLORS["bridge"])

# Skip legend
skip_patch = mpatches.Patch(color="#E87722", label="Skip connections (concatenation)")
enc_patch = mpatches.Patch(color=COLORS["enc"], label="Encoder block (ResNet34 stage)")
dec_patch = mpatches.Patch(color=COLORS["dec"], label="Decoder block (upsample + conv)")
ax.legend(handles=[enc_patch, dec_patch, skip_patch],
          loc="upper right", fontsize=8, framealpha=0.9)

ax.set_title("ResNet34 U-Net Architecture for Medical Image Reconstruction",
             fontsize=13, fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig("/home/user/medRecon/report/figures/unet_architecture.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved unet_architecture.png")


# ── 2. Training curves comparison: all 6 runs (PSNR) ─────────────────────────

runs = {
    "X-ray Baseline": ("/home/user/medRecon/results/xray_eta00/metrics.csv", "#4C72B0", "-"),
    "X-ray SSD": ("/home/user/medRecon/results/xray_eta08/metrics.csv", "#4C72B0", "--"),
    "CT Baseline": ("/home/user/medRecon/results/ct_eta00/metrics.csv", "#E87722", "-"),
    "CT SSD": ("/home/user/medRecon/results/ct_eta08/metrics.csv", "#E87722", "--"),
    "MRI Baseline": ("/home/user/medRecon/results/mri_eta00/metrics.csv", "#2ca02c", "-"),
    "MRI SSD": ("/home/user/medRecon/results/mri_eta08/metrics.csv", "#2ca02c", "--"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for label, (path, color, ls) in runs.items():
    df = pd.read_csv(path)
    axes[0].plot(df["epoch"], df["val_psnr"], color=color, linestyle=ls, lw=1.8, label=label)
    axes[1].plot(df["epoch"], df["val_ssim"], color=color, linestyle=ls, lw=1.8, label=label)

axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Validation PSNR (dB)", fontsize=11)
axes[0].set_title("Validation PSNR — All Runs", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=8.5, loc="lower right")
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Validation SSIM", fontsize=11)
axes[1].set_title("Validation SSIM — All Runs", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8.5, loc="lower right")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Training Curves: Baseline vs. SSD ($\\eta=0.8$) across Modalities",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("/home/user/medRecon/report/figures/all_curves.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved all_curves.png")


# ── 3. SSD masking illustration ───────────────────────────────────────────────

rng = np.random.default_rng(7)
img = rng.uniform(0, 1, (12, 12))

eta = 0.8
mask = rng.binomial(1, 1 - eta, img.shape).astype(float)
masked = img.copy()
masked[mask == 0] = np.nan  # masked-out pixels shown as gray

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
kw = dict(vmin=0, vmax=1, cmap="gray", interpolation="nearest")

im0 = axes[0].imshow(img, **kw)
axes[0].set_title("Target image\n(all pixels)", fontsize=10, fontweight="bold")
axes[0].axis("off")

masked_disp = np.ma.array(img, mask=(mask == 0))
cmap_m = plt.cm.gray.copy()
cmap_m.set_bad(color="#f28e2b", alpha=0.6)
axes[1].imshow(masked_disp, cmap=cmap_m, vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title(f"Bernoulli mask\n($\\eta=0.8$, 20\\% supervised)", fontsize=10, fontweight="bold")
axes[1].axis("off")

# Show loss region
loss_vis = np.zeros_like(img)
loss_vis[mask == 1] = img[mask == 1]
im2 = axes[2].imshow(loss_vis, **kw)
axes[2].set_title("Loss computed here\n(unmasked pixels only)", fontsize=10, fontweight="bold")
axes[2].axis("off")

orange_patch = mpatches.Patch(color="#f28e2b", alpha=0.6, label="Masked pixels (no gradient)")
axes[1].legend(handles=[orange_patch], loc="lower center", fontsize=8,
               bbox_to_anchor=(0.5, -0.22), framealpha=0.9)

fig.suptitle("Sparsely Supervised Loss: Bernoulli Pixel Masking Illustration",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/user/medRecon/report/figures/ssd_masking.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved ssd_masking.png")

print("\nAll figures generated successfully.")
