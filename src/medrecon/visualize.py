import os

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import for non-interactive use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_reconstruction_grid(model, dataset, device, out_path: str, n: int = 5):
    """5-column grid: Canny | Sobel | Laplacian | Prediction | Ground Truth"""
    model.eval()
    n = min(n, len(dataset))
    fig, axs = plt.subplots(n, 5, figsize=(18, 3.5 * n))
    if n == 1:
        axs = axs[np.newaxis, :]

    titles = ["Canny", "Sobel", "Laplacian", "Prediction", "Ground Truth"]
    for col, title in enumerate(titles):
        axs[0, col].set_title(title, fontsize=12)

    with torch.no_grad():
        for i in range(n):
            feat, orig = dataset[i]
            pred = model(feat.unsqueeze(0).to(device))[0, 0].cpu().numpy()

            for ch in range(3):
                axs[i, ch].imshow(feat[ch].numpy(), cmap="gray")
                axs[i, ch].axis("off")
            axs[i, 3].imshow(pred, cmap="gray")
            axs[i, 3].axis("off")
            axs[i, 4].imshow(orig[0].numpy(), cmap="gray")
            axs[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Grid saved: {out_path}")


def plot_training_curves(csv_path: str, out_path: str):
    """Two-panel figure: loss curves + PSNR/SSIM curves over epochs."""
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df["epoch"], df["train_loss"], label="train")
    ax1.plot(df["epoch"], df["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (L1)")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(df["epoch"], df["val_psnr"], label="PSNR (dB)", color="blue")
    ax2b = ax2.twinx()
    ax2b.plot(df["epoch"], df["val_ssim"], label="SSIM", color="orange", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2b.set_ylabel("SSIM")
    ax2.set_title("Val Metrics")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Curves saved: {out_path}")


def plot_modality_comparison(results: dict, out_path: str):
    """
    results = {run_name: {"psnr": float, "ssim": float}, ...}
    Bar chart comparing final val PSNR and SSIM across all runs.
    """
    names = list(results.keys())
    psnrs = [results[n]["psnr"] for n in names]
    ssims = [results[n]["ssim"] for n in names]

    x = np.arange(len(names))
    colors = ["steelblue" if "eta00" in n else "darkorange" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x, psnrs, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Final Val PSNR by Run")

    ax2.bar(x, ssims, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right")
    ax2.set_ylabel("SSIM")
    ax2.set_title("Final Val SSIM by Run")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved: {out_path}")
