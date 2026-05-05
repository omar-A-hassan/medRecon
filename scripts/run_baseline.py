"""Experiment 1: standard L1 baseline (eta=0) on all 3 modalities."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import torch

from medrecon.config import ExperimentConfig
from medrecon.loaders import get_dataloaders
from medrecon.model import build_model
from medrecon.trainer import Trainer, _get_device
from medrecon.visualize import (
    plot_modality_comparison,
    plot_reconstruction_grid,
    plot_training_curves,
)

MODALITIES = ["xray", "ct", "mri"]
results_summary = {}

for modality in MODALITIES:
    # skip if already completed (checkpoint exists and CSV has at least 1 data row)
    _cfg_check = ExperimentConfig(modality=modality, eta=0.0)
    _ckpt = os.path.join(_cfg_check.results_dir, _cfg_check.run_name, "best_model.pth")
    _csv  = os.path.join(_cfg_check.results_dir, _cfg_check.run_name, "metrics.csv")
    if os.path.exists(_ckpt) and os.path.exists(_csv):
        df = pd.read_csv(_csv)
        if len(df) > 0:
            print(f"\nSkipping {modality} — results already exist (epoch {int(df['epoch'].iloc[-1])}, PSNR={df['val_psnr'].iloc[-1]:.2f})")
            results_summary[_cfg_check.run_name] = {
                "psnr": df["val_psnr"].iloc[-1],
                "ssim": df["val_ssim"].iloc[-1],
            }
            continue
    print(f"\n{'='*50}\nRunning baseline — modality: {modality}\n{'='*50}")
    config = ExperimentConfig(modality=modality, eta=0.0, epochs=50, n_train=500, n_val=100)
    train_loader, val_loader = get_dataloaders(config)
    model = build_model(config)
    trainer = Trainer(model, config, train_loader, val_loader)
    out_dir = trainer.train()

    device = _get_device()
    model.load_state_dict(torch.load(f"{out_dir}/best_model.pth", map_location=device))
    plot_reconstruction_grid(model, val_loader.dataset, device, f"{out_dir}/grid.png")
    plot_training_curves(f"{out_dir}/metrics.csv", f"{out_dir}/curves.png")

    df = pd.read_csv(f"{out_dir}/metrics.csv")
    results_summary[config.run_name] = {
        "psnr": df["val_psnr"].iloc[-1],
        "ssim": df["val_ssim"].iloc[-1],
    }

base_results_dir = config.results_dir
plot_modality_comparison(results_summary, f"{base_results_dir}/baseline_comparison.png")
print("\nBaseline experiments complete.")
