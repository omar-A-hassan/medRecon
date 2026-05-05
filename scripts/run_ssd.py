"""
Experiment 2: SSD eta sweep on X-ray, then best eta on CT and MRI.

Usage:
    # Full sweep (default):
    uv run scripts/run_ssd.py

    # Single eta, X-ray only:
    uv run scripts/run_ssd.py --eta 0.5 --modalities xray --no-phase2

    # Single eta, all modalities:
    uv run scripts/run_ssd.py --eta 0.5 --modalities xray ct mri --no-phase2

    # Custom sweep:
    uv run scripts/run_ssd.py --eta 0.5 0.8 --modalities xray

    # Override epochs or dataset size:
    uv run scripts/run_ssd.py --eta 0.5 --modalities xray --epochs 100 --n-train 1000
"""
import argparse
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

DEFAULT_ETA_SWEEP = [0.0, 0.5, 0.8, 0.95]
DEFAULT_MODALITIES = ["xray"]  # phase 2 (ct, mri) runs automatically when sweep > 1 eta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, nargs="+", default=None,
                   help="Eta value(s) to train. Default: full sweep [0.0, 0.5, 0.8, 0.95]")
    p.add_argument("--modalities", nargs="+", default=None,
                   choices=["xray", "ct", "mri"],
                   help="Modalities to train on. Default: xray (+ ct/mri auto phase 2 "
                        "when running a sweep)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--no-phase2", action="store_true",
                   help="Skip the automatic ct/mri phase-2 run after a sweep")
    return p.parse_args()


def run_one(modality, eta, epochs, n_train, n_val):
    config = ExperimentConfig(modality=modality, eta=eta, epochs=epochs,
                              n_train=n_train, n_val=n_val)
    train_loader, val_loader = get_dataloaders(config)
    model = build_model(config)
    trainer = Trainer(model, config, train_loader, val_loader)
    out_dir = trainer.train()

    device = _get_device()
    model.load_state_dict(torch.load(f"{out_dir}/best_model.pth", map_location=device))
    plot_reconstruction_grid(model, val_loader.dataset, device, f"{out_dir}/grid.png")
    plot_training_curves(f"{out_dir}/metrics.csv", f"{out_dir}/curves.png")

    df = pd.read_csv(f"{out_dir}/metrics.csv")
    return config, out_dir, {
        "psnr": df["val_psnr"].iloc[-1],
        "ssim": df["val_ssim"].iloc[-1],
    }


def main():
    args = parse_args()
    eta_sweep = args.eta if args.eta is not None else DEFAULT_ETA_SWEEP
    modalities = args.modalities if args.modalities is not None else DEFAULT_MODALITIES
    run_sweep = len(eta_sweep) > 1
    results_summary = {}
    last_config = None

    print(f"eta={eta_sweep}  modalities={modalities}  epochs={args.epochs}")

    # --- Phase 1: requested etas x requested modalities ---
    for modality in modalities:
        for eta in eta_sweep:
            print(f"\n{'='*50}\nRunning: modality={modality}  eta={eta}\n{'='*50}")
            config, _, metrics = run_one(modality, eta, args.epochs, args.n_train, args.n_val)
            results_summary[config.run_name] = metrics
            last_config = config

    # --- Phase 2: best eta on ct + mri (only when sweeping xray and not suppressed) ---
    if run_sweep and not args.no_phase2 and modalities == ["xray"]:
        xray_runs = {k: v for k, v in results_summary.items() if k.startswith("xray")}
        best_run = max(xray_runs, key=lambda k: xray_runs[k]["psnr"])
        eta_map = {f"xray_eta{str(e).replace('.', '')}": e for e in eta_sweep}
        best_eta = eta_map[best_run]
        print(f"\nBest eta from X-ray sweep: {best_eta} (run: {best_run})")
        print(f"Phase 2: eta={best_eta} on CT and MRI")
        for modality in ["ct", "mri"]:
            print(f"\n{'='*50}\nRunning: modality={modality}  eta={best_eta}\n{'='*50}")
            config, _, metrics = run_one(modality, best_eta, args.epochs, args.n_train, args.n_val)
            results_summary[config.run_name] = metrics
            last_config = config

    results_dir = last_config.results_dir
    plot_modality_comparison(results_summary, f"{results_dir}/ssd_comparison.png")
    print("\nSSD experiments complete.")


if __name__ == "__main__":
    main()
