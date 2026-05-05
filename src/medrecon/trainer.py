import csv
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .config import ExperimentConfig
from .losses import ssd_loss
from .metrics import compute_metrics


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(self, model, config: ExperimentConfig, train_loader, val_loader):
        self.config = config
        self.device = _get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.best_val_loss = float("inf")

        self.out_dir = os.path.join(config.results_dir, config.run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, "metrics.csv")
        self._csv_init()

        print(f"[Trainer] device={self.device}  run={config.run_name}")

    def _csv_init(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_psnr", "val_ssim"])

    def _csv_append(self, row: list):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = ssd_loss(outputs, targets, self.config.eta)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _val_epoch(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = ssd_loss(outputs, targets, self.config.eta)
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        preds_np = np.concatenate(all_preds, axis=0)
        tgts_np = np.concatenate(all_targets, axis=0)
        m = compute_metrics(preds_np, tgts_np)
        return total_loss / len(self.val_loader), m

    def train(self) -> str:
        for epoch in tqdm(range(1, self.config.epochs + 1), desc=self.config.run_name):
            train_loss = self._train_epoch()
            val_loss, m = self._val_epoch()

            self._csv_append([epoch, train_loss, val_loss, m["mae"], m["psnr"], m["ssim"]])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt = os.path.join(self.out_dir, "best_model.pth")
                torch.save(self.model.state_dict(), ckpt)

            if epoch % 10 == 0 or epoch == 1:
                tqdm.write(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"PSNR={m['psnr']:.2f} | SSIM={m['ssim']:.4f}"
                )

        print(f"Done. Results: {self.out_dir}")
        return self.out_dir
