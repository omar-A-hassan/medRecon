import os
from dataclasses import dataclass, field

# Resolved relative to this file: src/medrecon/config.py -> ../../data
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


@dataclass
class ExperimentConfig:
    modality: str               # "xray" | "ct" | "mri"
    eta: float = 0.0            # SSD mask ratio — 0.0 = standard baseline
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    img_size: int = 256
    n_train: int = 500          # max images for train split
    n_val: int = 100            # max images for val split
    canny_low: int = 50
    canny_high: int = 150
    seed: int = 42
    base_dir: str = field(default_factory=lambda: _DATA_DIR)
    results_dir: str = field(default_factory=lambda: _RESULTS_DIR)

    @property
    def run_name(self) -> str:
        eta_str = str(self.eta).replace(".", "")
        return f"{self.modality}_eta{eta_str}"
