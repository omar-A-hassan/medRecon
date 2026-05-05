import glob
import os
import random
from torch.utils.data import DataLoader
from .config import ExperimentConfig
from .dataset import MedicalDataset


def collect_paths(modality: str, base_dir: str) -> list[str]:
    """Return all valid image paths for the given modality."""
    if modality == "xray":
        patterns = [
            os.path.join(base_dir, "COVID-19_Radiography_Dataset", "COVID", "images", "*.png"),
            os.path.join(base_dir, "COVID-19_Radiography_Dataset", "Normal", "images", "*.png"),
            os.path.join(base_dir, "COVID-19_Radiography_Dataset", "Lung_Opacity", "images", "*.png"),
        ]
    elif modality == "ct":
        patterns = [
            os.path.join(base_dir, "Chest_CT_Scan", "train", "**", "*.png"),
            os.path.join(base_dir, "Chest_CT_Scan", "valid", "**", "*.png"),
        ]
    elif modality == "mri":
        patterns = [
            os.path.join(base_dir, "brain_tumor_dataset", "no", "*.jpeg"),
            os.path.join(base_dir, "brain_tumor_dataset", "no", "*.jpg"),
            os.path.join(base_dir, "brain_tumor_dataset", "yes", "*.jpg"),
            os.path.join(base_dir, "brain_tumor_dataset", "yes", "*.JPG"),
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    paths = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    return [p for p in paths if os.path.isfile(p)]


def get_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    random.seed(config.seed)
    all_paths = collect_paths(config.modality, config.base_dir)
    random.shuffle(all_paths)

    total = config.n_train + config.n_val
    if len(all_paths) >= total:
        selected = all_paths[:total]
        train_paths = selected[:config.n_train]
        val_paths = selected[config.n_train:]
    else:
        # dataset smaller than requested — use 80/20 split on all available
        split = max(1, int(len(all_paths) * 0.8))
        train_paths = all_paths[:split]
        val_paths = all_paths[split:]

    print(f"[{config.modality}] train={len(train_paths)}  val={len(val_paths)}")

    train_ds = MedicalDataset(train_paths, config)
    val_ds = MedicalDataset(val_paths, config)

    # num_workers=0 and pin_memory=False required for macOS MPS stability
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader
