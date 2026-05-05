import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import ExperimentConfig


class MedicalDataset(Dataset):
    def __init__(self, image_paths: list[str], config: ExperimentConfig):
        self.paths = image_paths
        self.config = config
        self.size = config.img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.size, self.size), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        orig = img.astype(np.float32) / 255.0

        orig_u8 = img  # already uint8

        # 1. Canny edges
        canny = cv2.Canny(orig_u8, self.config.canny_low, self.config.canny_high) / 255.0

        # 2. Sobel magnitude — use float64 input; CV_64F dst from float32 unsupported in OpenCV 4.13+
        orig64 = orig.astype(np.float64)
        sx = cv2.Sobel(orig64, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(orig64, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.clip(cv2.magnitude(sx, sy), 0.0, 1.0).astype(np.float32)

        # 3. Laplacian
        lap = np.abs(cv2.Laplacian(orig64, cv2.CV_64F))
        laplacian = np.clip(lap, 0.0, 1.0).astype(np.float32)

        features = np.stack([canny.astype(np.float32), sobel, laplacian], axis=0)  # [3, H, W]
        target = orig[np.newaxis, ...]                                               # [1, H, W]

        return torch.from_numpy(features), torch.from_numpy(target)
