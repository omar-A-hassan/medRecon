import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    preds, targets: float32 arrays in [0,1], shape [B, H, W] or [B, 1, H, W].
    Returns dict with mae, psnr, ssim averaged over batch.
    """
    if preds.ndim == 4:
        preds = preds[:, 0]
        targets = targets[:, 0]

    maes, psnrs, ssims = [], [], []
    for p, t in zip(preds, targets):
        maes.append(np.mean(np.abs(p - t)))
        psnrs.append(psnr_fn(t, p, data_range=1.0))
        ssims.append(ssim_fn(t, p, data_range=1.0))

    return {
        "mae":  float(np.mean(maes)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }
