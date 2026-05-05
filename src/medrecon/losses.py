import torch
import torch.nn.functional as F


def ssd_loss(output: torch.Tensor, target: torch.Tensor, eta: float) -> torch.Tensor:
    """
    Sparsely Supervised loss adapted from arXiv 2602.02699 (Eq. 8).
    eta=0.0 → standard L1 loss (baseline, no masking).
    eta>0.0 → compute L1 only on Bernoulli-sampled unmasked pixels (fraction 1-eta).
    """
    per_pixel = F.l1_loss(output, target, reduction="none")  # [B, 1, H, W]
    if eta == 0.0:
        return per_pixel.mean()
    # mask=1 means supervised, mask=0 means masked out
    mask = torch.bernoulli(torch.full_like(target, 1.0 - eta))
    supervised = per_pixel * mask
    return supervised.sum() / (mask.sum() + 1e-8)
