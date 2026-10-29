"""PISTTN losses: focal spatial-structure loss (Eqs. 13-16) and CIoU loss (Eq. 17)."""
import math

import torch
import torch.nn.functional as F


def focal_loss(logits2, target, alpha=0.25, gamma=2.0):
    """logits2: (N, 2) background/target logits; target: (N,) bool.

    Normalized by the number of positives (RetinaNet convention) rather than by H*W as in
    Eq. 16: with ~1e5 mostly-negative voxels the mean is too small to train at lr 1e-4.
    """
    p = F.softmax(logits2, 1)[:, 1].clamp(1e-6, 1 - 1e-6)
    t = target.float()
    pt = torch.where(target, p, 1 - p)
    a = torch.where(target, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    loss = -a * (1 - pt) ** gamma * torch.log(pt)
    return loss.sum() / t.sum().clamp(min=1.0)


def ciou_loss(pred, gt, eps=1e-7):
    """pred, gt: (N, 4) boxes as (cx, cy, w, h). Returns mean(1 - CIoU)."""
    p1, p2 = pred[:, :2] - pred[:, 2:] / 2, pred[:, :2] + pred[:, 2:] / 2
    g1, g2 = gt[:, :2] - gt[:, 2:] / 2, gt[:, :2] + gt[:, 2:] / 2
    inter = (torch.min(p2, g2) - torch.max(p1, g1)).clamp(min=0).prod(1)
    union = pred[:, 2:].prod(1) + gt[:, 2:].prod(1) - inter + eps
    iou = inter / union
    c2 = ((torch.max(p2, g2) - torch.min(p1, g1)) ** 2).sum(1) + eps
    rho2 = ((pred[:, :2] - gt[:, :2]) ** 2).sum(1)
    v = (4 / math.pi ** 2) * (torch.atan(gt[:, 2] / (gt[:, 3] + eps)) - torch.atan(pred[:, 2] / (pred[:, 3] + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return (1 - (iou - rho2 / c2 - alpha * v)).mean()
