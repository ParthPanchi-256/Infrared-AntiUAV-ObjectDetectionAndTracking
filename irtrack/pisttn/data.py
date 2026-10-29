"""10-frame clip dataset and sparse-tensor / target construction for PISTTN."""
from pathlib import Path

import cv2
import numpy as np
import torch
import spconv.pytorch as sp

from .sparsify import tsvd_sparsify

T = 10
LOW_STRIDE = 8  # three (1,2,2) downsamples


def load_gt_boxes(seq_dir: Path, n_frames: int):
    """gt.txt -> list over frames of arrays [id, cx, cy, w, h] (continuous pixel coords)."""
    per = [[] for _ in range(n_frames)]
    for line in open(seq_dir / "gt" / "gt.txt"):
        if not line.strip():
            continue
        f, i, x1, y1, w, h = [float(v) for v in line.split(",")[:6]]
        per[int(f) - 1].append((i, x1 + w / 2, y1 + h / 2, w, h))
    return [np.array(p, np.float32).reshape(-1, 5) for p in per]


class ClipDataset(torch.utils.data.Dataset):
    """All 10-frame windows (step `stride`) from the given sequences."""

    def __init__(self, proc_root: Path, seqs, stride=1, augment=False):
        self.items, self.frames, self.gt = [], {}, {}
        self.augment = augment
        for s in seqs:
            sd = proc_root / "seqs" / s
            files = sorted((sd / "images").glob("*.png"))
            self.frames[s] = files
            self.gt[s] = load_gt_boxes(sd, len(files))
            self.items += [(s, st) for st in range(0, len(files) - T + 1, stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s, st = self.items[k]
        clip = np.stack([cv2.imread(str(self.frames[s][st + i]), cv2.IMREAD_GRAYSCALE) for i in range(T)])
        gt = [g.copy() for g in self.gt[s][st:st + T]]
        if self.augment:
            H, W = clip.shape[1:]
            if np.random.rand() < 0.5:
                clip = clip[:, :, ::-1]
                for g in gt:
                    g[:, 1] = W - g[:, 1]
            if np.random.rand() < 0.5:
                clip = clip[:, ::-1]
                for g in gt:
                    g[:, 2] = H - g[:, 2]
            if np.random.rand() < 0.5:  # play the clip backwards
                clip, gt = clip[::-1], gt[::-1]
        return torch.from_numpy(np.ascontiguousarray(clip)), gt, (s, st)


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def build_input(clips, thresh=20.0, device="cuda"):
    """List of (T, H, W) uint8 clips -> SparseConvTensor of the t-SVD sparse component."""
    coords, feats = [], []
    for b, clip in enumerate(clips):
        c, v = tsvd_sparsify(clip.to(device).float(), thresh=thresh)
        if len(c) == 0:  # static window: spconv cannot run on an empty tensor, add one zero voxel
            c, v = torch.zeros(1, 3, dtype=c.dtype, device=c.device), torch.zeros(1, device=v.device)
        coords.append(torch.cat([torch.full_like(c[:, :1], b), c], 1))
        feats.append(v[:, None] / 255.0)
    H, W = clips[0].shape[1:]
    return sp.SparseConvTensor(torch.cat(feats), torch.cat(coords).int(), [T, H, W], len(clips))


def _boxes_tensor(gts, device):
    """Pad per-(batch, frame) boxes into (B, T, K, 7): id, cx, cy, w, h, vx, vy, plus a valid mask."""
    B = len(gts)
    K = max(1, max(len(g) for gt in gts for g in gt))
    out = np.zeros((B, T, K, 7), np.float32)
    valid = np.zeros((B, T, K), bool)
    for b, gt in enumerate(gts):
        for t, g in enumerate(gt):
            for k, row in enumerate(g):
                out[b, t, k, :5] = row
                valid[b, t, k] = True
                # per-frame motion of the same object (forward difference, backward at the end)
                nxt = gt[t + 1] if t + 1 < T else None
                prv = gt[t - 1] if t > 0 else None
                for other, sign in ((nxt, 1.0), (prv, -1.0)):
                    if other is not None and len(other):
                        m = other[other[:, 0] == row[0]]
                        if len(m):
                            out[b, t, k, 5] = sign * (m[0, 1] - row[1])
                            out[b, t, k, 6] = sign * (m[0, 2] - row[2])
                            break
    return torch.from_numpy(out).to(device), torch.from_numpy(valid).to(device)


def voxel_targets(indices, gts, stride=1):
    """For voxels (b, t, y, x) at the given spatial stride, return (pos mask, matched box row).

    A voxel is positive if its cell overlaps a GT box of its frame.
    """
    dev = indices.device
    boxes, valid = _boxes_tensor(gts, dev)
    b, t = indices[:, 0].long(), indices[:, 1].long()
    y0 = indices[:, 2].float() * stride
    x0 = indices[:, 3].float() * stride
    bx = boxes[b, t]                              # (N, K, 7)
    vm = valid[b, t]
    x1, x2 = bx[..., 1] - bx[..., 3] / 2, bx[..., 1] + bx[..., 3] / 2
    y1, y2 = bx[..., 2] - bx[..., 4] / 2, bx[..., 2] + bx[..., 4] / 2
    inside = (x0[:, None] + stride > x1) & (x0[:, None] < x2) & (y0[:, None] + stride > y1) & (y0[:, None] < y2) & vm
    # if inside several boxes, take the nearest center
    d = (bx[..., 1] - (x0[:, None] + stride / 2)) ** 2 + (bx[..., 2] - (y0[:, None] + stride / 2)) ** 2
    d = torch.where(inside, d, torch.full_like(d, 1e9))
    k = d.argmin(1)
    pos = inside.any(1)
    return pos, bx[torch.arange(len(k), device=dev), k]
