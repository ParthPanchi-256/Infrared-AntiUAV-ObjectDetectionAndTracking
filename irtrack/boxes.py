"""Synthesize bounding boxes from center-point annotations.

The dataset only gives target centers. Box size is estimated from the half-maximum
blob around the annotated point (background = median of a ring around it), then
median-filtered along each track so transient clutter merges don't make it jitter.
"""
import cv2
import numpy as np

RING = 24        # half-size of the patch whose border is used as background estimate
SEARCH = 3       # peak is searched within this radius of the annotation
PAD = 1          # padding added around the measured blob
MIN_SIZE = 5     # smallest box side (single-pixel targets still get a usable box)
MAX_SIZE = 48
SMOOTH = 7       # temporal median window (frames)
TRACK_CAP = 2.0  # per-frame size is capped at this multiple of the track's median size


def blob_extent(img: np.ndarray, x: int, y: int) -> tuple[int, int]:
    """Half-maximum blob size (w, h) of the target at (x, y); (0, 0) if no contrast."""
    h, w = img.shape
    x0, x1, y0, y1 = max(0, x - RING), min(w, x + RING + 1), max(0, y - RING), min(h, y + RING + 1)
    p = img[y0:y1, x0:x1].astype(np.float32)
    bg = float(np.median(np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]])))
    cy, cx = y - y0, x - x0
    sy, sx = max(0, cy - SEARCH), max(0, cx - SEARCH)
    win = np.abs(p[sy:cy + SEARCH + 1, sx:cx + SEARCH + 1] - bg)
    py, px = np.unravel_index(np.argmax(win), win.shape)
    py, px = py + sy, px + sx
    amp = abs(p[py, px] - bg)
    if amp < 1:
        return 0, 0
    ph, pw = p.shape
    # a blob reaching the patch border has merged with extended background structure
    # (e.g. bright buildings); raise the threshold until only the target core remains
    for frac in (0.5, 0.65, 0.8, 0.95):
        mask = (np.abs(p - bg) >= frac * amp).astype(np.uint8)
        _, lab = cv2.connectedComponents(mask, connectivity=8)
        ys, xs = np.nonzero(lab == lab[py, px])
        # sides clipped by the image edge don't count as merges
        if ((ys.min() > 0 or y0 == 0) and (xs.min() > 0 or x0 == 0)
                and (ys.max() < ph - 1 or y1 == h) and (xs.max() < pw - 1 or x1 == w)):
            break
    else:
        return 0, 0
    # box must stay centered on the annotation, so take the symmetric extent
    half_w = max(cx - xs.min(), xs.max() - cx)
    half_h = max(cy - ys.min(), ys.max() - cy)
    return int(2 * half_w + 1), int(2 * half_h + 1)


def _median_filter(v: np.ndarray, k: int) -> np.ndarray:
    r = k // 2
    return np.array([np.median(v[max(0, i - r): i + r + 1]) for i in range(len(v))])


def synthesize(seq, read_gray) -> dict:
    """Return {frame: [(obj_id, x, y, w, h)]} with boxes centered on the annotated points.

    read_gray(idx) -> uint8 HxW image.
    """
    raw = {}  # obj_id -> list of (frame, x, y, w, h)
    for idx in sorted(seq.frames):
        objs = seq.frames[idx]
        if not objs:
            continue
        img = read_gray(idx)
        for oid, x, y in objs:
            bw, bh = blob_extent(img, x, y)
            raw.setdefault(oid, []).append((idx, x, y, bw, bh))

    out = {i: [] for i in seq.frames}
    H, W = 256, 256
    for oid, rows in raw.items():
        a = np.array(rows, dtype=np.float64)
        # clutter merges show up as outliers relative to the track's typical size
        for c in (3, 4):
            valid = a[:, c][a[:, c] > 0]
            if len(valid):
                a[:, c] = np.minimum(a[:, c], TRACK_CAP * np.median(valid) + 1)
        ws = _median_filter(a[:, 3], SMOOTH)
        hs = _median_filter(a[:, 4], SMOOTH)
        for (idx, x, y, _, _), bw, bh in zip(rows, ws, hs):
            bw = int(np.clip(bw + 2 * PAD, MIN_SIZE, MAX_SIZE))
            bh = int(np.clip(bh + 2 * PAD, MIN_SIZE, MAX_SIZE))
            out[idx].append((oid, x, y, bw, bh))
    return out


def clip_xyxy(x, y, w, h, W=256, H=256):
    """Box with continuous center (x, y) -> clipped (x1, y1, x2, y2)."""
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(float(W), x + w / 2), min(float(H), y + h / 2)
    return x1, y1, x2, y2
