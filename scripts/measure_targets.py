"""Measure empirical target extents around annotated centers (half-maximum blob size)."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from irtrack.labels import load_all

R = 10  # patch half-size


def measure(img: np.ndarray, x: int, y: int):
    h, w = img.shape
    x0, x1, y0, y1 = max(0, x - R), min(w, x + R + 1), max(0, y - R), min(h, y + R + 1)
    p = img[y0:y1, x0:x1].astype(np.float32)
    border = np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]])
    bg = float(np.median(border))
    # peak within 3 px of the annotation (targets may be brighter or darker than background)
    cy, cx = y - y0, x - x0
    win = p[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
    dev = np.abs(win - bg)
    py, px = np.unravel_index(np.argmax(dev), dev.shape)
    py += max(0, cy - 3); px += max(0, cx - 3)
    amp = abs(p[py, px] - bg)
    if amp < 1:
        return None
    mask = (np.abs(p - bg) >= 0.5 * amp).astype(np.uint8)
    _, lab = cv2.connectedComponents(mask, connectivity=8)
    ys, xs = np.nonzero(lab == lab[py, px])
    return dict(w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1),
                contrast=float(amp), bg_std=float(border.std()),
                dx=int(px - cx), dy=int(py - cy), sign=int(np.sign(p[py, px] - bg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-seq", type=int, default=30)
    ap.add_argument("--out", default="processed/target_stats.json")
    args = ap.parse_args()
    random.seed(0)
    root = Path(args.data)
    rows = []
    for seq in load_all(root):
        idxs = [i for i, o in seq.frames.items() if o]
        for i in random.sample(idxs, min(args.per_seq, len(idxs))):
            img = cv2.imread(str(seq.image_path(root, i)), cv2.IMREAD_GRAYSCALE)
            for oid, x, y in seq.frames[i]:
                m = measure(img, x, y)
                if m:
                    rows.append(dict(seq=seq.name, frame=i, obj=oid, **m))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    a = {k: np.array([r[k] for r in rows]) for k in ("w", "h", "contrast", "bg_std", "dx", "dy", "sign")}
    print(f"samples={len(rows)}")
    for k in ("w", "h", "contrast", "bg_std"):
        print(f"{k:8s} p10={np.percentile(a[k],10):.1f} p50={np.median(a[k]):.1f} p90={np.percentile(a[k],90):.1f} max={a[k].max():.1f}")
    print(f"peak offset from label: |d|<=1 in {np.mean((abs(a['dx'])<=1)&(abs(a['dy'])<=1))*100:.0f}% of samples")
    print(f"bright targets: {np.mean(a['sign']>0)*100:.0f}%")
    print("per-seq median w x h, contrast, SCR(contrast/bg_std):")
    for s in dict.fromkeys(r["seq"] for r in rows):
        rr = [r for r in rows if r["seq"] == s]
        print(f"  {s:7s} {np.median([r['w'] for r in rr]):4.1f} x {np.median([r['h'] for r in rr]):4.1f}"
              f"  c={np.median([r['contrast'] for r in rr]):6.1f}  scr={np.median([r['contrast']/max(r['bg_std'],1e-3) for r in rr]):6.1f}")


if __name__ == "__main__":
    main()
