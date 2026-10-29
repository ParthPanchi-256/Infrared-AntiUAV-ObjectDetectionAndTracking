"""Render annotated MP4s: predicted boxes + track IDs, GT points with the 5 px match radius."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from irtrack.metrics import load_centers

ROOT = Path(__file__).resolve().parents[1]


def id_color(i):
    rng = np.random.default_rng(int(i) * 7919)
    h = int(rng.integers(0, 180))
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in c)


def load_pred(path):
    """frame,id,x,y[,w,h,conf] -> {frame: [(id, x, y, w, h, conf)]}"""
    d = {}
    if not path.exists():
        return d
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 4:
            continue
        w, h, c = (float(v[4]), float(v[5]), float(v[6])) if len(v) >= 7 else (5.0, 5.0, 1.0)
        d.setdefault(int(v[0]), []).append((int(v[1]), float(v[2]), float(v[3]), w, h, c))
    return d


def draw_frame(img, k, preds, gts, scale, tau, label):
    big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    to_px = lambda v: int(round((v + 0.5) * scale))  # pixel-index coords -> upscaled pixel center
    for gid, x, y in gts:
        cv2.circle(big, (to_px(x), to_px(y)), int(tau * scale), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(big, (to_px(x), to_px(y)), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)
    for tid, x, y, w, h, c in preds:
        col = id_color(tid)
        cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
        p1 = (int(cx - w * scale / 2), int(cy - h * scale / 2))
        p2 = (int(cx + w * scale / 2), int(cy + h * scale / 2))
        cv2.rectangle(big, p1, p2, col, 2)
        cv2.putText(big, f"ID{tid} {c:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    cv2.rectangle(big, (0, 0), (big.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(big, f"{label}  frame {k}   GT: green (5px radius)   pred: colored box + ID", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "yolo_botsort"))
    ap.add_argument("--split", default="test", help="train, val, test or all")
    ap.add_argument("--size", type=int, default=768, choices=[512, 768])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "videos"))
    args = ap.parse_args()

    proc = ROOT / "processed"
    splits = json.loads((proc / "splits.json").read_text())["splits"]
    if args.split == "all":
        seqs = sorted((s for v in splits.values() for s in v["sequences"]), key=lambda n: int(n[4:]))
    else:
        seqs = splits[args.split]["sequences"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scale = args.size // 256
    label = Path(args.pred_dir).name
    for s in seqs:
        sd = proc / "seqs" / s
        files = sorted((sd / "images").glob("*.png"))
        preds = load_pred(Path(args.pred_dir) / f"{s}.txt")
        gt = load_centers(sd / "gt" / "centers.txt")
        path = out / f"{label}_{s}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.size, args.size))
        for k, f in enumerate(files):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            vw.write(draw_frame(img, k, preds.get(k, []), gt.get(k, []), scale, args.tau, f"{label} {s}"))
        vw.release()
        print(f"{path}  ({len(files)} frames)")


if __name__ == "__main__":
    main()
