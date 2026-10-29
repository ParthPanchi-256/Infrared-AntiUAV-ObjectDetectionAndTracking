"""Phase 1: convert the raw dataset into a cleaned, split, multi-format dataset.

Outputs (under --out, default processed/):
  seqs/<seq>/images/<idx:06d>.png   single-channel 8-bit frames
  seqs/<seq>/labels/<idx:06d>.txt   YOLO labels (class cx cy w h, normalized); empty = no target
  seqs/<seq>/gt/gt.txt              MOT ground truth: frame(1-based),id,x1,y1,w,h,1,1,1
  seqs/<seq>/gt/centers.txt         frame(0-based),id,x,y  (original point annotations)
  splits.json                       sequence-level train/val/test split + per-seq stats
  yolo_data.yaml                    Ultralytics dataset config
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from irtrack.boxes import clip_xyxy, synthesize
from irtrack.labels import load_all

# Sequence-level split (no frame-level leakage). Val/test hold the multi-target sequences.
SPLITS = {
    "val": ["data2", "data6", "data16"],
    "test": ["data3", "data5", "data19"],
}
W = H = 256


def process_sequence(seq, data_root: Path, out_root: Path) -> dict:
    sdir = out_root / "seqs" / seq.name
    for sub in ("images", "labels", "gt"):
        (sdir / sub).mkdir(parents=True, exist_ok=True)

    cache = {}

    def read_gray(idx):
        if idx not in cache:
            img = cv2.imread(str(seq.image_path(data_root, idx)), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (H, W):
                raise RuntimeError(f"bad frame {seq.name}/{idx}")
            cache[idx] = img
        return cache[idx]

    boxes = synthesize(seq, read_gray)

    def write_frame(idx):
        cv2.imwrite(str(sdir / "images" / f"{idx:06d}.png"), read_gray(idx))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(write_frame, range(seq.num_frames)))

    gt_lines, center_lines, sizes = [], [], []
    for idx in range(seq.num_frames):
        yolo = []
        for oid, x, y, bw, bh in boxes.get(idx, []):
            # annotation is a pixel index; its center in continuous coordinates is +0.5
            x1, y1, x2, y2 = clip_xyxy(x + 0.5, y + 0.5, bw, bh, W, H)
            yolo.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            gt_lines.append(f"{idx + 1},{oid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,1,1")
            center_lines.append(f"{idx},{oid},{x},{y}")
            sizes.append((bw, bh))
        (sdir / "labels" / f"{idx:06d}.txt").write_text("\n".join(yolo) + ("\n" if yolo else ""))
    (sdir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (sdir / "gt" / "centers.txt").write_text("\n".join(center_lines) + "\n")

    s = np.array(sizes) if sizes else np.zeros((0, 2))
    return dict(header_name=seq.header_name, frames=seq.num_frames,
                annotated_frames=sum(1 for o in seq.frames.values() if o),
                objects=len({o[0] for objs in seq.frames.values() for o in objs}),
                box_w_median=float(np.median(s[:, 0])) if len(s) else 0.0,
                box_h_median=float(np.median(s[:, 1])) if len(s) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="processed")
    args = ap.parse_args()
    data_root, out_root = Path(args.data), Path(args.out)

    seqs = load_all(data_root)
    held_out = {s for v in SPLITS.values() for s in v}
    split = {"train": [s.name for s in seqs if s.name not in held_out], **SPLITS}

    stats = {}
    for seq in seqs:
        stats[seq.name] = process_sequence(seq, data_root, out_root)
        st = stats[seq.name]
        print(f"{seq.name:7s} frames={st['frames']:5d} annotated={st['annotated_frames']:5d} "
              f"objs={st['objects']} box~{st['box_w_median']:.0f}x{st['box_h_median']:.0f}")

    summary = {k: dict(sequences=v, frames=sum(stats[s]["frames"] for s in v)) for k, v in split.items()}
    (out_root / "splits.json").write_text(json.dumps(dict(splits=summary, sequences=stats), indent=2))

    seq_dir = (out_root / "seqs").resolve()
    data_yaml = {"path": str(seq_dir), "names": {0: "aircraft"}}
    for k, v in split.items():
        data_yaml[k] = [f"{s}/images" for s in v]
    (out_root / "yolo_data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    for k, v in summary.items():
        print(f"{k:5s}: {len(v['sequences']):2d} seqs, {v['frames']:5d} frames  {v['sequences']}")


if __name__ == "__main__":
    main()
