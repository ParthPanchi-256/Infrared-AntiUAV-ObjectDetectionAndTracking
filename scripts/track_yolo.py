"""Phase 2: run YOLO + BoT-SORT sequence by sequence and evaluate against point GT."""
import argparse
import gc
import json
import logging
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils import LOGGER

from irtrack.metrics import evaluate, load_centers

ROOT = Path(__file__).resolve().parents[1]


class GMCFailureCounter(logging.Filter):
    """Counts (and hides) BoT-SORT GMC fallbacks on low-texture frames."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def filter(self, record):
        if "not enough matching points" in record.getMessage():
            self.count += 1
            return False
        return True


GMC_FAILS = GMCFailureCounter()
LOGGER.addFilter(GMC_FAILS)


def pad_boxes_callback(pad):
    """Grow detections by `pad` px per side before BoT-SORT sees them.

    IoU association breaks down for ~5 px boxes: a 2-3 px Kalman/camera-shake error drops
    IoU to 0 and spawns a new ID. Padding keeps the IoU gate usable; it is removed again
    from the tracker output. Boxes are set directly (not via Results.update) so they are not
    clipped at the image border, which would shift their centers.
    """
    def cb(predictor):
        for r in predictor.results:
            if r.boxes is not None and len(r.boxes):
                d = r.boxes.data.clone()
                d[:, :2] -= pad
                d[:, 2:4] += pad
                r.boxes = Boxes(d, r.orig_shape)
    return cb


def track_sequence(weights, seq_dir: Path, args) -> dict:
    # fresh model per sequence so tracker state never leaks between sequences
    model = YOLO(weights)
    if args.pad > 0:
        # registered before the first track() call, so it runs before the tracker's callback
        model.add_callback("on_predict_postprocess_end", pad_boxes_callback(args.pad))
    pred = {}
    frames = sorted((seq_dir / "images").glob("*.png"))
    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        r = model.track(img, persist=True, tracker=args.tracker, conf=args.conf, iou=args.iou,
                        imgsz=args.imgsz, device=0, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            # box centers are continuous coordinates; GT points are pixel indices
            pred[idx] = [(int(i), float(x) - 0.5, float(y) - 0.5,
                          max(1.0, float(w) - 2 * args.pad), max(1.0, float(h) - 2 * args.pad), float(c))
                         for (x, y, w, h), i, c in zip(r.boxes.xywh.tolist(), r.boxes.id.tolist(),
                                                       r.boxes.conf.tolist())]
    del model
    gc.collect(); torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", help="train, val, test or all")
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "botsort_ir.yaml"))
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--pad", type=float, default=0.0, help="px added per box side for association only")
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "yolo_botsort"))
    args = ap.parse_args()

    proc = ROOT / "processed"
    splits = json.loads((proc / "splits.json").read_text())["splits"]
    if args.split == "all":
        seqs = sorted((s for v in splits.values() for s in v["sequences"]), key=lambda n: int(n[4:]))
    else:
        seqs = splits[args.split]["sequences"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"config_{args.split}.json").write_text(json.dumps(vars(args), indent=2))
    allgt, allpred, rows = {}, {}, {}
    for k, s in enumerate(seqs):
        GMC_FAILS.count = 0
        t0 = time.time()
        full = track_sequence(args.weights, proc / "seqs" / s, args)
        elapsed = time.time() - t0
        pred = {fr: [p[:3] for p in v] for fr, v in full.items()}
        gt = load_centers(proc / "seqs" / s / "gt" / "centers.txt")
        with open(out / f"{s}.txt", "w") as f:
            # frame(0-based),id,x,y,w,h,conf -- x,y in the GT pixel-index convention
            for fr in sorted(full):
                for i, x, y, w, h, c in full[fr]:
                    f.write(f"{fr},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.3f}\n")
        n_frames = len(list((proc / "seqs" / s / "images").glob("*.png")))
        rows[s] = dict(evaluate(gt, pred, args.tau), GMC_fail_rate=GMC_FAILS.count / n_frames,
                       FPS=n_frames / elapsed)
        # offset frames and ids so sequences can be pooled into one combined score
        off = (k + 1) * 100000
        for fr, v in gt.items():
            allgt[off + fr] = [(off + i, x, y) for i, x, y in v]
        for fr, v in pred.items():
            allpred[off + fr] = [(off + i, x, y) for i, x, y in v]
        print(s, {m: round(float(v), 3) for m, v in rows[s].items()})
    rows["COMBINED"] = evaluate(allgt, allpred, args.tau)
    print("COMBINED", {m: round(float(v), 3) for m, v in rows["COMBINED"].items()})
    (out / f"metrics_{args.split}.json").write_text(
        json.dumps({s: {m: float(v) for m, v in r.items()} for s, r in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
