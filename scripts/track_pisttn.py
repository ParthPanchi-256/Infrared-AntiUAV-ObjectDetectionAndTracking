"""Phase 3: PISTTN inference + segment tracking on a split, evaluated with the Phase 2 metrics."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from irtrack.metrics import evaluate, load_centers
from irtrack.pisttn.data import T, build_input
from irtrack.pisttn.decode import decode_boxes, frame_detections
from irtrack.pisttn.model import PISTTN, SEG_STARTS, T_DET
from irtrack.pisttn.tracker import SegmentTracker

ROOT = Path(__file__).resolve().parents[1]
STEP = T - 1  # windows overlap by one frame so the 4-frame segments chain with 1-frame overlap


@torch.no_grad()
def track_sequence(model, seq_dir: Path, args):
    files = sorted((seq_dir / "images").glob("*.png"))
    n = len(files)
    frames = np.stack([cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files])
    starts = list(range(0, n - T + 1, STEP))
    if starts[-1] + T < n:
        starts.append(n - T)
    trk = SegmentTracker(buffer=args.buffer, gate=args.gate, min_len=args.min_len)
    done_until = -1
    for st in starts:
        out = model(build_input([torch.from_numpy(frames[st:st + T])], args.thresh))
        score = F.softmax(out["q"], 1)[:, 1]
        dec = decode_boxes(out["hh"].indices, out["det"])
        dets = frame_detections(dec, score[dec["src"]], args.score, args.nms_iou)
        for s, t0 in enumerate(SEG_STARTS):
            if st + t0 + T_DET - 1 <= done_until:
                continue
            seg = {st + t0 + k: np.zeros((0, 7)) for k in range(T_DET)}
            for k in range(T_DET):
                d = dets.get((0, s, t0 + k))
                if d is not None:
                    seg[st + t0 + k] = d.cpu().numpy()
            trk.update(seg)
        done_until = st + T - 1
    # box centers are continuous coordinates; GT points are pixel indices
    return {f: [(i, x - 0.5, y - 0.5) for i, x, y in v] for f, v in trk.results().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresh", type=float, default=None, help="t-SVD lambda (default: from checkpoint)")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--buffer", type=int, default=60)
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "pisttn"))
    args = ap.parse_args()

    ck = torch.load(args.weights, map_location="cuda")
    if args.thresh is None:
        args.thresh = ck["args"]["thresh"]
    model = PISTTN(with_detector=True).cuda().eval()
    model.load_state_dict(ck["model"])

    proc = ROOT / "processed"
    seqs = json.loads((proc / "splits.json").read_text())["splits"][args.split]["sequences"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allgt, allpred, rows = {}, {}, {}
    for k, s in enumerate(seqs):
        t0 = time.time()
        pred = track_sequence(model, proc / "seqs" / s, args)
        fps = len(list((proc / "seqs" / s / "images").glob("*.png"))) / (time.time() - t0)
        gt = load_centers(proc / "seqs" / s / "gt" / "centers.txt")
        with open(out / f"{s}.txt", "w") as f:
            for fr in sorted(pred):
                for i, x, y in pred[fr]:
                    f.write(f"{fr},{i},{x:.2f},{y:.2f}\n")
        rows[s] = dict(evaluate(gt, pred, args.tau), FPS=fps)
        off = (k + 1) * 100000
        for fr, v in gt.items():
            allgt[off + fr] = [(off + i, x, y) for i, x, y in v]
        for fr, v in pred.items():
            allpred[off + fr] = [(off + i, x, y) for i, x, y in v]
        print(s, {m: round(float(v), 3) for m, v in rows[s].items()}, flush=True)
    rows["COMBINED"] = evaluate(allgt, allpred, args.tau)
    print("COMBINED", {m: round(float(v), 3) for m, v in rows["COMBINED"].items()})
    (out / f"metrics_{args.split}.json").write_text(
        json.dumps({s: {m: float(v) for m, v in r.items()} for s, r in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
