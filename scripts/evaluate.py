"""Score saved track files against the ground-truth points, without re-running a model.

Reads <pred-dir>/<seq>.txt (frame,id,x,y[,w,h,conf], as written by track_yolo.py or
track_pisttn.py) and prints a per-sequence table plus a pooled row.

  python scripts/evaluate.py --pred-dir results/yolo_botsort --split test
  python scripts/evaluate.py --pred-dir results/yolo_botsort_all --split all --tau 10
"""
import argparse
import json
from pathlib import Path

from irtrack.metrics import evaluate, load_centers

ROOT = Path(__file__).resolve().parents[1]
COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "MOTP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "yolo_botsort"))
    ap.add_argument("--split", default="test", help="train, val, test or all")
    ap.add_argument("--tau", type=float, default=5.0, help="center-distance match threshold (px)")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    proc = ROOT / "processed"
    splits = json.loads((proc / "splits.json").read_text())["splits"]
    if args.split == "all":
        seqs = sorted((s for v in splits.values() for s in v["sequences"]), key=lambda n: int(n[4:]))
    else:
        seqs = splits[args.split]["sequences"]

    pred_dir = Path(args.pred_dir)
    p_key = f"P@{args.tau:g}"
    r_key = f"R@{args.tau:g}"
    f_key = f"F1@{args.tau:g}"
    cols = COLS[:6] + [p_key, r_key, f_key] + COLS[6:]
    print(f"{'sequence':10s}" + "".join(f"{c:>8s}" for c in cols))

    rows, allgt, allpred = {}, {}, {}
    for k, s in enumerate(seqs):
        track_file = pred_dir / f"{s}.txt"
        if not track_file.exists():
            print(f"{s:10s}  (no track file in {pred_dir})")
            continue
        gt = load_centers(proc / "seqs" / s / "gt" / "centers.txt")
        pred = load_centers(track_file)
        rows[s] = evaluate(gt, pred, args.tau)
        # offset frames and ids so sequences can be pooled into one combined score
        off = (k + 1) * 100000
        for fr, v in gt.items():
            allgt[off + fr] = [(off + i, x, y) for i, x, y in v]
        for fr, v in pred.items():
            allpred[off + fr] = [(off + i, x, y) for i, x, y in v]
    if not rows:
        raise SystemExit(f"no track files found in {pred_dir}")
    rows["COMBINED"] = evaluate(allgt, allpred, args.tau)

    for s, r in rows.items():
        cells = [f"{int(r[c]):8d}" if c in ("IDSW", "FP", "FN") else f"{float(r[c]):8.3f}" for c in cols]
        print(f"{s:10s}" + "".join(cells))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({s: {m: float(v) for m, v in r.items()} for s, r in rows.items()},
                                             indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
