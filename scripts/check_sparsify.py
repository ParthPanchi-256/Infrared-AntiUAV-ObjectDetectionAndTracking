"""Check what fraction of GT targets survive t-SVD sparsification, and how sparse the tensor is."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from irtrack.metrics import load_centers
from irtrack.pisttn.sparsify import tsvd_sparsify

ap = argparse.ArgumentParser()
ap.add_argument("--thresh", type=float, nargs="+", default=[10, 15, 20, 30])
ap.add_argument("--windows", type=int, default=25)
ap.add_argument("--radius", type=int, default=1)
ap.add_argument("--csv", default=None, help="write seq,split,thresh,hit,active rows here")
args = ap.parse_args()
random.seed(0)
proc = Path("processed")
splits = json.loads((proc / "splits.json").read_text())["splits"]
split_of = {s: k for k, v in splits.items() for s in v["sequences"]}
seqs = list(split_of)
rows = []
print("seq      " + "  ".join(f"thr{t:<3g} hit%  act%" for t in args.thresh))
for s in sorted(seqs, key=lambda n: int(n[4:])):
    sd = proc / "seqs" / s
    gt = load_centers(sd / "gt" / "centers.txt")
    n = len(list((sd / "images").glob("*.png")))
    res = {t: [0, 0, 0.0] for t in args.thresh}
    for st in random.sample(range(n - 9), min(args.windows, n - 9)):
        clip = torch.from_numpy(np.stack([cv2.imread(str(sd / "images" / f"{i:06d}.png"), 0)
                                          for i in range(st, st + 10)])).float().cuda()
        for t in args.thresh:
            c, _ = tsvd_sparsify(clip, thresh=t)
            vol = torch.zeros(10, 256, 256, dtype=torch.bool, device="cuda")
            vol[c[:, 0], c[:, 1], c[:, 2]] = True
            r = args.radius
            for k in range(10):
                for _, x, y in gt.get(st + k, []):
                    x, y = int(x), int(y)
                    res[t][0] += int(vol[k, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any())
                    res[t][1] += 1
            res[t][2] += len(c) / vol.numel()
    nw = min(args.windows, n - 9)
    rows += [(s, split_of[s], t, h / max(1, g), a / nw) for t, (h, g, a) in res.items()]
    print(f"{s:7s}  " + "  ".join(f"      {100*h/max(1,g):5.1f} {100*a/nw:5.2f}" for h, g, a in res.values()))
if args.csv:
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w") as f:
        f.write("seq,split,thresh,target_hit_rate,active_fraction\n")
        f.writelines(f"{a},{b},{c:g},{d:.4f},{e:.5f}\n" for a, b, c, d, e in rows)
