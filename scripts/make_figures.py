"""Report figures -> results/figures/.

  python scripts/make_figures.py data       # target crops, box overlays, motion stats, t-SVD recall
  python scripts/make_figures.py results    # YOLO training curves, success/failure frames
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from irtrack.metrics import load_centers

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
FIG = ROOT / "results" / "figures"

# reference categorical order (dataviz palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
})


def splits():
    return json.loads((PROC / "splits.json").read_text())["splits"]


def all_seqs():
    order = [s for k in ("train", "val", "test") for s in splits()[k]["sequences"]]
    return sorted(order, key=lambda n: int(n[4:]))


def split_of(s):
    return next(k for k, v in splits().items() if s in v["sequences"])


def read_gray(s, i):
    return cv2.imread(str(PROC / "seqs" / s / "images" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)


def gt_boxes(s):
    d = {}
    for line in open(PROC / "seqs" / s / "gt" / "gt.txt"):
        if line.strip():
            f, i, x1, y1, w, h = [float(v) for v in line.split(",")[:6]]
            d.setdefault(int(f) - 1, []).append((int(i), x1, y1, w, h))
    return d


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- data figures
def fig_target_crops():
    """33x33 crops around the annotated point: first / middle / last annotated frame of each sequence."""
    tiles = []
    for s in all_seqs():
        gt = load_centers(PROC / "seqs" / s / "gt" / "centers.txt")
        idx = sorted(gt)
        row = []
        for i in (idx[0], idx[len(idx) // 2], idx[-1]):
            _, x, y = gt[i][0]
            x, y = int(x), int(y)
            p = cv2.copyMakeBorder(read_gray(s, i), 16, 16, 16, 16, cv2.BORDER_CONSTANT)[y:y + 33, x:x + 33]
            p = cv2.cvtColor(cv2.resize(p, (132, 132), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            row.append(p)
        tiles.append(label(np.hstack(row), f"{s} ({split_of(s)})"))
    half = (len(tiles) + 1) // 2
    a = np.vstack(tiles[:half])
    b = np.vstack(tiles[half:] + [np.full_like(tiles[0], 255)] * (half - len(tiles[half:])))
    cv2.imwrite(str(FIG / "target_crops.png"), np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b]))


def fig_box_overlays():
    """Synthesized GT boxes on full frames, with a 4x zoom inset around the first target."""
    rows = []
    for s in ["data1", "data3", "data7", "data14", "data16", "data5"]:
        n = len(list((PROC / "seqs" / s / "images").glob("*.png")))
        boxes = gt_boxes(s)
        tiles = []
        for i in (n // 5, n // 2, 4 * n // 5):
            big = cv2.resize(cv2.cvtColor(read_gray(s, i), cv2.COLOR_GRAY2BGR), (512, 512),
                             interpolation=cv2.INTER_NEAREST)
            for _, x1, y1, w, h in boxes.get(i, []):
                cv2.rectangle(big, (int(x1 * 2), int(y1 * 2)), (int((x1 + w) * 2), int((y1 + h) * 2)), (0, 255, 0), 1)
            if boxes.get(i):
                _, x1, y1, w, h = boxes[i][0]
                cx, cy = int((x1 + w / 2) * 2), int((y1 + h / 2) * 2)
                pad = cv2.copyMakeBorder(big, 32, 32, 32, 32, cv2.BORDER_CONSTANT)
                big[0:192, 320:512] = cv2.resize(pad[cy:cy + 64, cx:cx + 64], (192, 192), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(big, (320, 0), (511, 191), (255, 255, 255), 1)
            tiles.append(label(cv2.resize(big, (320, 320), interpolation=cv2.INTER_AREA), f"{s} frame {i}"))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(FIG / "box_overlays.png"), np.vstack(rows))


def fig_motion():
    """Per-frame displacement of each target (p50 / p95 / max) per sequence."""
    seqs, stats = all_seqs(), []
    for s in seqs:
        gt = load_centers(PROC / "seqs" / s / "gt" / "centers.txt")
        d = [np.hypot(x2 - x, y2 - y) for f in gt for i, x, y in gt[f]
             for j, x2, y2 in gt.get(f + 1, []) if i == j]
        d = np.array(d)
        stats.append((s, split_of(s), np.median(d), np.percentile(d, 95), d.max(), np.mean(d > 4)))
    with open(FIG / "motion_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "split", "p50_px", "p95_px", "max_px", "frac_gt_4px"])
        w.writerows([(a, b, f"{c:.2f}", f"{d:.2f}", f"{e:.2f}", f"{g:.4f}") for a, b, c, d, e, g in stats])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(stats))[::-1]
    for k, (name, col, mk) in enumerate([("median", C[0], "o"), ("95th percentile", C[1], "D"), ("max", C[6], "^")]):
        v = [max(0.3, r[2 + k]) for r in stats]  # log axis: zero displacement is drawn at the "0" tick (0.3)
        ax.scatter(v, y, s=36, color=col, marker=mk, label=name, zorder=3, edgecolor=SURF, linewidth=1.5)
    for yy, r in zip(y, stats):
        ax.plot([max(0.3, r[2]), r[4]], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}  ({r[1]})" for r in stats])
    ax.axvline(4, color=INK2, lw=1, ls="--")
    ax.text(4.2, y[0] + 0.6, "4 px", color=INK2, fontsize=9)
    ax.set_xlabel("target displacement between consecutive frames (px, log scale)")
    ax.set_title("Per-frame target motion by sequence")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "motion_stats.png", dpi=150)
    plt.close(fig)


def fig_tsvd(csv_path):
    """t-SVD target hit-rate heatmap (sequence x threshold) from scripts/check_sparsify.py --csv."""
    rows = list(csv.DictReader(open(csv_path)))
    seqs = sorted({r["seq"] for r in rows}, key=lambda n: int(n[4:]))
    th = sorted({float(r["thresh"]) for r in rows})
    hit = np.array([[next(float(r["target_hit_rate"]) for r in rows if r["seq"] == s and float(r["thresh"]) == t)
                     for t in th] for s in seqs])
    act = np.array([[next(float(r["active_fraction"]) for r in rows if r["seq"] == s and float(r["thresh"]) == t)
                     for t in th] for s in seqs])
    # markdown table for the report
    with open(FIG / "tsvd_recall.md", "w", encoding="utf-8") as f:
        f.write("| Sequence | Split | " + " | ".join(f"λ={t:g} hit / active" for t in th) + " |\n")
        f.write("|---|---|" + "---|" * len(th) + "\n")
        for s, h, a in zip(seqs, hit, act):
            f.write(f"| {s} | {split_of(s)} | " + " | ".join(f"{100*x:.0f}% / {100*y:.2f}%" for x, y in zip(h, a)) + " |\n")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#e9f1fc", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(5.2, 7.5))
    im = ax.imshow(hit * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(th)))
    ax.set_xticklabels([f"λ={t:g}" for t in th])
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels([f"{s} ({split_of(s)})" for s in seqs])
    ax.grid(False)
    for i in range(len(seqs)):
        for j in range(len(th)):
            v = hit[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK)
    ax.set_title("Targets kept by t-SVD sparsification (%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("GT targets with an active voxel within 1 px (%)", color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "tsvd_recall.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- result figures
def fig_yolo_curves(run_dir):
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(run_dir / "results.csv"))]
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, keys, title in [
        (axes[0], [("train/box_loss", "train box"), ("val/box_loss", "val box")], "Box loss"),
        (axes[1], [("train/cls_loss", "train cls"), ("val/cls_loss", "val cls")], "Classification loss"),
        (axes[2], [("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall"),
                   ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")], "Validation metrics"),
    ]:
        for k, (key, name) in enumerate(keys):
            ax.plot(ep, [r[key] for r in rows], color=C[k], label=name, marker="o", ms=3)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(keys), fontsize=9)
    axes[2].set_ylim(0, 1)
    best = max(rows, key=lambda r: r["metrics/mAP50(B)"] * 0.1 + r["metrics/mAP50-95(B)"] * 0.9)  # Ultralytics fitness
    axes[2].axvline(best["epoch"], color=INK2, lw=1, ls="--")
    axes[2].text(best["epoch"], 0.03, f"best.pt (epoch {int(best['epoch'])}) ", color=INK2, fontsize=9, ha="right")
    fig.suptitle("YOLO11n fine-tuning (val = data2, data6, data16)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "yolo_training_curves.png", dpi=150)
    plt.close(fig)


def fig_pad_sweep(sweep_dir):
    """Validation tracking metrics vs. association box padding."""
    rows = []
    for d in Path(sweep_dir).glob("pad*"):
        m = json.loads((d / "metrics_val.json").read_text())["COMBINED"]
        rows.append((float(d.name[3:]), m))
    if not rows:
        return
    rows.sort()
    pads = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, key in enumerate(["HOTA", "IDF1", "MOTA", "AssA", "DetA"]):
        ax.plot(pads, [r[1][key] for r in rows], color=C[k], marker="o", ms=4, label=key)
    best = max(rows, key=lambda r: r[1]["HOTA"])
    ax.axvline(best[0], color=INK2, lw=1, ls="--")
    ax.text(best[0], 0.42, f" chosen: {best[0]:g} px", color=INK2, fontsize=9)
    ax.set_xticks(pads)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("box padding per side for association (px)")
    ax.set_title("BoT-SORT on val: effect of association box padding")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "pad_sweep_val.png", dpi=150)
    plt.close(fig)


def frame_status(gt, pred, tau=5.0):
    """Per frame: 'ok' if every GT is matched and there are no FPs, else a failure kind."""
    from scipy.optimize import linear_sum_assignment
    status = {}
    for f in sorted(set(gt) | set(pred)):
        g, p = gt.get(f, []), pred.get(f, [])
        if not g and not p:
            continue
        m = 0
        if g and p:
            d = np.array([[np.hypot(a[1] - b[1], a[2] - b[2]) for b in p] for a in g])
            r, c = linear_sum_assignment(d)
            m = int((d[r, c] <= tau).sum())
        if m == len(g) == len(p):
            status[f] = "ok"
        elif m < len(g) and len(p) > m:
            status[f] = "missed + false alarm"   # predictions exist but none within tau of the target
        elif m < len(g):
            status[f] = "missed"
        else:
            status[f] = "false alarm"
    return status


def fig_success_failure(pred_dir, n_ok=3, n_fail=3):
    from render_videos import draw_frame, load_pred
    test = splits()["test"]["sequences"]
    picks_ok, picks_fail = [], []
    rng = np.random.default_rng(0)
    for s in test:
        gt = load_centers(PROC / "seqs" / s / "gt" / "centers.txt")
        pred_full = load_pred(Path(pred_dir) / f"{s}.txt")
        st = frame_status(gt, {f: [p[:3] for p in v] for f, v in pred_full.items()})
        ok = [f for f, v in st.items() if v == "ok"]
        bad = [f for f, v in st.items() if v != "ok"]
        if ok:
            picks_ok.append((s, int(rng.choice(ok)), "ok", gt, pred_full))
        for kind in ("missed", "false alarm", "missed + false alarm"):
            fs = [f for f in bad if st[f] == kind]
            if fs:
                picks_fail.append((s, int(rng.choice(fs)), kind, gt, pred_full))
    # one failure per kind, each from a different sequence where possible, then fill
    chosen_fail, kinds, seqs = [], set(), set()
    for prefer_new_seq in (True, False):
        for p in picks_fail:
            if p[2] not in kinds and (not prefer_new_seq or p[0] not in seqs):
                chosen_fail.append(p); kinds.add(p[2]); seqs.add(p[0])
    chosen_fail += [p for p in picks_fail if p not in chosen_fail]
    sel = picks_ok[:n_ok] + chosen_fail[:n_fail]
    tiles = []
    for s, f, kind, gt, pred in sel:
        img = draw_frame(read_gray(s, f), f, pred.get(f, []), gt.get(f, []), 2, 5.0, "")
        # 3x zoom inset (bottom-right) around the GT target, or the first prediction if none
        anchor = gt.get(f) or pred.get(f)
        if anchor:
            cx, cy = int((anchor[0][1] + 0.5) * 2), int((anchor[0][2] + 0.5) * 2)
            pad = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT)
            inset = cv2.resize(pad[cy:cy + 64, cx:cx + 64], (192, 192), interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
            img[h - 194:h - 2, w - 194:w - 2] = inset
            cv2.rectangle(img, (w - 195, h - 195), (w - 2, h - 2), (255, 255, 255), 1)
        img = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        col = (60, 200, 60) if kind == "ok" else (60, 60, 230)
        cv2.putText(img, f"{s} frame {f}: {'SUCCESS' if kind == 'ok' else 'FAILURE - ' + kind}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(FIG / "success_failure_frames.png"), grid)
    return [(s, f, k) for s, f, k, _, _ in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "results", "all"])
    ap.add_argument("--tsvd-csv", default=str(FIG / "tsvd_recall.csv"))
    ap.add_argument("--yolo-run", default=str(ROOT / "runs" / "yolo" / "yolo11n_ir"))
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "yolo_botsort"))
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.what in ("data", "all"):
        fig_target_crops(); fig_box_overlays(); fig_motion()
        if Path(args.tsvd_csv).exists():
            fig_tsvd(args.tsvd_csv)
        else:
            print("skip t-SVD figure: run scripts/check_sparsify.py --csv", args.tsvd_csv)
    if args.what in ("results", "all"):
        fig_yolo_curves(Path(args.yolo_run))
        fig_pad_sweep(Path(args.pred_dir) / "pad_sweep")
        print("success/failure frames:", fig_success_failure(args.pred_dir))
    print("figures in", FIG)


if __name__ == "__main__":
    main()
