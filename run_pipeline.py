"""Run the whole baseline pipeline, or selected stages of it.

Stages, in order:
  data      raw BMPs + point labels -> processed/ (PNG frames, synthesized boxes, YOLO/MOT labels, split)
  train     fine-tune YOLO11n                     -> runs/yolo/yolo11n_ir/weights/best.pt
  track     YOLO + BoT-SORT on val and test       -> results/yolo_botsort/
  evaluate  re-score the saved tracks             -> printed tables
  videos    annotated MP4s of the test sequences  -> results/videos/
  figures   report figures                        -> results/figures/

Examples:
  python run_pipeline.py                         # everything; skips data/train if their outputs exist
  python run_pipeline.py --stages track evaluate videos
  python run_pipeline.py --stages train --force  # retrain even though best.pt exists
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["data", "train", "track", "evaluate", "videos", "figures"]
WEIGHTS = ROOT / "runs" / "yolo" / "yolo11n_ir" / "weights" / "best.pt"


def run(script, *args):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    print("\n>>", " ".join(cmd[1:]), flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--force", action="store_true", help="re-run data/train even if their outputs exist")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pad", type=float, default=16, help="BoT-SORT association box padding (px), chosen on val")
    ap.add_argument("--video-split", default="test", help="split to render videos for (train, val, test, all)")
    args = ap.parse_args()
    stages = [s for s in STAGES if s in args.stages]

    if not (ROOT / "data" / "data_label").exists() and "data" in stages:
        sys.exit("data/ not found: place the raw dataset as data/data1/data1/*.bmp ... and data/data_label/*.txt")

    if "data" in stages:
        if (ROOT / "processed" / "splits.json").exists() and not args.force:
            print("\n== data: processed/ exists, skipping (use --force to rebuild)")
        else:
            run("measure_targets.py")
            run("prepare_data.py")

    if "train" in stages:
        if Path(args.weights).exists() and not args.force:
            print(f"\n== train: {args.weights} exists, skipping (use --force to retrain)")
        else:
            run("train_yolo.py", "--epochs", args.epochs, "--workers", 0)

    if any(s in stages for s in ("track", "evaluate", "videos", "figures")) and not Path(args.weights).exists():
        sys.exit(f"weights not found: {args.weights} (run the train stage first)")

    if "track" in stages:
        for split in ("val", "test"):
            run("track_yolo.py", "--weights", args.weights, "--split", split, "--pad", args.pad)

    if "evaluate" in stages:
        for split in ("val", "test"):
            run("evaluate.py", "--pred-dir", ROOT / "results" / "yolo_botsort", "--split", split)

    if "videos" in stages:
        run("render_videos.py", "--split", args.video_split)

    if "figures" in stages:
        tsvd = ROOT / "results" / "figures" / "tsvd_recall.csv"
        if not tsvd.exists():
            run("check_sparsify.py", "--csv", tsvd)
        run("make_figures.py", "all")

    print("\nPipeline finished:", ", ".join(stages))


if __name__ == "__main__":
    main()
