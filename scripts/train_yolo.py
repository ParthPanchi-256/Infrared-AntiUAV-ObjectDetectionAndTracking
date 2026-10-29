"""Phase 2: fine-tune YOLO11n on the processed dataset (processed/yolo_data.yaml)."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--data", default="processed/yolo_data.yaml")
    # 256x256 frames are upsampled 2x: P3 stride is 8, so a 5 px box at native size is < 1 cell
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="yolo11n_ir")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        workers=args.workers,
        device=0,
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        # absolute: a relative project resolves under the global Ultralytics runs_dir
        project=str(ROOT / "runs" / "yolo"),
        name=args.name,
        exist_ok=True,
        seed=0,
        # grayscale IR: no hue/saturation jitter
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # mild scale jitter so 1-3 px targets don't vanish; mosaic off for the last 10 epochs
        scale=0.25, mosaic=1.0, close_mosaic=10,
        fliplr=0.5,
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()
