"""Phase 3: two-stage PISTTN training (Sec. III-B).

Stage 1: STE + STQ, focal loss on the queries.
Stage 2: attach the profile detector, fine-tune end to end with 0.1*L_ss + 2.0*L_CIoU.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from irtrack.pisttn.data import ClipDataset, build_input, collate, voxel_targets, LOW_STRIDE
from irtrack.pisttn.decode import BASE_SIZE, decode_boxes
from irtrack.pisttn.losses import ciou_loss, focal_loss
from irtrack.pisttn.model import PISTTN

ROOT = Path(__file__).resolve().parents[1]


def compute_loss(model, out, gts, stage):
    pos_l, _ = voxel_targets(out["hl"].indices, gts, LOW_STRIDE)
    pos_h, row_h = voxel_targets(out["hh"].indices, gts, 1)
    l_ss = focal_loss(out["ql"], pos_l) + focal_loss(out["q"], pos_h)
    logs = dict(l_ss=l_ss.item(), npos_h=int(pos_h.sum()))
    if stage == 1:
        return l_ss, logs
    dec = decode_boxes(out["hh"].indices, out["det"])
    m = pos_h[dec["src"]]
    if m.any():
        gt = row_h[dec["src"]][m]
        l_iou = ciou_loss(dec["box"][m], gt[:, 1:5])
        l_vel = F.smooth_l1_loss(dec["vel"][m], gt[:, 5:7])
    else:
        l_iou = l_vel = out["q"].sum() * 0
    logs.update(l_iou=l_iou.item(), l_vel=l_vel.item())
    return 0.1 * l_ss + 2.0 * l_iou + 0.1 * l_vel, logs


@torch.no_grad()
def query_scores(out, gts, thr=0.5):
    """Voxel-level precision/recall of the fused high-res query."""
    pos, _ = voxel_targets(out["hh"].indices, gts, 1)
    pred = F.softmax(out["q"], 1)[:, 1] >= thr
    return int((pred & pos).sum()), int(pred.sum()), int(pos.sum())


def run_epoch(model, loader, args, opt=None):
    train = opt is not None
    model.train(train)
    tot, n, tp, npred, npos = 0.0, 0, 0, 0, 0
    t0 = time.time()
    for it, (clips, gts, _) in enumerate(loader):
        if train and args.iters and it >= args.iters:
            break
        x = build_input(clips, args.thresh)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss, logs = compute_loss(model, out, gts, args.stage)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        a, b, c = query_scores(out, gts)
        tp, npred, npos = tp + a, npred + b, npos + c
        tot += loss.item(); n += 1
        if train and it % args.log_every == 0:
            print(f"  it {it:5d} loss {loss.item():.4f} {logs} {n / (time.time() - t0):.1f} it/s", flush=True)
    return dict(loss=tot / max(1, n), q_prec=tp / max(1, npred), q_rec=tp / max(1, npos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=2000, help="train iterations per epoch (0 = full pass)")
    ap.add_argument("--val-stride", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optim", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--thresh", type=float, default=20.0, help="t-SVD hard threshold lambda_seg")
    ap.add_argument("--init", default=None, help="stage-1 checkpoint for stage 2")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    args.epochs = args.epochs or (150 if args.stage == 1 else 50)
    name = args.name or f"stage{args.stage}"
    out_dir = ROOT / "runs" / "pisttn" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    proc = ROOT / "processed"
    splits = json.loads((proc / "splits.json").read_text())["splits"]
    train_ds = ClipDataset(proc, splits["train"]["sequences"], stride=1, augment=True)
    val_ds = ClipDataset(proc, splits["val"]["sequences"], stride=args.val_stride)
    # batch size 1 as in the paper
    train_dl = torch.utils.data.DataLoader(train_ds, 1, shuffle=True, collate_fn=collate, num_workers=args.workers)
    val_dl = torch.utils.data.DataLoader(val_ds, 1, collate_fn=collate, num_workers=args.workers)

    model = PISTTN(with_detector=args.stage == 2).cuda()
    if args.init:
        sd = torch.load(args.init, map_location="cuda")["model"]
        missing, _ = model.load_state_dict(sd, strict=False)
        print("loaded", args.init, "| newly initialized:", sorted({k.split(".")[0] for k in missing}))
    if args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"train windows {len(train_ds)}, val windows {len(val_ds)}, params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, args {vars(args)}")

    best = float("inf")
    hist = []
    for ep in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_dl, args, opt)
        va = run_epoch(model, val_dl, args)
        hist.append(dict(epoch=ep, train=tr, val=va, sec=time.time() - t0))
        print(f"epoch {ep:3d} train {tr} | val {va} | {time.time() - t0:.0f}s", flush=True)
        ck = dict(model=model.state_dict(), epoch=ep, args=vars(args), val=va)
        torch.save(ck, out_dir / "last.pt")
        if va["loss"] < best:
            best = va["loss"]
            torch.save(ck, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
