"""Turn PISTTN detector outputs into per-frame boxes, and compute training targets for them."""
import torch
from torchvision.ops import nms

BASE_SIZE = 5.0  # size channels predict log(w / BASE_SIZE), so a zero output is a 5 px box


def decode_boxes(hh_indices, det, stride=1):
    """-> per detector voxel: (b, t, cx, cy, w, h, vx, vy) with t in clip coordinates."""
    src, seg = det["src"][:, 0], det["src"][:, 1]
    raw = det["raw"]
    idx = hh_indices[src].long()
    cx = idx[:, 3].float() + 0.5 + raw[:, 0]
    cy = idx[:, 2].float() + 0.5 + raw[:, 1]
    w = BASE_SIZE * torch.exp(raw[:, 2].clamp(-3, 3))
    h = BASE_SIZE * torch.exp(raw[:, 3].clamp(-3, 3))
    return dict(b=idx[:, 0], t=idx[:, 1], seg=seg, src=src,
                box=torch.stack([cx, cy, w, h], 1), vel=raw[:, 4:6])


def frame_detections(dec, score, score_thr=0.5, nms_iou=0.3, max_det=10):
    """Group decoded voxel boxes by (batch, segment, frame), threshold on query score, NMS.

    Returns {(b, seg, t): tensor (K, 7) of cx, cy, w, h, score, vx, vy}.
    """
    keep = score >= score_thr
    out = {}
    if not keep.any():
        return out
    b, s, t = dec["b"][keep], dec["seg"][keep], dec["t"][keep]
    box, sc, vel = dec["box"][keep], score[keep], dec["vel"][keep]
    key = (b * 1000 + s * 100 + t)
    for k in key.unique().tolist():
        m = key == k
        bb, ss, vv = box[m], sc[m], vel[m]
        xyxy = torch.cat([bb[:, :2] - bb[:, 2:] / 2, bb[:, :2] + bb[:, 2:] / 2], 1)
        sel = nms(xyxy, ss, nms_iou)[:max_det]
        out[(k // 1000, (k % 1000) // 100, k % 100)] = torch.cat([bb[sel], ss[sel, None], vv[sel]], 1)
    return out
