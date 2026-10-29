"""Unit tests for the core library on synthetic inputs (no dataset needed).

  python tests/test_core.py        # plain runner
  python -m pytest tests           # if pytest is installed
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from irtrack.boxes import blob_extent, clip_xyxy  # noqa: E402
from irtrack.labels import parse_label_file  # noqa: E402
from irtrack.metrics import detection_pr, evaluate  # noqa: E402
from irtrack.pisttn.tracker import SegmentTracker  # noqa: E402


# ---------------------------------------------------------------- labels
def test_parse_label_file_handles_quirks():
    # CRLF, an empty header field (extra tab), two objects, an empty frame, a trailing line with no image
    text = ("0\tdata9\t\t3\t2\t\r\n"
            "frame:0\t2\tobject:1\t10\t20\tobject:2\t30\t40\r\n"
            "frame:1\t0\t\t\t\r\n"
            "frame:2\t1\tobject:1\t11\t21\r\n"
            "frame:3\t\t\t\t\r\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "data9.txt"
        p.write_bytes(text.encode())
        seq = parse_label_file(p, num_frames=3)
    assert seq.header_name == "data9" and seq.num_objects == 2
    assert seq.frames[0] == [(1, 10, 20), (2, 30, 40)]
    assert seq.frames[1] == []
    assert seq.frames[2] == [(1, 11, 21)]
    assert 3 not in seq.frames  # label line without an image is dropped


# ---------------------------------------------------------------- box synthesis
def _blob_image(sigma, x=64, y=64, amp=120.0, bg=80.0):
    yy, xx = np.mgrid[0:128, 0:128]
    img = bg + amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img.clip(0, 255).astype(np.uint8)


def test_blob_extent_grows_with_target_size():
    small = blob_extent(_blob_image(0.6), 64, 64)
    large = blob_extent(_blob_image(4.0), 64, 64)
    assert 1 <= small[0] <= 3, small
    assert large[0] >= 7, large  # FWHM of sigma=4 is ~9.4 px


def test_blob_extent_flat_patch_is_zero():
    assert blob_extent(np.full((64, 64), 100, np.uint8), 32, 32) == (0, 0)


def test_clip_xyxy_at_border():
    assert clip_xyxy(1.0, 1.0, 6, 6) == (0.0, 0.0, 4.0, 4.0)
    assert clip_xyxy(100.0, 100.0, 6, 6) == (97.0, 97.0, 103.0, 103.0)


# ---------------------------------------------------------------- metrics
def _track(n=50, ids=(1,), offset=0.0):
    return {f: [(i, 20.0 + f + 40 * k + offset, 30.0) for k, i in enumerate(ids)] for f in range(n)}


def test_metrics_perfect():
    gt = _track(ids=(1, 2))
    r = evaluate(gt, gt)
    for k in ("HOTA", "MOTA", "IDF1", "P@5", "R@5"):
        assert abs(r[k] - 1.0) < 1e-9, (k, r[k])
    assert r["IDSW"] == 0


def test_metrics_empty_prediction():
    r = evaluate(_track(), {})
    assert r["MOTA"] == 0 and r["IDF1"] == 0 and r["HOTA"] == 0 and r["FN"] == 50


def test_metrics_id_switch():
    gt = _track(n=100)
    pred = {f: [(1 if f < 50 else 2, x, y) for _, x, y in v] for f, v in gt.items()}
    r = evaluate(gt, pred)
    assert r["IDSW"] == 1
    assert abs(r["IDF1"] - 0.5) < 1e-9
    assert r["MOTA"] > 0.98 and r["HOTA"] < 0.8


def test_center_distance_threshold():
    gt = _track()
    assert detection_pr(gt, _track(offset=4.9))["R"] == 1.0
    assert detection_pr(gt, _track(offset=5.1))["R"] == 0.0


# ---------------------------------------------------------------- PISTTN segment tracker
def test_segment_tracker_keeps_identity_through_jitter():
    rng = np.random.default_rng(0)
    gt = {f: [(1, 50 + 2.0 * f + rng.normal(0, 1.5), 60.0)] for f in range(40)}
    trk = SegmentTracker()
    for f0 in range(0, 37, 3):
        trk.update({f: np.array([[x, y] for _, x, y in gt.get(f, [])]).reshape(-1, 2) for f in range(f0, f0 + 4)})
    r = evaluate(gt, trk.results())
    assert r["IDSW"] == 0 and r["IDF1"] > 0.99, r


# ---------------------------------------------------------------- PISTTN sparsification (GPU optional)
def test_tsvd_keeps_moving_target_and_drops_static_background():
    import torch
    from irtrack.pisttn.sparsify import tsvd_sparsify

    clip = torch.full((10, 32, 32), 100.0)
    clip[:, 5, 5] = 200.0                     # static bright pixel: background
    for t in range(10):
        clip[t, 20, 5 + 2 * t] = 220.0        # moving target
    coords, _ = tsvd_sparsify(clip, thresh=20)
    active = {tuple(c) for c in coords.tolist()}
    assert all((t, 20, 5 + 2 * t) in active for t in range(10))
    assert not any(c[1:] == (5, 5) for c in active)


def _run_all():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
