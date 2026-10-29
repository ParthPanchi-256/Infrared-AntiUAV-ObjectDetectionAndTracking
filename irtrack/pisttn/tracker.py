"""Segment-based tracking for PISTTN (Sec. II-D, Eqs. 10-11).

The sequence is covered by 4-frame segments that overlap by one frame. Within a segment,
per-frame detections are linked into short "3-D anchors" (tracklets). Consecutive
segments share a frame: anchors at the same place in that frame continue the same ID.
Anchors left unmatched are matched to recently lost tracks with Eq. 11:
    score = lambda * |trend_k - trend_i| + beta * |x_k - x_p|,  x_p = x_n + h * trend_i (Eq. 10)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class SegmentTracker:
    # gates are wide because camera shake moves targets up to ~30 px between frames,
    # and a frame holds only one or two targets
    def __init__(self, link_dist=15.0, overlap_dist=3.0, gate=20.0, lam=0.5, beta=0.5,
                 buffer=60, min_len=3):
        self.link_dist, self.overlap_dist, self.gate = link_dist, overlap_dist, gate
        self.lam, self.beta, self.buffer, self.min_len = lam, beta, buffer, min_len
        self.tracks = {}      # id -> dict(points={frame: (x, y)}, last=frame)
        self.next_id = 1

    @staticmethod
    def _trend(points):
        f = sorted(points)
        if len(f) < 2:
            return np.zeros(2)
        a, b = np.array(points[f[0]]), np.array(points[f[-1]])
        return (b - a) / (f[-1] - f[0])

    def _anchors(self, seg_dets):
        """seg_dets: {frame: array (K, >=2) of x, y, ...} for one segment -> list of {frame: (x, y)}."""
        frames = sorted(seg_dets)
        anchors = []
        for f in frames:
            dets = [tuple(d[:2]) for d in seg_dets[f]]
            open_ = [a for a in anchors if max(a) == f - 1]
            used = set()
            if open_ and dets:
                # constant-velocity prediction from the anchor's last two points
                pred = [np.array(a[f - 1]) + (np.array(a[f - 1]) - a[f - 2] if f - 2 in a else 0) for a in open_]
                cost = np.array([[np.hypot(p[0] - x, p[1] - y) for x, y in dets] for p in pred])
                for r, c in zip(*linear_sum_assignment(cost)):
                    if cost[r, c] <= self.link_dist:
                        open_[r][f] = dets[c]
                        used.add(c)
            anchors += [{f: d} for c, d in enumerate(dets) if c not in used]
        return anchors

    def update(self, seg_dets):
        anchors = self._anchors(seg_dets)
        if not anchors:
            return
        f0 = min(seg_dets)
        # 1) continuation through the shared frame
        live = [tid for tid, t in self.tracks.items() if f0 in t["points"]]
        cand = [a for a in anchors if f0 in a]
        taken_t, taken_a = set(), set()
        if live and cand:
            cost = np.array([[np.hypot(*(np.array(self.tracks[t]["points"][f0]) - a[f0])) for a in cand] for t in live])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= self.overlap_dist:
                    self._extend(live[r], cand[c]); taken_t.add(live[r]); taken_a.add(id(cand[c]))
        unmatched = [a for a in anchors if id(a) not in taken_a]
        # 2) motion-trend matching against lost tracks (Eqs. 10-11)
        lost = [t for t, tr in self.tracks.items() if t not in taken_t and 0 <= f0 - tr["last"] <= self.buffer]
        if lost and unmatched:
            cost = np.full((len(lost), len(unmatched)), 1e6)
            for i, t in enumerate(lost):
                tr = self.tracks[t]
                trend_i = self._trend({f: tr["points"][f] for f in sorted(tr["points"])[-4:]})
                for k, a in enumerate(unmatched):
                    fk = min(a)
                    if fk <= tr["last"]:
                        continue
                    h = fk - tr["last"]
                    xp = np.array(tr["points"][tr["last"]]) + h * trend_i            # Eq. 10
                    d = np.linalg.norm(np.array(a[fk]) - xp)
                    if d <= self.gate + 0.5 * h:
                        cost[i, k] = (self.lam * np.linalg.norm(self._trend(a) - trend_i)
                                      + self.beta * d)                                  # Eq. 11
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e6:
                    self._extend(lost[r], unmatched[c]); unmatched[c] = None
        for a in unmatched:
            if a is not None:
                self.tracks[self.next_id] = dict(points={}, last=-1)
                self._extend(self.next_id, a)
                self.next_id += 1

    def _extend(self, tid, anchor):
        tr = self.tracks[tid]
        for f, p in anchor.items():
            tr["points"].setdefault(f, p)
        tr["last"] = max(tr["points"])

    def results(self):
        """{frame: [(id, x, y)]} for tracks with at least min_len frames."""
        out = {}
        for tid, tr in self.tracks.items():
            if len(tr["points"]) < self.min_len:
                continue
            for f, (x, y) in tr["points"].items():
                out.setdefault(f, []).append((tid, float(x), float(y)))
        return out
