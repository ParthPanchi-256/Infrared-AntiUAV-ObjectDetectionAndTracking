"""Point-based tracking metrics (CLEAR-MOT, IDF1, HOTA, detection P/R@tau).

GT boxes in this dataset are synthesized from center points, so all matching is done on
center distance rather than box IoU. Inputs are dicts {frame: [(id, x, y), ...]}.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _dist(g, p):
    if not g or not p:
        return np.zeros((len(g), len(p)))
    g = np.array([(x, y) for _, x, y in g], float)
    p = np.array([(x, y) for _, x, y in p], float)
    return np.linalg.norm(g[:, None] - p[None], axis=2)


def detection_pr(gt, pred, tau=5.0):
    tp = fp = fn = 0
    for f in set(gt) | set(pred):
        g, p = gt.get(f, []), pred.get(f, [])
        d = _dist(g, p)
        m = 0
        if d.size:
            r, c = linear_sum_assignment(d)
            m = int((d[r, c] <= tau).sum())
        tp += m; fp += len(p) - m; fn += len(g) - m
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(P=prec, R=rec, F1=2 * prec * rec / max(1e-9, prec + rec), TP=tp, FP=fp, FN=fn)


def clear_mot(gt, pred, tau=5.0):
    """MOTA/MOTP with the standard rule of keeping last frame's correspondences when still valid."""
    tp = fp = fn = idsw = 0
    dist_sum = 0.0
    last = {}  # gt id -> pred id of its last match
    n_gt = 0
    for f in sorted(set(gt) | set(pred)):
        g, p = gt.get(f, []), pred.get(f, [])
        n_gt += len(g)
        d = _dist(g, p)
        matched = []
        if d.size:
            cost = d.copy()
            cost[cost > tau] = 1e6
            # prefer continuing correspondences
            for i, (gid, _, _) in enumerate(g):
                for j, (pid, _, _) in enumerate(p):
                    if last.get(gid) == pid and d[i, j] <= tau:
                        cost[i, j] -= 1e3
            r, c = linear_sum_assignment(cost)
            matched = [(i, j) for i, j in zip(r, c) if d[i, j] <= tau]
        for i, j in matched:
            gid, pid = g[i][0], p[j][0]
            if gid in last and last[gid] != pid:
                idsw += 1
            last[gid] = pid
            dist_sum += d[i, j]
        tp += len(matched); fp += len(p) - len(matched); fn += len(g) - len(matched)
    return dict(MOTA=1 - (fn + fp + idsw) / max(1, n_gt), MOTP=dist_sum / max(1, tp),
                IDSW=idsw, FP=fp, FN=fn, TP=tp)


def idf1(gt, pred, tau=5.0):
    gids = sorted({i for v in gt.values() for i, _, _ in v})
    pids = sorted({i for v in pred.values() for i, _, _ in v})
    gi, pi = {k: n for n, k in enumerate(gids)}, {k: n for n, k in enumerate(pids)}
    overlap = np.zeros((len(gids), len(pids)))
    for f in set(gt) & set(pred):
        g, p = gt[f], pred[f]
        d = _dist(g, p)
        for a in range(len(g)):
            for b in range(len(p)):
                if d[a, b] <= tau:
                    overlap[gi[g[a][0]], pi[p[b][0]]] += 1
    n_g = sum(len(v) for v in gt.values())
    n_p = sum(len(v) for v in pred.values())
    idtp = 0.0
    if overlap.size:
        r, c = linear_sum_assignment(-overlap)
        idtp = overlap[r, c].sum()
    return dict(IDF1=2 * idtp / max(1, n_g + n_p), IDP=idtp / max(1, n_p), IDR=idtp / max(1, n_g))


def hota(gt, pred, sim_radius=10.0):
    """HOTA (Luiten et al. 2021, TrackEval algorithm) with similarity = max(0, 1 - d / sim_radius)."""
    alphas = np.arange(0.05, 0.99, 0.05)
    gids = sorted({i for v in gt.values() for i, _, _ in v})
    pids = sorted({i for v in pred.values() for i, _, _ in v})
    gi, pi = {k: n for n, k in enumerate(gids)}, {k: n for n, k in enumerate(pids)}
    frames = sorted(set(gt) | set(pred))

    def sim(g, p):
        return np.clip(1 - _dist(g, p) / sim_radius, 0, None)

    pot = np.zeros((len(gids), len(pids)))
    g_cnt = np.zeros((len(gids), 1)); p_cnt = np.zeros((1, len(pids)))
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        gx = [gi[i] for i, _, _ in g]; px = [pi[i] for i, _, _ in p]
        if g and p:
            s = sim(g, p)
            den = s.sum(0)[None] + s.sum(1)[:, None] - s
            s_iou = np.where(den > 1e-10, s / np.maximum(den, 1e-10), 0)
            pot[np.ix_(gx, px)] += s_iou
        g_cnt[gx] += 1; p_cnt[0, px] += 1
    glob = pot / np.maximum(1e-10, g_cnt + p_cnt - pot)

    na = len(alphas)
    TP, FN, FP, LocA = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    mc = [np.zeros_like(pot) for _ in alphas]
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        if not g:
            FP += len(p); continue
        if not p:
            FN += len(g); continue
        gx = np.array([gi[i] for i, _, _ in g]); px = np.array([pi[i] for i, _, _ in p])
        s = sim(g, p)
        r, c = linear_sum_assignment(-(glob[np.ix_(gx, px)] * s))
        for a, alpha in enumerate(alphas):
            ok = s[r, c] >= alpha - 1e-10
            rr, cc = r[ok], c[ok]
            TP[a] += len(rr); FN[a] += len(g) - len(rr); FP[a] += len(p) - len(rr)
            LocA[a] += s[rr, cc].sum()
            mc[a][gx[rr], px[cc]] += 1
    AssA = np.zeros(na)
    for a in range(na):
        m = mc[a]
        AssA[a] = (m * (m / np.maximum(1, g_cnt + p_cnt - m))).sum() / max(1, TP[a])
    DetA = TP / np.maximum(1, TP + FN + FP)
    H = np.sqrt(DetA * AssA)
    return dict(HOTA=float(H.mean()), DetA=float(DetA.mean()), AssA=float(AssA.mean()),
                LocA=float((LocA / np.maximum(1e-10, TP)).mean()))


def evaluate(gt, pred, tau=5.0):
    out = {}
    out.update(hota(gt, pred, sim_radius=2 * tau))
    out.update(clear_mot(gt, pred, tau))
    out.update(idf1(gt, pred, tau))
    det = detection_pr(gt, pred, tau)
    out.update({f"P@{tau:g}": det["P"], f"R@{tau:g}": det["R"], f"F1@{tau:g}": det["F1"]})
    return out


def load_centers(path):
    """Read 'frame,id,x,y' lines (0-based frames) into {frame: [(id, x, y)]}."""
    d = {}
    for line in open(path):
        if line.strip():
            f, i, x, y = line.strip().split(",")[:4]
            d.setdefault(int(f), []).append((int(i), float(x), float(y)))
    return d
