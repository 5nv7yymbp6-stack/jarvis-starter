"""Pure evaluation math: no I/O, no models. Scores are similarity (higher =
more like the enrolled speaker); YUSUF are the positives, everything else
the negatives (impostors)."""
import math

import numpy as np

BUCKETS = ("<2s", "2-5s", ">=5s")


def bucket(duration_s: float) -> str:
    if duration_s < 2.0:
        return "<2s"
    if duration_s < 5.0:
        return "2-5s"
    return ">=5s"


def far_frr(pos, neg, thr: float) -> tuple[float | None, float | None]:
    """(FAR, FRR) at `thr`. FAR = impostors accepted; FRR = Yusuf rejected.
    None when a side is empty (no invented numbers)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    far = float(np.mean(neg >= thr)) if len(neg) else None
    frr = float(np.mean(pos < thr)) if len(pos) else None
    return far, frr


def roc(pos, neg) -> list[tuple[float, float, float]]:
    """[(threshold, FAR, FRR)] over every distinct score, ascending."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return []
    pts = sorted(set(pos.tolist()) | set(neg.tolist()))
    pts = [pts[0] - 1e-6] + pts + [pts[-1] + 1e-6]
    out = []
    for t in pts:
        far, frr = far_frr(pos, neg, t)
        out.append((float(t), far, frr))
    return out


def eer(pos, neg) -> tuple[float, float] | None:
    """(equal error rate, threshold) or None if a side is empty. EER is the
    point where FAR and FRR are closest; with few samples it is coarse."""
    curve = roc(pos, neg)
    if not curve:
        return None
    t, far, frr = min(curve, key=lambda p: (abs(p[1] - p[2]), p[0]))
    return (far + frr) / 2, t


def d_prime(pos, neg) -> float | None:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return None
    v = (pos.var(ddof=1) + neg.var(ddof=1)) / 2
    return float((pos.mean() - neg.mean()) / math.sqrt(v)) if v > 0 else None


def overlap(pos, neg, bins: int = 20) -> float | None:
    """Histogram overlap coefficient in [0,1]: 0 = fully separated,
    1 = identical distributions."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    if hi == lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hp = np.histogram(pos, edges)[0] / len(pos)
    hn = np.histogram(neg, edges)[0] / len(neg)
    return float(np.minimum(hp, hn).sum())


def describe(scores) -> dict:
    s = np.asarray(scores, float)
    if not len(s):
        return {"n": 0}
    return {"n": int(len(s)), "mean": float(s.mean()),
            "p10": float(np.percentile(s, 10)), "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)), "min": float(s.min()),
            "max": float(s.max())}


def threshold_table(pos, neg, thresholds) -> list[dict]:
    rows = []
    for t in thresholds:
        far, frr = far_frr(pos, neg, t)
        rows.append({"threshold": float(t), "far": far, "frr": frr})
    return rows


def split_by_bucket(items) -> dict:
    """items: iterable of (duration_s, score) -> {bucket: [scores]}."""
    out = {b: [] for b in BUCKETS}
    for dur, sc in items:
        out[bucket(dur)].append(sc)
    return out


def report_block(pos_items, neg_items, thresholds) -> dict:
    """Everything for one (positives, negatives) slice. *_items are lists of
    (duration_s, score)."""
    pos = [s for _, s in pos_items]
    neg = [s for _, s in neg_items]
    e = eer(pos, neg)
    return {"pos": describe(pos), "neg": describe(neg),
            "eer": e[0] if e else None, "eer_threshold": e[1] if e else None,
            "d_prime": d_prime(pos, neg), "overlap": overlap(pos, neg),
            "table": threshold_table(pos, neg, thresholds)}
