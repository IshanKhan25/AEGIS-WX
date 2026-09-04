from __future__ import annotations

import numpy as np


def _valid(pred, truth):
    """Return paired finite values, or empty arrays when inputs are unusable."""
    try:
        p, t = np.broadcast_arrays(np.asarray(pred, dtype=float), np.asarray(truth, dtype=float))
    except (TypeError, ValueError):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    return p[mask], t[mask]


def rmse(pred, truth):
    p, t = _valid(pred, truth)
    return float(np.sqrt(np.mean((p - t) ** 2))) if len(p) else float("nan")


def mae(pred, truth):
    p, t = _valid(pred, truth)
    return float(np.mean(np.abs(p - t))) if len(p) else float("nan")


def bias(pred, truth):
    p, t = _valid(pred, truth)
    return float(np.mean(p - t)) if len(p) else float("nan")


def correlation(pred, truth):
    p, t = _valid(pred, truth)
    if len(p) < 2 or np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    value = float(np.corrcoef(p, t)[0, 1])
    return value if np.isfinite(value) else float("nan")


def csi(pred, truth, threshold: float):
    p, t = _valid(pred, truth)
    if not len(p):
        return float("nan")
    p, t = p >= threshold, t >= threshold
    hits = np.sum(p & t)
    denominator = hits + np.sum(p & ~t) + np.sum(~p & t)
    # With no observed or forecast event, CSI has no meaningful denominator.
    return float(hits / denominator) if denominator else float("nan")


def roc_auc(pred, truth, threshold: float):
    scores, actual = _valid(pred, truth)
    y = (actual >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        value = float(roc_auc_score(y, scores))
    except ImportError:
        pos, neg = scores[y == 1], scores[y == 0]
        value = float((pos[:, None] > neg).mean() + .5 * (pos[:, None] == neg).mean())
    return value if np.isfinite(value) else float("nan")


def crps_ensemble(members, truth):
    try:
        m, y = np.asarray(members, dtype=float), np.asarray(truth, dtype=float)[:, None]
        term1 = np.mean(np.abs(m - y), axis=1)
        term2 = .5 * np.mean(np.abs(m[:, :, None] - m[:, None, :]), axis=(1, 2))
        value = float(np.mean(term1 - term2))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")
