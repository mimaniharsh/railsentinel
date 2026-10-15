"""Model-agnostic local explanations by Monte Carlo estimation of Shapley values.

For each sampled permutation of the features and each sampled background row,
the estimator walks from the background row to the explained row, switching one
feature at a time in permutation order, and credits each feature with the change
in model output it causes. Averaging over samples estimates each feature's
Shapley value. Within every walk the credits add up exactly to
f(x) - f(background), so the returned contributions sum to
prediction - base_value, where base_value is the mean output over the sampled
background rows.

All walks are evaluated in one batched model call, so explaining a prediction
costs a single `predict` on n_samples * (n_features + 1) rows.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def shapley_explain(predict_fn: Callable[[pd.DataFrame], np.ndarray], x: pd.DataFrame, background: pd.DataFrame,
                    features: list[str], n_samples: int = 48, seed: int = 0) -> dict:
    if len(x) != 1:
        raise ValueError("explain one row at a time")
    rng = np.random.default_rng(seed)
    n_feat = len(features)
    bg_idx = rng.integers(0, len(background), size=n_samples)
    perms = np.array([rng.permutation(n_feat) for _ in range(n_samples)])
    # position of each feature inside its permutation: pos[m, f]
    pos = np.empty_like(perms)
    for m in range(n_samples):
        pos[m, perms[m]] = np.arange(n_feat)
    steps = n_feat + 1
    total = n_samples * steps
    k = np.tile(np.arange(steps), n_samples)  # step index within each walk
    m_of_row = np.repeat(np.arange(n_samples), steps)
    columns = {}
    xrow = x.iloc[0]
    for f_i, f in enumerate(features):
        bg_vals = background[f].to_numpy()[bg_idx][m_of_row]
        use_x = pos[m_of_row, f_i] < k
        col = np.where(use_x, np.full(total, xrow[f], dtype=object), bg_vals.astype(object))
        columns[f] = col
    frame = pd.DataFrame(columns, columns=features)
    for f in features:
        if pd.api.types.is_numeric_dtype(background[f]):
            frame[f] = pd.to_numeric(frame[f])
        else:
            frame[f] = frame[f].astype(str)
    preds = np.asarray(predict_fn(frame), dtype=float).reshape(n_samples, steps)
    deltas = np.diff(preds, axis=1)  # (n_samples, n_feat) in permutation order
    contrib = np.zeros(n_feat)
    for m in range(n_samples):
        contrib[perms[m]] += deltas[m]
    contrib /= n_samples
    base = float(preds[:, 0].mean())
    prediction = float(np.asarray(predict_fn(x[features]), dtype=float)[0])
    return {"base_value": base, "prediction": prediction,
            "contributions": {f: float(c) for f, c in zip(features, contrib)}}


def top_factors(explanation: dict, x: dict, labels: dict[str, str], k: int = 5, unit: str = "") -> list[dict]:
    items = sorted(explanation["contributions"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    out = []
    for feat, val in items:
        value = x.get(feat)
        out.append({"feature": feat, "label": labels.get(feat, feat), "value": value,
                    "impact": round(val, 4), "direction": "raises" if val > 0 else "lowers",
                    "unit": unit})
    return out
