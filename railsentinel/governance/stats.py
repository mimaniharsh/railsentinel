"""Statistical tests used by the monitors.

* Population Stability Index (PSI) for numeric and categorical drift. The
  0.1 / 0.25 warning and alert levels are a widely used rule of thumb.
* Two-sample Kolmogorov-Smirnov test (numeric) and chi-square test of homogeneity
  (categorical), both from scipy, as the significance half of a drift alert: PSI
  measures how big a shift is, the test says whether it is distinguishable from
  sampling noise. Small or clustered windows can show a large PSI by chance.
* Page-Hinkley test for a sustained increase in a stream of errors (concept drift).
* CUSUM change-point estimate (argmax of |S_k - k * mean|), the maximum-likelihood
  location of a single shift in mean, used to decide which data is "after the change".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # nearly constant reference: compare share equal to the mode vs not
        mode = edges[0]
        p = np.array([np.mean(ref == mode), np.mean(ref != mode)]) + EPS
        q = np.array([np.mean(cur == mode), np.mean(cur != mode)]) + EPS
        return float(np.sum((q - p) * np.log(q / p)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + EPS
    q = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    cats = sorted(set(ref.index) | set(cur.index))
    p = np.array([ref.get(c, 0.0) for c in cats]) + EPS
    q = np.array([cur.get(c, 0.0) for c in cats]) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return 1.0
    return float(stats.ks_2samp(ref, cur).pvalue)


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(str).value_counts()
    cur = current.astype(str).value_counts()
    cats = sorted(set(ref.index) | set(cur.index))
    if len(cats) < 2 or len(reference) < 10 or len(current) < 10:
        return 1.0
    table = np.array([[ref.get(c, 0) for c in cats], [cur.get(c, 0) for c in cats]])
    return float(stats.chi2_contingency(table)[1])


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, categorical: list[str],
                 numeric: list[str]) -> dict[str, dict]:
    out = {}
    for c in categorical:
        if c in reference and c in current:
            out[c] = {"psi": round(psi_categorical(reference[c], current[c]), 4),
                      "chi2_p": round(chi2_pvalue(reference[c], current[c]), 5), "type": "categorical"}
    for c in numeric:
        if c in reference and c in current:
            r = pd.to_numeric(reference[c], errors="coerce").to_numpy()
            k = pd.to_numeric(current[c], errors="coerce").to_numpy()
            out[c] = {"psi": round(psi_numeric(r, k), 4), "ks_p": round(ks_pvalue(r, k), 5), "type": "numeric",
                      "ref_mean": round(float(np.nanmean(r)), 4) if len(r) else None,
                      "cur_mean": round(float(np.nanmean(k)), 4) if len(k) else None}
    return out


def page_hinkley(errors: list[float], delta: float = 0.005, threshold: float = 5.0) -> dict:
    """Detect an upward shift in the mean of `errors` (e.g. 0/1 misclassifications)."""
    if len(errors) < 30:
        return {"drift": False, "statistic": 0.0, "change_index": None, "change_point": None}
    mean = 0.0
    cum = 0.0
    min_cum = 0.0
    min_at = 0
    for i, e in enumerate(errors, start=1):
        mean += (e - mean) / i
        cum += e - mean - delta
        if cum < min_cum:
            min_cum, min_at = cum, i
        if cum - min_cum > threshold:
            # detection happens after the change; the minimum of the cumulative sum estimates when it began
            return {"drift": True, "statistic": round(cum - min_cum, 3), "change_index": i, "change_point": min_at}
    return {"drift": False, "statistic": round(cum - min_cum, 3), "change_index": None, "change_point": None}


def change_point(errors: list[float], min_segment: int = 20) -> int | None:
    """Index where the error rate most plausibly shifted (None if the stream is too short)."""
    e = np.asarray(errors, dtype=float)
    n = len(e)
    if n < 2 * min_segment:
        return None
    s = np.cumsum(e) - np.arange(1, n + 1) * e.mean()
    k = int(np.argmax(np.abs(s[min_segment - 1:n - min_segment]))) + min_segment
    return k
