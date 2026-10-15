"""Common machinery for the governed models.

A `ModelSpec` describes one model end to end: inputs and their data contract,
how the twin generates training data, how to train, predict, explain and
evaluate, and which upstream feed owns each input (used by root-cause analysis).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from railsentinel.ml.contracts import DataContract


def hgb_pipeline(categorical: list[str], numeric: list[str], estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1, dtype=float), categorical),
         ("num", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )
    if isinstance(estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)) and categorical:
        estimator.set_params(categorical_features=list(range(len(categorical))))
    return Pipeline([("pre", pre), ("model", estimator)])


PROB_FLOOR = 1e-6


def fit_probability_recalibration(probs: np.ndarray, classes: list[str], labels: list) -> dict | None:
    """Multinomial logistic regression on the log-probabilities of an existing classifier.

    Adapting a strong model to a changed world this way needs far fewer labels than
    retraining: the model's learned structure is kept and only its output mapping moves."""
    y = np.asarray(labels).astype(str)
    if len(set(y)) < 2:
        return None
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(np.log(np.clip(probs, PROB_FLOOR, 1.0)), y)
    return {"kind": "multinomial_logit_on_log_probs", "classes": list(classes), "model": lr}


def apply_probability_recalibration(probs: np.ndarray, classes: list[str], steps: list[dict]) -> np.ndarray:
    """Apply recalibration steps in order; classes a step never saw get probability 0."""
    out = np.asarray(probs, dtype=float)
    for step in steps or []:
        lr = step["model"]
        raw = lr.predict_proba(np.log(np.clip(out, PROB_FLOOR, 1.0)))
        mapped = np.zeros_like(out)
        for j, c in enumerate(lr.classes_):
            if c in classes:
                mapped[:, classes.index(c)] = raw[:, j]
        out = mapped
    return out


def fit_affine_recalibration(predicted: np.ndarray, actual: np.ndarray) -> dict | None:
    """Least-squares map actual ~ a + b * predicted, used to shift and rescale regression quantiles."""
    x, y = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if len(x) < 20 or float(np.std(x)) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    return {"kind": "affine", "a": float(a), "b": float(np.clip(b, 0.2, 5.0))}


def apply_affine_recalibration(values: np.ndarray, steps: list[dict]) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    for step in steps or []:
        out = step["a"] + step["b"] * out
    return out


def disparity_ratio(values: dict[str, float]) -> float | None:
    """Smallest group value divided by the largest (1.0 = parity, lower = larger gap).

    The ratio is symmetric, so it reads the same way for "higher is better" metrics
    (accuracy) and for error metrics (MAE): 1.0 means every group is served equally."""
    vals = [v for v in values.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    lo, hi = min(vals), max(vals)
    return float(lo / hi) if hi > 0 else None


class ModelSpec(ABC):
    name: str = ""
    title: str = ""
    kind: str = "ml"
    owner: str = "data-science"
    risk_tier: str = "medium"
    description: str = ""
    categorical: list[str] = []
    numeric: list[str] = []
    label_field: str = ""
    segment_field: str | None = None
    feature_labels: dict[str, str] = {}
    primary_metric: str = ""
    higher_is_better: bool = True
    explain_unit: str = ""
    contract: DataContract = DataContract([])
    fairness_metric: str | None = None
    default_train_size: int = 20000

    # ---------------- data
    @property
    def features(self) -> list[str]:
        return list(self.categorical) + list(self.numeric)

    @abstractmethod
    def generate(self, network, n: int, rng: np.random.Generator, world) -> Any:
        """Draw `n` labelled records from the digital twin under `world`."""

    def records_to_frame(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for c in self.categorical:
            df[c] = df[c].astype(str)
        for c in self.numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def frame_to_records(self, data: Any) -> list[dict]:
        return data.to_dict(orient="records")

    def labels_from(self, data: Any) -> list[Any]:
        return list(data[self.label_field])

    def segment_of(self, record: dict) -> str | None:
        return str(record.get(self.segment_field)) if self.segment_field else None

    def feed_of(self, feature: str) -> str:
        return self.contract.feeds().get(feature, "unknown")

    # ---------------- model lifecycle
    def train_models(self, X: pd.DataFrame, records: list[dict], seed: int,
                     weights: np.ndarray | None = None) -> dict:
        """Fit the estimators used by the default `fit`. Specs that override `fit` do not need this."""
        raise TypeError(f"{type(self).__name__} must implement train_models() or override fit()")

    def fit(self, data: Any, seed: int = 0, sample_weight: np.ndarray | None = None) -> dict:
        """Train on `data`, hold out 20% for baseline metrics and keep a reference
        sample of inputs for drift detection."""
        rng = np.random.default_rng(seed)
        records = self.frame_to_records(data) if isinstance(data, pd.DataFrame) else list(data)
        idx = rng.permutation(len(records))
        cut = int(len(records) * 0.8)
        train_idx, test_idx = idx[:cut], idx[cut:]
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        weights = None if sample_weight is None else np.asarray(sample_weight)[train_idx]
        artifact = {"spec": self.name, "features": self.features,
                    "models": self.train_models(self._frame_for_training(train_records), train_records, seed,
                                                weights)}
        artifact.update(self.extra_artifact(train_records))
        test_outputs = self.predict(artifact, self._frame_for_training(test_records), {})
        artifact["baseline_metrics"] = self.metrics(test_outputs, [self.label_of(r) for r in test_records],
                                                    [self.segment_of(r) for r in test_records])
        ref = self.drift_frame(train_records)
        artifact["reference"] = ref.sample(n=min(len(ref), 3000), random_state=seed).reset_index(drop=True)
        artifact["train_rows"] = len(train_records)
        return artifact

    def _frame_for_training(self, records: list[dict]) -> pd.DataFrame:
        return self.records_to_frame(records)

    def extra_artifact(self, train_records: list[dict]) -> dict:
        return {}

    @abstractmethod
    def predict(self, artifact: dict, df: pd.DataFrame, context: dict) -> list[dict]:
        """Score prepared inputs and return one output dict per record."""

    @abstractmethod
    def explain_output(self, artifact: dict, df: pd.DataFrame) -> np.ndarray:
        """Scalar output that explanations decompose (probability, minutes, ...)."""

    def label_of(self, record: dict) -> Any:
        return record.get(self.label_field)

    @abstractmethod
    def metrics(self, outputs: list[dict], labels: list, segments: list | None = None) -> dict:
        """Quality metrics of `outputs` against observed `labels`, optionally per segment."""

    @abstractmethod
    def sample_losses(self, outputs: list[dict], labels: list) -> np.ndarray:
        """Per-record loss under a proper scoring rule (lower is better).

        Used for paired comparisons between two model versions scored on the same
        records: a per-record loss uses every record's full output (probabilities,
        quantiles), so it separates two models with far fewer samples than a
        difference in accuracy or recall does."""

    def drift_frame(self, records: list[dict]) -> pd.DataFrame:
        return self.records_to_frame(records)[self.features]

    def drift_columns(self) -> tuple[list[str], list[str]]:
        return list(self.categorical), list(self.numeric)

    def explanation_background(self, artifact: dict) -> pd.DataFrame:
        return artifact["reference"]

    def recalibrate(self, artifact: dict, records: list[dict]) -> dict | None:
        """Return a copy of `artifact` whose outputs are recalibrated on `records` (labelled), or None
        when the model type has no recalibration (the lifecycle then only considers retraining)."""
        return None

    # ---------------- runtime hooks
    supports_shapley: bool = True

    def validation_record(self, record: dict) -> dict:
        """What the data contract validates (the raw request for tabular models)."""
        return record

    def prepare(self, records: list[dict]) -> Any:
        return self.records_to_frame(records)

    def log_features(self, record: dict) -> dict:
        """What is stored with each prediction: everything needed to retrain later."""
        return {k: v for k, v in record.items() if k != self.label_field}

    def record_from_log(self, features: dict, label: Any) -> dict:
        rec = dict(features)
        rec[self.label_field] = label
        return rec
