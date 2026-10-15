"""Fraud, security and bot detection for booking sessions.

A supervised gradient-boosting classifier separates known patterns (bots,
payment fraud, account misuse) from legitimate sessions. An Isolation Forest
trained on legitimate sessions adds an anomaly score that catches behaviour
the classifier has never seen. Reason codes compare the session with the
median legitimate session so every alert is explainable to an investigator.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from railsentinel.ml.base import (ModelSpec, apply_probability_recalibration, disparity_ratio,
                                   fit_probability_recalibration, hgb_pipeline)
from railsentinel.ml.contracts import DataContract, FieldRule
from railsentinel.twin import generators
from railsentinel.twin.generators import FRAUD_CLASSES

ANOMALY_WEIGHT = 0.15
REVIEW_BAND = 0.3

REASON_TEXT = {
    "requests_per_min": "request rate {v:.0f}/min vs typical {m:.0f}/min",
    "interarrival_cv": "machine-regular request timing (variation {v:.2f} vs typical {m:.2f})",
    "form_fill_seconds": "passenger form filled in {v:.1f}s vs typical {m:.0f}s",
    "ms_after_tatkal_open": "booked {v:.0f} ms after Tatkal opening",
    "accounts_on_device": "{v:.0f} accounts used from the same device",
    "accounts_on_payment_instrument": "payment instrument shared by {v:.0f} accounts",
    "account_age_days": "account only {v:.0f} days old",
    "failed_payments": "{v:.0f} failed payment attempts",
    "distinct_passenger_names_30d": "{v:.0f} different passenger names in 30 days",
    "ip_bookings_last_hour": "{v:.0f} bookings from this IP in the last hour",
    "headless_signals": "headless-browser signals present",
    "booking_attempts": "{v:.0f} booking attempts in one session",
    "passengers_per_booking": "{v:.0f} passengers per booking",
}
HIGHER_IS_RISKY = {"requests_per_min", "accounts_on_device", "accounts_on_payment_instrument", "failed_payments",
                   "distinct_passenger_names_30d", "ip_bookings_last_hour", "headless_signals", "booking_attempts",
                   "passengers_per_booking"}
LOWER_IS_RISKY = {"interarrival_cv", "form_fill_seconds", "account_age_days"}


class FraudDetector(ModelSpec):
    name = "fraud_detector"
    title = "Fraud, Security & Bot Detector"
    risk_tier = "high"
    description = ("Scores booking sessions for bulk-booking bots, payment fraud and account misuse, with "
                   "reason codes and an anomaly score for novel behaviour.")
    categorical: list[str] = []
    numeric = ["requests_per_min", "interarrival_cv", "form_fill_seconds", "headless_signals", "is_tatkal_window",
               "ms_after_tatkal_open", "booking_attempts", "passengers_per_booking", "accounts_on_device",
               "accounts_on_payment_instrument", "account_age_days", "failed_payments",
               "distinct_passenger_names_30d", "ip_bookings_last_hour"]
    label_field = "label"
    segment_field = "account_tenure"
    primary_metric = "recall"
    higher_is_better = True
    fairness_metric = "fpr"
    explain_unit = "risk"
    default_train_size = 30000
    feature_labels = {k: k.replace("_", " ") for k in numeric}
    contract = DataContract([
        FieldRule("requests_per_min", minimum=0, maximum=10000, feed="session_stream"),
        FieldRule("interarrival_cv", minimum=0, maximum=10, feed="session_stream"),
        FieldRule("form_fill_seconds", minimum=0, maximum=3600, feed="session_stream"),
        FieldRule("headless_signals", minimum=0, maximum=1, feed="session_stream"),
        FieldRule("is_tatkal_window", minimum=0, maximum=1, feed="session_stream"),
        FieldRule("ms_after_tatkal_open", minimum=-1, maximum=86_400_000, feed="session_stream"),
        FieldRule("booking_attempts", minimum=0, maximum=1000, feed="session_stream"),
        FieldRule("passengers_per_booking", minimum=1, maximum=6, feed="session_stream"),
        FieldRule("accounts_on_device", minimum=1, maximum=1000, feed="device_graph_feed"),
        FieldRule("accounts_on_payment_instrument", minimum=1, maximum=1000, feed="payments_feed"),
        FieldRule("account_age_days", minimum=0, maximum=20000, feed="account_feed"),
        FieldRule("failed_payments", minimum=0, maximum=1000, feed="payments_feed"),
        FieldRule("distinct_passenger_names_30d", minimum=0, maximum=10000, feed="account_feed"),
        FieldRule("ip_bookings_last_hour", minimum=0, maximum=100000, feed="session_stream"),
    ])

    def generate(self, network, n, rng, world):
        return generators.fraud_sessions(n, rng, world)

    def train_models(self, X, records, seed, weights=None):
        y = np.array([r[self.label_field] for r in records])
        clf = hgb_pipeline([], self.numeric, HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, class_weight="balanced", random_state=seed))
        clf.fit(X[self.features], y, model__sample_weight=weights)
        legit = X[self.features][y == "legit"]
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
        iso.fit(legit.to_numpy())
        raw = -iso.score_samples(legit.to_numpy())
        lo, hi = np.quantile(raw, 0.5), np.quantile(raw, 0.995)
        return {"clf": clf, "iso": iso, "iso_scale": (float(lo), float(hi))}

    def extra_artifact(self, train_records):
        legit = pd.DataFrame([r for r in train_records if r[self.label_field] == "legit"])
        return {"legit_medians": {c: float(legit[c].median()) for c in self.numeric}}

    def _scores(self, artifact, df):
        m = artifact["models"]
        X = df[self.features]
        clf = m["clf"]
        proba = pd.DataFrame(clf.predict_proba(X), columns=list(clf.classes_))
        for c in FRAUD_CLASSES:
            if c not in proba:
                proba[c] = 0.0
        proba = pd.DataFrame(apply_probability_recalibration(proba[list(FRAUD_CLASSES)].to_numpy(),
                                                             list(FRAUD_CLASSES), artifact.get("recalibration")),
                             columns=list(FRAUD_CLASSES))
        sup = 1 - proba["legit"].to_numpy()
        lo, hi = m["iso_scale"]
        raw = -m["iso"].score_samples(X.to_numpy())
        anom = np.clip((raw - lo) / max(hi - lo, 1e-9), 0, 1)
        risk = np.clip((1 - ANOMALY_WEIGHT) * sup + ANOMALY_WEIGHT * anom, 0, 1)
        return proba[list(FRAUD_CLASSES)], anom, risk

    def predict(self, artifact, df, context):
        threshold = float(context.get("threshold", artifact.get("threshold", 0.5)))
        proba, anom, risk = self._scores(artifact, df)
        med = artifact.get("legit_medians", {})
        out = []
        for i, row in enumerate(df.to_dict(orient="records")):
            p = proba.iloc[i]
            fraud_types = p.drop("legit")
            ptype = str(fraud_types.idxmax()) if risk[i] >= REVIEW_BAND else "legit"
            decision = ("block_and_review" if risk[i] >= threshold
                        else "step_up_verification" if risk[i] >= REVIEW_BAND else "allow")
            out.append({
                "risk_score": round(float(risk[i]), 4), "anomaly_score": round(float(anom[i]), 4),
                "predicted_type": ptype, "decision": decision, "threshold": threshold,
                "class_probs": {c: round(float(p[c]), 4) for c in FRAUD_CLASSES},
                "reason_codes": self.reason_codes(row, med) if risk[i] >= REVIEW_BAND else [],
                "confidence": round(float(max(risk[i], 1 - risk[i])), 4),
            })
        return out

    @staticmethod
    def reason_codes(row: dict, medians: dict, k: int = 4) -> list[str]:
        scored = []
        for f, template in REASON_TEXT.items():
            v, m = row.get(f), medians.get(f)
            if v is None or m is None:
                continue
            v = float(v)
            if f == "ms_after_tatkal_open":
                if v >= 0 and v < 3000:
                    scored.append((3.0, template.format(v=v, m=m)))
                continue
            if f in HIGHER_IS_RISKY and v > max(m, 1) * 2.5:
                scored.append((v / max(m, 1), template.format(v=v, m=m)))
            elif f in LOWER_IS_RISKY and m > 0 and v < m / 3:
                scored.append((m / max(v, 0.1), template.format(v=v, m=m)))
        scored.sort(reverse=True)
        return [t for _, t in scored[:k]]

    def explain_output(self, artifact, df):
        return self._scores(artifact, df)[2]

    def recalibrate(self, artifact, records):
        proba, _, _ = self._scores(artifact, self.prepare(records))
        step = fit_probability_recalibration(proba.to_numpy(), list(FRAUD_CLASSES),
                                             [r[self.label_field] for r in records])
        if step is None:
            return None
        new = copy.deepcopy(artifact)
        new["recalibration"] = list(artifact.get("recalibration") or []) + [step]
        return new

    def metrics(self, outputs, labels, segments=None):
        if not outputs:
            return {}
        y_fraud = np.array([lbl != "legit" for lbl in labels])
        flagged = np.array([o["decision"] == "block_and_review" for o in outputs])
        tp = int((flagged & y_fraud).sum())
        fp = int((flagged & ~y_fraud).sum())
        fn = int((~flagged & y_fraud).sum())
        tn = int((~flagged & ~y_fraud).sum())
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        fpr = fp / (fp + tn) if fp + tn else None
        m = {"recall": None if recall is None else round(recall, 4),
             "precision": None if precision is None else round(precision, 4),
             "fpr": None if fpr is None else round(fpr, 4),
             "fnr": None if recall is None else round(1 - recall, 4),
             "false_positives": fp, "false_negatives": fn, "n": int(len(labels)), "fraud_n": int(y_fraud.sum())}
        if segments:
            seg = np.array(segments)
            by = {}
            for s in sorted(set(segments)):
                mask = (seg == s) & ~y_fraud
                if mask.sum() >= 30:
                    by[s] = round(float(flagged[mask].mean()), 4)
            m["fpr_by_segment"] = by
            m["n_by_segment"] = {s: int(((seg == s) & ~y_fraud).sum()) for s in by}
            # for an error rate, parity compares the complement (share of genuine sessions allowed)
            m["fairness_ratio"] = disparity_ratio({k: 1 - v for k, v in by.items()})
        return m

    def sample_losses(self, outputs, labels):
        """Brier score of the risk score against the investigated outcome (fraud of any type = 1)."""
        risk = np.array([o["risk_score"] for o in outputs], dtype=float)
        y = np.array([lbl != "legit" for lbl in labels], dtype=float)
        return (risk - y) ** 2
