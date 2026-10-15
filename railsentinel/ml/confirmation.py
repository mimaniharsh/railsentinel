"""Ticket confirmation predictor: will a WL/RAC booking end as CNF, RAC or WL at chart time?"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from railsentinel.ml.base import (ModelSpec, apply_probability_recalibration, disparity_ratio,
                                   fit_probability_recalibration, hgb_pipeline)
from railsentinel.ml.contracts import DataContract, FieldRule
from railsentinel.twin import generators
from railsentinel.twin.inventory import ARP_DAYS
from railsentinel.twin.seasons import SEASONS

STATUSES = ["CNF", "RAC", "WL"]


class ConfirmationPredictor(ModelSpec):
    name = "confirmation_predictor"
    title = "Ticket Confirmation Predictor"
    risk_tier = "high"
    description = ("Predicts the final status (CNF / RAC / WL) of a waitlisted or RAC ticket at chart "
                   "preparation, with the probability of confirmation.")
    categorical = ["train_category", "travel_class", "quota", "booking_status", "season"]
    numeric = ["queue_position", "days_to_departure", "distance_km", "route_demand", "hist_cancel_rate",
               "weekday", "month", "capacity"]
    label_field = "final_status"
    segment_field = "region"
    primary_metric = "accuracy"
    higher_is_better = True
    fairness_metric = "accuracy"
    explain_unit = "probability"
    default_train_size = 40000
    feature_labels = {
        "queue_position": "Waitlist / RAC position", "days_to_departure": "Days to departure",
        "booking_status": "Current status", "hist_cancel_rate": "Historical cancellation rate",
        "route_demand": "Route demand", "season": "Travel season", "quota": "Quota",
        "travel_class": "Class", "train_category": "Train type", "distance_km": "Distance (km)",
        "weekday": "Day of week", "month": "Month", "capacity": "Coach capacity",
    }
    contract = DataContract([
        FieldRule("train_category", "cat", allowed=("premium", "superfast", "express", "intercity"), feed="timetable_feed"),
        FieldRule("travel_class", "cat", allowed=("SL", "CC", "3A", "2A", "1A"), feed="booking_feed"),
        FieldRule("quota", "cat", allowed=("GN", "TQ"), feed="booking_feed"),
        FieldRule("booking_status", "cat", allowed=("WL", "RAC"), feed="booking_feed"),
        FieldRule("season", "cat", allowed=SEASONS, feed="calendar"),
        FieldRule("queue_position", minimum=1, maximum=2000, feed="booking_feed"),
        FieldRule("days_to_departure", minimum=0, maximum=ARP_DAYS, feed="booking_feed"),
        FieldRule("distance_km", minimum=1, maximum=4500, feed="timetable_feed"),
        FieldRule("route_demand", minimum=0, maximum=3, feed="demand_stats_feed"),
        FieldRule("hist_cancel_rate", minimum=0, maximum=1, feed="cancellation_stats_feed"),
        FieldRule("weekday", minimum=0, maximum=6, feed="calendar"),
        FieldRule("month", minimum=1, maximum=12, feed="calendar"),
        FieldRule("capacity", minimum=1, maximum=1000, feed="timetable_feed"),
    ])

    def generate(self, network, n, rng, world):
        return generators.confirmation_samples(network, n, rng, world)

    def train_models(self, X, records, seed, weights=None):
        y = np.array([r[self.label_field] for r in records])
        clf = hgb_pipeline(self.categorical, self.numeric,
                           HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                                          l2_regularization=0.1, random_state=seed))
        clf.fit(X[self.features], y, model__sample_weight=weights)
        return {"clf": clf}

    def _proba(self, artifact, df: pd.DataFrame) -> pd.DataFrame:
        clf = artifact["models"]["clf"]
        p = clf.predict_proba(df[self.features])
        out = pd.DataFrame(p, columns=list(clf.classes_))
        for s in STATUSES:
            if s not in out:
                out[s] = 0.0
        probs = apply_probability_recalibration(out[STATUSES].to_numpy(), STATUSES, artifact.get("recalibration"))
        return pd.DataFrame(probs, columns=STATUSES)

    def recalibrate(self, artifact, records):
        step = fit_probability_recalibration(self._proba(artifact, self.prepare(records)).to_numpy(), STATUSES,
                                             [r[self.label_field] for r in records])
        if step is None:
            return None
        new = copy.deepcopy(artifact)
        new["recalibration"] = list(artifact.get("recalibration") or []) + [step]
        return new

    def predict(self, artifact, df, context):
        probs = self._proba(artifact, df)
        results = []
        for _, row in probs.iterrows():
            status = str(row.idxmax())
            results.append({
                "p_confirm": round(float(row["CNF"]), 4),
                "p_at_least_rac": round(float(row["CNF"] + row["RAC"]), 4),
                "predicted_final_status": status,
                "class_probs": {s: round(float(row[s]), 4) for s in STATUSES},
                "confidence": round(float(row.max()), 4),
            })
        return results

    def explain_output(self, artifact, df):
        return self._proba(artifact, df)["CNF"].to_numpy()

    def metrics(self, outputs, labels, segments=None):
        if not outputs:
            return {}
        pred = np.array([o["predicted_final_status"] for o in outputs])
        y = np.array(labels)
        p_cnf = np.array([o["p_confirm"] for o in outputs])
        acc = float((pred == y).mean())
        brier = float(np.mean((p_cnf - (y == "CNF")) ** 2))
        m = {"accuracy": round(acc, 4), "brier_cnf": round(brier, 4), "n": int(len(y)),
             "confirm_rate_actual": round(float((y == "CNF").mean()), 4),
             "confirm_rate_predicted": round(float(p_cnf.mean()), 4)}
        if segments:
            by, counts = {}, {}
            seg = np.array(segments)
            for s in sorted(set(segments)):
                mask = seg == s
                if mask.sum() >= 20:
                    by[s] = round(float((pred[mask] == y[mask]).mean()), 4)
                    counts[s] = int(mask.sum())
            m["accuracy_by_segment"] = by
            m["n_by_segment"] = counts
            m["fairness_ratio"] = disparity_ratio(by)
        return m

    def sample_losses(self, outputs, labels):
        """Multi-class Brier score: squared error of the CNF/RAC/WL probabilities against the outcome."""
        probs = np.array([[o["class_probs"][s] for s in STATUSES] for o in outputs], dtype=float)
        onehot = np.array([[float(y == s) for s in STATUSES] for y in labels])
        return ((probs - onehot) ** 2).sum(axis=1)
