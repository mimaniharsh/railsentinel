"""Train delay predictor: arrival delay at a station with a p10-p90 interval."""
from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from railsentinel.ml.base import (ModelSpec, apply_affine_recalibration, disparity_ratio,
                                   fit_affine_recalibration, hgb_pipeline)
from railsentinel.ml.contracts import DataContract, FieldRule
from railsentinel.twin import generators

REGIONS = ("North", "West", "Central", "East", "South", "Northeast")


class DelayPredictor(ModelSpec):
    name = "delay_predictor"
    title = "Train Delay & Disruption Predictor"
    risk_tier = "medium"
    description = ("Predicts arrival delay (minutes) at a target station with an 80% prediction interval, "
                   "from schedule position, live running delay, weather and station load.")
    categorical = ["train_category", "region"]
    numeric = ["month", "weekday", "sched_hour", "stop_index", "frac_route", "distance_ahead_km", "stops_ahead",
               "known_delay_min", "journey_started", "weather_severity", "station_load", "hist_mean_delay"]
    label_field = "delay_min"
    segment_field = "region"
    primary_metric = "mae"
    higher_is_better = False
    fairness_metric = None
    explain_unit = "minutes"
    default_train_size = 40000
    feature_labels = {
        "known_delay_min": "Current running delay", "weather_severity": "Weather severity",
        "distance_ahead_km": "Distance still to run", "stops_ahead": "Stops still to run",
        "station_load": "Station congestion", "hist_mean_delay": "Train's usual delay",
        "train_category": "Train type", "journey_started": "Train has started", "sched_hour": "Scheduled hour",
        "frac_route": "Position along route", "month": "Month", "region": "Region", "weekday": "Day of week",
        "stop_index": "Stop number",
    }
    contract = DataContract([
        FieldRule("train_category", "cat", allowed=("premium", "superfast", "express", "intercity"), feed="timetable_feed"),
        FieldRule("region", "cat", allowed=REGIONS, feed="timetable_feed"),
        FieldRule("month", minimum=1, maximum=12, feed="calendar"),
        FieldRule("weekday", minimum=0, maximum=6, feed="calendar"),
        FieldRule("sched_hour", minimum=0, maximum=23, feed="timetable_feed"),
        FieldRule("stop_index", minimum=0, maximum=40, feed="timetable_feed"),
        FieldRule("frac_route", minimum=0, maximum=1, feed="timetable_feed"),
        FieldRule("distance_ahead_km", minimum=0, maximum=4500, feed="timetable_feed"),
        FieldRule("stops_ahead", minimum=0, maximum=40, feed="timetable_feed"),
        FieldRule("known_delay_min", minimum=0, maximum=1440, feed="live_running_feed"),
        FieldRule("journey_started", minimum=0, maximum=1, feed="live_running_feed"),
        FieldRule("weather_severity", minimum=0, maximum=1, feed="weather_feed"),
        FieldRule("station_load", minimum=0, maximum=10, feed="station_ops_feed"),
        FieldRule("hist_mean_delay", minimum=0, maximum=600, feed="punctuality_stats_feed"),
    ])

    def generate(self, network, n, rng, world):
        return generators.delay_samples(network, n, rng, world)

    def train_models(self, X, records, seed, weights=None):
        y = np.array([r[self.label_field] for r in records], dtype=float)
        models = {}
        for q in (0.1, 0.5, 0.9):
            m = hgb_pipeline(self.categorical, self.numeric,
                             HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=220,
                                                           learning_rate=0.08, max_leaf_nodes=31, random_state=seed))
            m.fit(X[self.features], y, model__sample_weight=weights)
            models[f"q{int(q * 100)}"] = m
        return models

    def predict(self, artifact, df, context):
        X = df[self.features]
        m = artifact["models"]
        steps = artifact.get("recalibration")
        p10, p50, p90 = (np.maximum(0, apply_affine_recalibration(m[k].predict(X), steps))
                         for k in ("q10", "q50", "q90"))
        out = []
        for a, b, c in zip(p10, p50, p90):
            lo, mid, hi = sorted((float(a), float(b), float(c)))
            width = hi - lo
            conf = float(np.clip(1 - width / (2 * (mid + 20)), 0.05, 0.99))
            out.append({"arrival_delay_min": round(mid, 1), "interval_p10_p90": [round(lo, 1), round(hi, 1)],
                        "departure_delay_min": round(max(0.0, mid - 2.0), 1),
                        "confidence": round(conf, 3),
                        "risk_level": "high" if mid >= 60 else "medium" if mid >= 20 else "low"})
        return out

    def explain_output(self, artifact, df):
        return np.maximum(0, apply_affine_recalibration(artifact["models"]["q50"].predict(df[self.features]),
                                                        artifact.get("recalibration")))

    def recalibrate(self, artifact, records):
        predicted = self.explain_output(artifact, self.prepare(records))
        step = fit_affine_recalibration(predicted, [r[self.label_field] for r in records])
        if step is None:
            return None
        new = copy.deepcopy(artifact)
        new["recalibration"] = list(artifact.get("recalibration") or []) + [step]
        return new

    def metrics(self, outputs, labels, segments=None):
        if not outputs:
            return {}
        y = np.array(labels, dtype=float)
        p = np.array([o["arrival_delay_min"] for o in outputs])
        lo = np.array([o["interval_p10_p90"][0] for o in outputs])
        hi = np.array([o["interval_p10_p90"][1] for o in outputs])
        m = {"mae": round(float(np.mean(np.abs(p - y))), 3),
             "bias": round(float(np.mean(p - y)), 3),
             "interval_coverage": round(float(np.mean((y >= lo) & (y <= hi))), 4),
             "n": int(len(y))}
        if segments:
            seg = np.array(segments)
            by = {s: round(float(np.mean(np.abs(p[seg == s] - y[seg == s]))), 3)
                  for s in sorted(set(segments)) if (seg == s).sum() >= 20}
            m["mae_by_segment"] = by
            m["fairness_ratio"] = disparity_ratio(by)
        return m

    def sample_losses(self, outputs, labels):
        """Mean pinball loss of the p10/p50/p90 forecast (scores the median and the interval together)."""
        y = np.array(labels, dtype=float)
        loss = np.zeros(len(y))
        for q, pick in ((0.1, lambda o: o["interval_p10_p90"][0]), (0.5, lambda o: o["arrival_delay_min"]),
                        (0.9, lambda o: o["interval_p10_p90"][1])):
            pred = np.array([pick(o) for o in outputs], dtype=float)
            diff = y - pred
            loss += np.maximum(q * diff, (q - 1) * diff)
        return loss / 3
