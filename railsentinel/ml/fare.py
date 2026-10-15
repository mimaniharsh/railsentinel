"""Fare and demand forecaster.

Forecasts how full a train/class will be at a future point in the booking
window. For dynamically priced (premium) trains the twin's flexi-style rule
turns occupancy into fare; for fixed-fare trains the same forecast tells the
passenger when the class is likely to sell out. A quote-anomaly check compares
each fare quote with the price the rule implies for the current occupancy.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from railsentinel.ml.base import ModelSpec, apply_affine_recalibration, fit_affine_recalibration, hgb_pipeline
from railsentinel.ml.contracts import DataContract, FieldRule
from railsentinel.twin import generators
from railsentinel.twin.inventory import ARP_DAYS, flexi_multiplier
from railsentinel.twin.seasons import SEASONS

SOLD_OUT = 0.98
ANOMALY_LOG_RATIO = 0.1  # quotes more than ~10% away from the rule-implied fare are anomalous


def _quote_ratio_check(record: dict):
    try:
        base, quote = float(record.get("base_fare")), float(record.get("quoted_fare"))
    except (TypeError, ValueError):
        return None
    if base > 0 and quote > 0 and not (0.95 <= quote / base <= 1.55):
        return {"field": "quoted_fare", "feed": "fare_feed", "issue": "quote outside 0.95x-1.55x of base fare",
                "value": round(quote / base, 3)}
    return None


def _lead_check(record: dict):
    try:
        if float(record.get("target_lead")) >= float(record.get("current_lead")):
            return {"field": "target_lead", "feed": "client", "issue": "target_lead must be earlier than current_lead"}
    except (TypeError, ValueError):
        return None
    return None


def rule_fare(base_fare: float, occupancy: float, dynamic: bool) -> float:
    return base_fare * (flexi_multiplier(occupancy) if dynamic else 1.0)


class FareForecaster(ModelSpec):
    name = "fare_forecaster"
    title = "Dynamic Fare & Demand Forecaster"
    risk_tier = "medium"
    description = ("Forecasts occupancy and fare at a future booking date with an 80% interval, the "
                   "probability that the class sells out, and flags anomalous fare quotes.")
    categorical = ["train_category", "travel_class", "season"]
    numeric = ["weekday", "month", "route_demand", "distance_km", "dynamic_fare", "current_lead",
               "current_occupancy", "target_lead"]
    label_field = "target_occupancy"
    segment_field = "region"
    primary_metric = "mae"
    higher_is_better = False
    explain_unit = "occupancy"
    default_train_size = 30000
    feature_labels = {
        "current_occupancy": "Seats sold so far", "current_lead": "Days to departure now",
        "target_lead": "Days to departure at forecast date", "route_demand": "Route demand",
        "season": "Travel season", "travel_class": "Class", "train_category": "Train type",
        "dynamic_fare": "Dynamic pricing", "distance_km": "Distance (km)", "weekday": "Day of week",
        "month": "Month",
    }
    contract = DataContract([
        FieldRule("train_category", "cat", allowed=("premium", "superfast", "express", "intercity"), feed="timetable_feed"),
        FieldRule("travel_class", "cat", allowed=("2S", "SL", "CC", "3A", "2A", "1A"), feed="timetable_feed"),
        FieldRule("season", "cat", allowed=SEASONS, feed="calendar"),
        FieldRule("weekday", minimum=0, maximum=6, feed="calendar"),
        FieldRule("month", minimum=1, maximum=12, feed="calendar"),
        FieldRule("route_demand", minimum=0, maximum=3, feed="demand_stats_feed"),
        FieldRule("distance_km", minimum=1, maximum=4500, feed="timetable_feed"),
        FieldRule("dynamic_fare", minimum=0, maximum=1, feed="timetable_feed"),
        FieldRule("current_lead", minimum=1, maximum=ARP_DAYS, feed="booking_feed"),
        FieldRule("current_occupancy", minimum=0, maximum=2, feed="booking_feed"),
        FieldRule("target_lead", minimum=0, maximum=ARP_DAYS, feed="client"),
        FieldRule("base_fare", minimum=1, maximum=50000, feed="fare_feed"),
        FieldRule("quoted_fare", minimum=1, maximum=100000, feed="fare_feed"),
    ], cross_checks=[_quote_ratio_check, _lead_check])

    def generate(self, network, n, rng, world):
        return generators.fare_samples(network, n, rng, world)

    def train_models(self, X, records, seed, weights=None):
        y = np.array([r[self.label_field] for r in records], dtype=float)
        models = {}
        for q in (0.1, 0.5, 0.9):
            m = hgb_pipeline(self.categorical, self.numeric,
                             HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                                           learning_rate=0.08, random_state=seed))
            m.fit(X[self.features], y, model__sample_weight=weights)
            models[f"q{int(q * 100)}"] = m
        sold = (y >= SOLD_OUT).astype(int)
        if sold.min() == sold.max():
            # Every training record has the same outcome (possible in small retraining sets): a constant
            # model predicts exactly that outcome instead of fitting a classifier on a single class.
            sellout = DummyClassifier(strategy="prior").fit(X[self.features], sold)
        else:
            sellout = hgb_pipeline(self.categorical, self.numeric,
                                   HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                  random_state=seed))
            sellout.fit(X[self.features], sold, model__sample_weight=weights)
        models["sellout"] = sellout
        return models

    def predict(self, artifact, df, context):
        X = df[self.features]
        m = artifact["models"]
        steps = artifact.get("recalibration")
        p10, p50, p90 = (apply_affine_recalibration(m[k].predict(X), steps) for k in ("q10", "q50", "q90"))
        clf = m["sellout"]
        classes = list(clf.classes_)
        ps = clf.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        out = []
        for i, row in enumerate(df.to_dict(orient="records")):
            lo, mid, hi = sorted((float(p10[i]), float(p50[i]), float(p90[i])))
            cur = float(row["current_occupancy"])
            lo, mid, hi = max(lo, cur), max(mid, cur), max(hi, cur)  # occupancy never decreases in the twin
            dynamic = bool(int(row["dynamic_fare"]))
            base = float(row.get("base_fare") or 0)
            quote = float(row.get("quoted_fare") or 0)
            fare_now = rule_fare(base, cur, dynamic)
            fare_then = rule_fare(base, mid, dynamic)
            change = (fare_then - fare_now) / fare_now if fare_now else 0.0
            anomaly = bool(fare_now > 0 and quote > 0 and abs(math.log(quote / fare_now)) > ANOMALY_LOG_RATIO)
            out.append({
                "forecast_occupancy": round(mid, 4), "occupancy_interval": [round(lo, 4), round(hi, 4)],
                "sellout_probability": round(float(ps[i]), 4),
                "fare_now": round(fare_now), "fare_forecast": round(fare_then),
                "fare_forecast_range": [round(rule_fare(base, lo, dynamic)), round(rule_fare(base, hi, dynamic))],
                "fare_change_pct": round(change * 100, 1),
                "quote_anomaly": anomaly,
                "confidence": round(float(np.clip(1 - (hi - lo) / 0.8, 0.05, 0.99)), 3),
            })
        return out

    def explain_output(self, artifact, df):
        return apply_affine_recalibration(artifact["models"]["q50"].predict(df[self.features]),
                                          artifact.get("recalibration"))

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
        p = np.array([o["forecast_occupancy"] for o in outputs])
        lo = np.array([o["occupancy_interval"][0] for o in outputs])
        hi = np.array([o["occupancy_interval"][1] for o in outputs])
        return {"mae": round(float(np.mean(np.abs(p - y))), 4),
                "interval_coverage": round(float(np.mean((y >= lo - 1e-9) & (y <= hi + 1e-9))), 4),
                "quote_anomaly_rate": round(float(np.mean([o["quote_anomaly"] for o in outputs])), 4),
                "n": int(len(y))}

    def sample_losses(self, outputs, labels):
        """Mean pinball loss of the p10/p50/p90 occupancy forecast."""
        y = np.array(labels, dtype=float)
        loss = np.zeros(len(y))
        for q, pick in ((0.1, lambda o: o["occupancy_interval"][0]), (0.5, lambda o: o["forecast_occupancy"]),
                        (0.9, lambda o: o["occupancy_interval"][1])):
            pred = np.array([pick(o) for o in outputs], dtype=float)
            diff = y - pred
            loss += np.maximum(q * diff, (q - 1) * diff)
        return loss / 3
