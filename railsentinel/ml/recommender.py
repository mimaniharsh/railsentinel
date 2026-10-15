"""Smart train recommendation engine.

Ranks every bookable train/class option for a journey with a utility model
learned from what passengers actually chose. Training uses pairwise logistic
regression: for each past choice, the chosen option's features minus each
rejected option's features is a positive example (and the reverse a negative
one), which recovers the weights of a linear utility (a conditional-logit
style choice model). Personalisation comes from the passenger's history
(usual class, night-travel share, fare level paid), not from demographic
segments; segments are used only to audit fairness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from railsentinel.ml.base import ModelSpec, disparity_ratio
from railsentinel.ml.contracts import DataContract, FieldRule
from railsentinel.twin import generators

RANK_FEATURES = ["price_ratio", "over_budget", "duration_ratio", "punctuality", "confirm_prob", "comfort_norm",
                 "night_match", "class_match", "price_x_sensitivity"]
# (text when the factor pushed this option up, text when it pushed it down)
FEATURE_TEXT = {
    "price_ratio": ("cheaper than most options", "pricier than most options"),
    "over_budget": ("within your budget", "over your budget"),
    "duration_ratio": ("faster than most options", "slower than most options"),
    "punctuality": ("better punctuality record", "weaker punctuality record"),
    "confirm_prob": ("better chance of a confirmed berth than most options",
                     "lower chance of a confirmed berth than most options"),
    "comfort_norm": ("more comfortable class", "less comfortable class"),
    "night_match": ("fits your usual travel time", "outside your usual travel time"),
    "class_match": ("your usual class", "not your usual class"),
    "price_x_sensitivity": ("good value for what you usually pay", "expensive for what you usually pay"),
}


def option_features(option: dict, user: dict, budget: float, min_duration: float) -> list[float]:
    sensitivity = 0.5 / max(float(user.get("hist_fare_per_km", 0.5)), 0.2)
    price_ratio = option["fare"] / max(budget, 1.0)
    night = float(user.get("hist_night_share", 0.5))
    night_match = option["overnight"] * night + (1 - option["overnight"]) * (1 - night)
    return [price_ratio, float(option["fare"] > budget), option["duration_min"] / max(min_duration, 1.0),
            float(option["punctuality"]), float(option["confirm_prob"]), option["comfort"] / 6.0, night_match,
            float(option["travel_class"] == user.get("hist_class_mode")), price_ratio * sensitivity]


def episode_matrix(episode: dict) -> np.ndarray:
    options = episode["options"]
    min_dur = min(o["duration_min"] for o in options)
    return np.array([option_features(o, episode["user"], episode["budget"], min_dur) for o in options])


class TrainRecommender(ModelSpec):
    name = "train_recommender"
    title = "Smart Train Recommender"
    risk_tier = "medium"
    description = ("Ranks train/class options using a choice model learned from passenger decisions, "
                   "personalised by travel history, with per-option rationale.")
    label_field = "chosen_index"
    segment_field = "segment"
    primary_metric = "hit_at_1"
    higher_is_better = True
    fairness_metric = "hit_at_1"
    explain_unit = "utility"
    default_train_size = 3000
    contract = DataContract([
        FieldRule("budget", minimum=50, maximum=200000, feed="user_profile_store"),
        FieldRule("n_options", minimum=1, maximum=500, feed="timetable_feed"),
        FieldRule("hist_night_share", minimum=0, maximum=1, feed="user_profile_store"),
        FieldRule("hist_fare_per_km", minimum=0, maximum=20, feed="user_profile_store"),
    ])

    def generate(self, network, n, rng, world):
        return generators.recommendation_episodes(network, n, rng, world)

    # Episodes are not flat rows, so the request-level summary is what gets
    # validated, drift-monitored and logged as "features".
    @staticmethod
    def summary(episode: dict) -> dict:
        u = episode["user"]
        fares = [o["fare"] for o in episode["options"]] or [0]
        return {"budget": float(episode["budget"]), "n_options": len(episode["options"]),
                "hist_night_share": float(u.get("hist_night_share", 0.5)),
                "hist_fare_per_km": float(u.get("hist_fare_per_km", 0.5)),
                "mean_fare": float(np.mean(fares)), "hist_class_mode": str(u.get("hist_class_mode", "SL")),
                "segment": str(u.get("segment", "unknown"))}

    def drift_columns(self):
        return ["hist_class_mode", "segment"], ["budget", "n_options", "hist_night_share", "hist_fare_per_km",
                                               "mean_fare"]

    def drift_frame(self, records):
        return pd.DataFrame([self.summary(r) if "options" in r else r for r in records])

    def segment_of(self, record):
        return str(record.get("user", {}).get("segment", record.get("segment", "unknown")))

    def label_of(self, record):
        return record.get("chosen_index")

    def frame_to_records(self, data):
        return list(data)

    def records_to_frame(self, records):
        return list(records)

    def fit(self, data, seed=0, sample_weight=None):
        episodes = list(data)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(episodes))
        cut = int(len(episodes) * 0.8)
        train = [episodes[i] for i in idx[:cut]]
        test = [episodes[i] for i in idx[cut:]]
        Xs, ys = [], []
        for ep in train:
            M = episode_matrix(ep)
            c = ep["chosen_index"]
            for j in range(len(M)):
                if j == c:
                    continue
                Xs.append(M[c] - M[j]); ys.append(1)
                Xs.append(M[j] - M[c]); ys.append(0)
        model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
        model.fit(np.array(Xs), np.array(ys))
        artifact = {"spec": self.name, "features": RANK_FEATURES, "coef": model.coef_[0].tolist(),
                    "premium_boost": 0.0, "train_rows": len(train)}
        outs = self.predict(artifact, test, {})
        artifact["baseline_metrics"] = self.metrics(outs, [e["chosen_index"] for e in test],
                                                    [self.segment_of(e) for e in test])
        ref = self.drift_frame(train)
        artifact["reference"] = ref.sample(n=min(len(ref), 3000), random_state=seed).reset_index(drop=True)
        return artifact

    def predict(self, artifact, episodes, context):
        coef = np.array(artifact["coef"])
        boost = float(artifact.get("premium_boost", 0.0))
        results = []
        for ep in episodes:
            M = episode_matrix(ep)
            premium = np.array([o.get("is_premium", 0) for o in ep["options"]], dtype=float)
            scores = M @ coef + boost * premium
            exp = np.exp(scores - scores.max())
            probs = exp / exp.sum()
            order = [int(i) for i in np.argsort(-scores)]
            mean = M.mean(axis=0)
            ranked = []
            for rank, i in enumerate(order[:10]):
                contrib = coef * (M[i] - mean)
                reasons = sorted(zip(RANK_FEATURES, contrib), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                ranked.append({"index": i, "rank": rank + 1, "score": round(float(scores[i]), 4),
                               "match": round(float(probs[i]), 4),
                               "rationale": [{"factor": FEATURE_TEXT[f][0 if v >= 0 else 1], "feature": f,
                                              "impact": round(float(v), 4)} for f, v in reasons],
                               "boost_applied": round(float(boost * premium[i]), 4)})
            results.append({"ranking": order, "top": ranked,
                            "top1_is_premium": int(premium[order[0]]) if len(order) else 0,
                            "premium_share": float(premium.mean()) if len(premium) else 0.0,
                            "confidence": round(float(probs.max()), 4)})
        return results

    def metrics(self, outputs, labels, segments=None):
        if not outputs:
            return {}
        ranks = []
        for o, lbl in zip(outputs, labels):
            ranking = o["ranking"]
            ranks.append(ranking.index(int(lbl)) if int(lbl) in ranking else len(ranking))
        ranks = np.array(ranks)
        hit = (ranks == 0).astype(float)
        exposure = [(o["top1_is_premium"], o["premium_share"]) for o in outputs if o["premium_share"] > 0]
        m = {"hit_at_1": round(float(hit.mean()), 4), "mrr": round(float(np.mean(1 / (ranks + 1))), 4),
             "n": int(len(ranks))}
        if exposure:
            top = np.mean([e[0] for e in exposure])
            share = np.mean([e[1] for e in exposure])
            m["premium_exposure_ratio"] = round(float(top / share), 3) if share else None
        if segments:
            seg = np.array(segments)
            by = {s: round(float(hit[seg == s].mean()), 4) for s in sorted(set(segments)) if (seg == s).sum() >= 15}
            m["hit_at_1_by_segment"] = by
            m["n_by_segment"] = {s: int((seg == s).sum()) for s in by}
            m["fairness_ratio"] = disparity_ratio(by)
        return m

    def sample_losses(self, outputs, labels):
        """1 - reciprocal rank of the option the passenger actually chose (0 when it was ranked first)."""
        losses = []
        for o, lbl in zip(outputs, labels):
            ranking = o["ranking"]
            rank = ranking.index(int(lbl)) if int(lbl) in ranking else len(ranking)
            losses.append(1 - 1 / (rank + 1))
        return np.array(losses, dtype=float)

    supports_shapley = False

    def explain_output(self, artifact, df):
        """Utility of the top-ranked option per request (rationale per option is returned with every ranking)."""
        return np.array([max(item["score"] for item in out["top"]) if out["top"] else 0.0
                         for out in self.predict(artifact, df, {})])

    def validation_record(self, record):
        return self.summary(record)

    def prepare(self, records):
        return list(records)

    def log_features(self, record):
        keep = ("user", "budget", "origin", "destination", "journey_date")
        out = {k: record[k] for k in keep if k in record}
        out["options"] = [{k: v for k, v in o.items() if k != "snapshot"} for o in record["options"]]
        return out
