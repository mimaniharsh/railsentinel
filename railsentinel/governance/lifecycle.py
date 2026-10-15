"""Automated model lifecycle: retraining, evaluation, shadow deployment,
promotion and rollback.

Retraining waits for enough labelled production data from after the detected
change point (Page-Hinkley / CUSUM), then builds three candidates that compete
on a held-out slice of the newest data:
* the production model recalibrated on the recent data (a multinomial logit on
  its log-probabilities for classifiers, an affine correction of the quantiles
  for regressors): keeps everything the model learned and moves only its output
  mapping, which is the most label-efficient fix when outcome rates shift;
* a model retrained on the recent data alone: adapts fully when the
  input-outcome relationship itself has changed (for example, new bot behaviour);
* a model retrained on the recent data plus a sample of the twin's history, with
  the recent data carrying three times the history's total weight: steadier when
  the change is narrow.
The candidate with the lowest holdout loss goes forward.

Every "is the new version better?" decision is a paired comparison on the same
records: both versions score each record, a per-record loss under a proper
scoring rule (Brier, pinball, reciprocal rank) is computed for each, and a
one-sided paired t-test asks whether the candidate's loss is lower. The
candidate must also be no worse than production on the policy's headline metric
(accuracy, MAE, recall, hit@1). This gate is applied twice: on a held-out slice
of the newest labelled data before the candidate may become a shadow
(screening, alpha 0.10), and on live shadow traffic before it may be promoted
(confirmation, alpha 0.05). At promotion the new version's
baseline is reset to what it achieved on live traffic, so monitors compare it
with a realistic figure rather than the holdout score it was selected on.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from railsentinel.ml.runtime import recent_predictions
from railsentinel.twin.world import WorldParams

# One-sided significance levels for "candidate loss is lower". The holdout is a screening step on a
# small slice; the live shadow test that follows is the independent confirmation, so it is stricter.
SCREEN_ALPHA = 0.10
CONFIRM_ALPHA = 0.05
NON_INFERIORITY = 0.01        # the headline metric may not get more than 1% (relative) worse
MIN_SHADOW_LABELS = 120       # labelled shadow comparisons needed before promotion
HISTORY_SAMPLE = 2000
RECENT_WEIGHT_RATIO = 3.0  # recent data carries 3x the total weight of the history sample


def relative_gain(higher_is_better: bool, candidate: float | None, production: float | None) -> float:
    if production in (None, 0) or candidate is None:
        return 0.0
    return (candidate - production) / abs(production) if higher_is_better else \
        (production - candidate) / abs(production)


def paired_comparison(spec, metric: str, candidate_outputs: list[dict], production_outputs: list[dict],
                      labels: list, segments: list | None = None, alpha: float = CONFIRM_ALPHA) -> dict:
    """Compare two versions scored on the same labelled records."""
    cand_m = spec.metrics(candidate_outputs, labels, segments)
    prod_m = spec.metrics(production_outputs, labels, segments)
    gain = relative_gain(spec.higher_is_better, cand_m.get(metric), prod_m.get(metric))
    loss_c = spec.sample_losses(candidate_outputs, labels)
    loss_p = spec.sample_losses(production_outputs, labels)
    diff = loss_p - loss_c  # positive where the candidate is better
    if len(diff) < 2:
        p_value = 1.0
    elif float(np.std(diff)) > 0:
        p_value = float(stats.ttest_rel(loss_p, loss_c, alternative="greater").pvalue)
    else:  # identical difference on every record: certain improvement if positive, none otherwise
        p_value = 0.0 if float(np.mean(diff)) > 0 else 1.0
    better = bool(float(np.mean(diff)) > 0 and p_value < alpha and gain >= -NON_INFERIORITY)
    return {"metric": metric, "candidate": cand_m.get(metric), "production": prod_m.get(metric),
            "relative_gain": round(gain, 4), "loss_candidate": round(float(np.mean(loss_c)), 5),
            "loss_production": round(float(np.mean(loss_p)), 5),
            "loss_reduction": round(float(np.mean(diff)), 5), "p_value": round(p_value, 6), "n": int(len(diff)),
            "better": better, "candidate_metrics": cand_m, "production_metrics": prod_m}


class Lifecycle:
    def __init__(self, platform):
        self.p = platform

    def retrain(self, asset: str, actor: str = "supervisor-agent", since_ts: float | None = None,
                reason: str = "") -> dict:
        from railsentinel.governance.monitors import window_floor

        spec = self.p.specs[asset]
        pol = self.p.policies.for_asset(asset)
        needed = int(pol.get("retraining", {}).get("min_labels", 450))
        if since_ts is None:
            # No localised change point (for example, every record in the monitoring window is already
            # post-change): train on the data the monitors judged, not on records from before the window.
            since_ts = window_floor(self.p.db, asset) or None
        rows = recent_predictions(self.p.db, asset, 3000, labelled_only=True, sources=("traffic", "api"))
        recent = [r for r in rows if since_ts is None or r["ts"] >= since_ts]
        if len(recent) < needed:
            # Retraining on a handful of post-change outcomes produces noisy models: wait for more labels.
            return {"ok": True, "status": "waiting_for_labels", "labels": len(recent), "needed": needed}
        recent = sorted(recent, key=lambda r: r["ts"])
        # one prediction log row per request: drop duplicates from earlier versions
        seen, uniq = set(), []
        for r in recent:
            if r["request_id"] not in seen:
                seen.add(r["request_id"])
                uniq.append(r)
        records = [spec.record_from_log(r["features"], r["label"]) for r in uniq]
        cut = int(len(records) * 0.75)
        train_recent, holdout = records[:cut], records[cut:]
        rng = self.p.rng()
        seed = int(rng.integers(0, 1_000_000))
        # Two training recipes compete on the holdout. Recent data alone adapts fully to a changed world
        # (concept drift makes the old history actively misleading); recent data plus a weighted sample of
        # the twin's history is steadier when the change is narrower. The holdout decides, not a fixed rule.
        hist_records = spec.frame_to_records(
            spec.generate(self.p.network, max(HISTORY_SAMPLE, 2 * len(train_recent)), rng, WorldParams()))
        w_recent = RECENT_WEIGHT_RATIO * len(hist_records) / max(1, len(train_recent))
        recipes = {
            "recent_only": (train_recent, None),
            "recent_plus_history": (train_recent + hist_records,
                                    np.array([w_recent] * len(train_recent) + [1.0] * len(hist_records))),
        }
        prod_row = self.p.registry.production(asset)
        prod_art = self.p.registry.load(asset, prod_row["version"])
        ctx = self.p.runtime(asset)._context()
        labels = [spec.label_of(r) for r in holdout]
        segs = [spec.segment_of(r) for r in holdout]
        metric = pol["performance"]["metric"]
        prod_out = spec.predict(prod_art, spec.prepare(holdout), ctx)
        trials = {}
        for name, (train, weights) in recipes.items():
            art = spec.fit(train, seed=seed, sample_weight=weights) if spec.name != "train_recommender" \
                else spec.fit(train, seed=seed)
            cmp = paired_comparison(spec, metric, spec.predict(art, spec.prepare(holdout), ctx), prod_out, labels,
                                    segs, alpha=SCREEN_ALPHA)
            trials[name] = (art, cmp, len(train))
        # Third recipe: keep the production model and recalibrate its outputs on the recent data. With few
        # post-change labels this often beats retraining, because the model's learned structure is kept.
        recalibrated = spec.recalibrate(prod_art, train_recent)
        if recalibrated is not None:
            cmp = paired_comparison(spec, metric, spec.predict(recalibrated, spec.prepare(holdout), ctx), prod_out,
                                    labels, segs, alpha=SCREEN_ALPHA)
            trials["recalibrated_production"] = (recalibrated, cmp, len(train_recent))
        recipe = min(trials, key=lambda k: trials[k][1]["loss_candidate"])
        candidate, cmp, train_rows = trials[recipe]
        # The candidate's baseline and drift reference describe the world it was retrained for.
        cand_m = cmp["candidate_metrics"]
        candidate["baseline_metrics"] = cand_m
        candidate["reference"] = spec.drift_frame(train_recent + holdout).reset_index(drop=True)
        candidate["train_rows"] = train_rows
        if asset == "train_recommender":
            candidate["premium_boost"] = 0.0
        recipe_scores = {k: {"loss": v[1]["loss_candidate"], metric: v[1]["candidate"]} for k, v in trials.items()}
        version = self.p.registry.add_version(
            asset, candidate, cand_m, params={"recipe": recipe, "recipe_scores": recipe_scores,
                                              "recent_rows": len(train_recent), "history_rows": len(hist_records),
                                              "recent_weight": round(w_recent, 2), "since_ts": since_ts, "seed": seed},
            stage="candidate", notes=f"retrained by {actor} ({recipe}): {reason}"[:300], parent=prod_row["version"],
            actor=actor)
        result = {"ok": True, "version": version, "recipe": recipe, "recipe_scores": recipe_scores,
                  "holdout": len(holdout), "recent_rows": len(train_recent),
                  **{k: v for k, v in cmp.items() if not k.endswith("_metrics")}}
        verdict = (f"{metric} {cmp['relative_gain']:+.1%}, loss {cmp['loss_production']} -> {cmp['loss_candidate']}, "
                   f"p={cmp['p_value']:.3g}")
        if cmp["better"]:
            self.p.registry.set_stage(asset, version, "shadow", actor=actor,
                                      reason=f"beat production on recent holdout ({verdict})")
            result["status"] = "shadow"
        else:
            self.p.registry.set_stage(asset, version, "rejected", actor=actor,
                                      reason=f"not better than production on recent holdout ({verdict})")
            result["status"] = "rejected"
        return result

    def compare_shadow(self, asset: str, prod: dict, shadow: dict) -> dict:
        spec = self.p.specs[asset]
        metric = self.p.policies.for_asset(asset)["performance"]["metric"]
        sh = recent_predictions(self.p.db, asset, 2000, version=shadow["version"], shadow=1, labelled_only=True)
        if not sh:
            return {"status": "insufficient", "labels": 0, "shadow_version": shadow["version"]}
        ids = {r["request_id"] for r in sh}
        pr = [r for r in recent_predictions(self.p.db, asset, 4000, version=prod["version"], labelled_only=True)
              if r["request_id"] in ids]
        pr_by = {r["request_id"]: r for r in pr}
        pairs = [(pr_by[r["request_id"]], r) for r in sh if r["request_id"] in pr_by]
        if len(pairs) < MIN_SHADOW_LABELS:
            return {"status": "insufficient", "labels": len(pairs), "shadow_version": shadow["version"],
                    "needed": MIN_SHADOW_LABELS}
        labels = [p[0]["label"] for p in pairs]
        segs = [p[0]["segment"] for p in pairs]
        cmp = paired_comparison(spec, metric, [p[1]["output"] for p in pairs], [p[0]["output"] for p in pairs],
                                labels, segs)
        return {"status": "ready", "labels": len(pairs), "metric": metric, "production": cmp["production"],
                "shadow": cmp["candidate"], "relative_gain": cmp["relative_gain"],
                "loss_production": cmp["loss_production"], "loss_shadow": cmp["loss_candidate"],
                "p_value": cmp["p_value"], "shadow_version": shadow["version"],
                "production_version": prod["version"], "promote": cmp["better"],
                "shadow_metrics": cmp["candidate_metrics"]}

    def promote(self, asset: str, version: int, actor: str, reason: str, live_metrics: dict | None = None) -> dict:
        """Put `version` into production. `live_metrics` (from shadow evaluation) become its monitoring baseline."""
        self.p.registry.set_stage(asset, version, "production", actor=actor, reason=reason)
        if live_metrics:
            self.p.registry.update_metrics(asset, version, live_metrics, actor=actor,
                                           reason="baseline reset to live shadow-traffic performance")
        from railsentinel.governance.monitors import set_window_floor
        set_window_floor(self.p.db, asset)
        return {"ok": True, "version": version}

    def reject_shadow(self, asset: str, version: int, actor: str, reason: str) -> None:
        self.p.registry.set_stage(asset, version, "rejected", actor=actor, reason=reason)

    def rollback(self, asset: str, actor: str, reason: str, to_version: int | None = None) -> dict:
        result = self.p.registry.rollback(asset, actor=actor, reason=reason, to_version=to_version)
        if result.get("ok"):
            from railsentinel.governance.monitors import set_window_floor
            set_window_floor(self.p.db, asset)
        return result
