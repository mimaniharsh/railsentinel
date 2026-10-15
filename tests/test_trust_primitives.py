"""Explanations, drift statistics, hallucination checks, signatures, PII masking and the audit chain."""
import numpy as np
import pandas as pd

from railsentinel.governance.hallucination import check_grounding
from railsentinel.governance.stats import change_point, page_hinkley, psi_categorical, psi_numeric
from railsentinel.ml.explain import shapley_explain
from railsentinel.security import RateLimiter, mask_pii, sign, verify


def test_shapley_is_additive_and_exact_for_linear_models():
    rng = np.random.default_rng(0)
    bg = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    f = lambda df: 2 * df["a"] - 3 * df["b"] + 0.5 * df["c"]  # noqa: E731
    x = pd.DataFrame([{"a": 1.0, "b": 2.0, "c": -1.0}])
    ex = shapley_explain(lambda df: f(df).to_numpy(), x, bg, ["a", "b", "c"], n_samples=64)
    total = ex["base_value"] + sum(ex["contributions"].values())
    assert abs(total - ex["prediction"]) < 1e-9
    assert ex["contributions"]["b"] < 0 < ex["contributions"]["a"]


def test_psi_and_change_detection():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    assert psi_numeric(ref, rng.normal(0, 1, 2000)) < 0.05
    assert psi_numeric(ref, rng.normal(1.5, 1, 2000)) > 0.25
    assert psi_categorical(pd.Series(["a"] * 90 + ["b"] * 10), pd.Series(["a"] * 30 + ["b"] * 70)) > 0.25
    errors = list((rng.random(150) < 0.1).astype(float)) + list((rng.random(250) < 0.4).astype(float))
    assert page_hinkley(errors)["drift"]
    assert 120 <= change_point(errors) <= 220


def test_hallucination_detector_flags_invented_numbers_only():
    evidence = ["A confirmed ticket cancelled more than 48 hours before departure attracts a flat charge of Rs 180 "
                "for AC 3-tier."]
    good = check_grounding("Cancelling a confirmed AC 3-tier ticket more than 48 hours before departure costs a flat "
                           "Rs 180.", evidence, "cancellation charge 3A")
    bad = check_grounding("Cancelling a confirmed AC 3-tier ticket more than 48 hours before departure costs a flat "
                          "Rs 450.", evidence, "cancellation charge 3A")
    meta = check_grounding("Please share your 10-digit PNR and I will look it up.", [], "status of my ticket")
    assert not good["hallucination"]
    assert bad["hallucination"] and bad["unsupported_numbers"] == ["450"]
    assert not meta["hallucination"]


def test_passport_signature_detects_tampering():
    body = {"model": "m", "confidence": 0.8, "top_factors": [{"feature": "x", "impact": 0.1}]}
    body["signature"] = sign(body, "secret")
    assert verify(body, "secret")
    assert not verify({**body, "confidence": 0.99}, "secret")
    assert not verify(body, "other-secret")


def test_pii_masking_keeps_pnr():
    text = "PNR 4512345678, mail me at a.b@example.com or call +91 98765 43210"
    masked = mask_pii(text)
    assert "4512345678" in masked and "example.com" not in masked and "98765" not in masked


def test_rate_limiter():
    rl = RateLimiter(2)
    assert rl.allow("k", now=0) and rl.allow("k", now=1)
    assert not rl.allow("k", now=2)
    assert rl.allow("k", now=70)


def test_audit_chain_detects_tampering(platform):
    assert platform.audit.verify()["valid"]
    row = platform.db.query_one("SELECT seq, details FROM audit_log ORDER BY seq LIMIT 1 OFFSET 3")
    platform.db.execute("UPDATE audit_log SET details=? WHERE seq=?", ('{"tampered": true}', row["seq"]))
    result = platform.audit.verify()
    platform.db.execute("UPDATE audit_log SET details=? WHERE seq=?", (row["details"], row["seq"]))
    assert not result["valid"] and result["broken_at_seq"] == row["seq"]
    assert platform.audit.verify()["valid"]


def test_paired_comparison_needs_a_significant_and_non_inferior_improvement():
    from railsentinel.governance.lifecycle import paired_comparison
    from railsentinel.ml.confirmation import ConfirmationPredictor

    spec = ConfirmationPredictor()
    rng = np.random.default_rng(3)
    labels = list(rng.choice(["CNF", "RAC", "WL"], size=400, p=[0.4, 0.2, 0.4]))

    def outputs(concentration, seed):
        # Dirichlet probabilities centred on the true label: higher concentration = sharper, better model
        r = np.random.default_rng(seed)
        out = []
        for y in labels:
            alpha = [concentration if s == y else 1.0 for s in ("CNF", "RAC", "WL")]
            probs = dict(zip(("CNF", "RAC", "WL"), r.dirichlet(alpha)))
            out.append({"class_probs": probs, "p_confirm": probs["CNF"],
                        "predicted_final_status": max(probs, key=probs.get)})
        return out

    good, weak = outputs(6.0, 1), outputs(1.6, 2)
    better = paired_comparison(spec, "accuracy", good, weak, labels)
    assert better["better"] and better["p_value"] < 0.05 and better["loss_reduction"] > 0
    same = paired_comparison(spec, "accuracy", weak, weak, labels)
    assert not same["better"] and same["p_value"] == 1.0
    worse = paired_comparison(spec, "accuracy", weak, good, labels)
    assert not worse["better"]
