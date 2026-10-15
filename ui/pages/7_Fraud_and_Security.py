"""Fraud, security and bot detection: score a session and review the alert queue."""
import pandas as pd
import streamlit as st

import client
from components import call, fmt_ts, passport_card, pill, setup

setup("Fraud, Security & Bots", "Risk scores for booking sessions with reason codes an investigator can act on, "
      "plus an anomaly detector for behaviour the model has never seen.", "🛡️")

PRESETS = {
    "Ordinary passenger": dict(requests_per_min=4, interarrival_cv=0.9, form_fill_seconds=38, headless_signals=0,
                               is_tatkal_window=0, ms_after_tatkal_open=-1, booking_attempts=1, passengers_per_booking=2,
                               accounts_on_device=1, accounts_on_payment_instrument=1, account_age_days=820,
                               failed_payments=0, distinct_passenger_names_30d=3, ip_bookings_last_hour=1),
    "Bulk-booking bot (Tatkal)": dict(requests_per_min=70, interarrival_cv=0.05, form_fill_seconds=1.5,
                                      headless_signals=1, is_tatkal_window=1, ms_after_tatkal_open=300,
                                      booking_attempts=8, passengers_per_booking=6, accounts_on_device=14,
                                      accounts_on_payment_instrument=9, account_age_days=4, failed_payments=0,
                                      distinct_passenger_names_30d=45, ip_bookings_last_hour=25),
    "Stealth bot (human pacing)": dict(requests_per_min=6, interarrival_cv=0.9, form_fill_seconds=27, headless_signals=0,
                                       is_tatkal_window=1, ms_after_tatkal_open=2400, booking_attempts=4,
                                       passengers_per_booking=3, accounts_on_device=3, accounts_on_payment_instrument=1,
                                       account_age_days=310, failed_payments=0, distinct_passenger_names_30d=7,
                                       ip_bookings_last_hour=3),
    "Stolen card": dict(requests_per_min=5, interarrival_cv=0.8, form_fill_seconds=30, headless_signals=0,
                        is_tatkal_window=0, ms_after_tatkal_open=-1, booking_attempts=3, passengers_per_booking=2,
                        accounts_on_device=1, accounts_on_payment_instrument=11, account_age_days=12, failed_payments=5,
                        distinct_passenger_names_30d=4, ip_bookings_last_hour=2),
}

tab1, tab2 = st.tabs(["Score a session", "Alert queue"])
with tab1:
    preset = st.selectbox("Start from a preset", list(PRESETS))
    vals = dict(PRESETS[preset])
    cols = st.columns(4)
    for i, (k, v) in enumerate(PRESETS[preset].items()):
        col = cols[i % 4]
        if isinstance(v, int) and k not in ("ms_after_tatkal_open",):
            vals[k] = col.number_input(k.replace("_", " "), value=int(v), min_value=0, key=f"{preset}_{k}")
        else:
            vals[k] = col.number_input(k.replace("_", " "), value=float(v), key=f"{preset}_{k}")
    for k in ("headless_signals", "is_tatkal_window"):
        vals[k] = min(1, int(vals[k]))
    vals["passengers_per_booking"] = max(1, min(6, int(vals["passengers_per_booking"])))
    for k in ("accounts_on_device", "accounts_on_payment_instrument"):
        vals[k] = max(1, int(vals[k]))
    if st.button("Score session", type="primary"):
        res = call(client.post, "/fraud/score", vals)
        if res and res["status"] == "ok":
            out = res["output"]
            a, b, c, d = st.columns(4)
            a.metric("Risk score", f"{out['risk_score']:.2f}")
            b.markdown(f"**Decision**<br>{pill('critical' if out['decision'] == 'block_and_review' else 'warning' if out['decision'] == 'step_up_verification' else 'ok', out['decision'].replace('_', ' '))}",
                       unsafe_allow_html=True)
            c.metric("Most likely pattern", out["predicted_type"])
            d.metric("Anomaly score", f"{out['anomaly_score']:.2f}")
            if out["reason_codes"]:
                st.markdown("**Reason codes**\n" + "\n".join(f"- {r}" for r in out["reason_codes"]))
            st.caption(f"Decision threshold {out['threshold']} (set by governance; changes need human approval).")
            passport_card(res["passport"], f"fraud_{res['prediction_id'][:8]}")
        elif res:
            st.error(f"Blocked by data contract: {res['output']['reasons']}")

with tab2:
    alerts = call(client.get, "/gov/fraud/alerts", limit=60) or []
    if not alerts:
        st.info("No flagged sessions yet.")
    else:
        df = pd.DataFrame([{"time": fmt_ts(a["ts"]), "risk": a["risk_score"], "decision": a["decision"],
                            "pattern": a["predicted_type"], "investigation outcome": a["outcome"] or "pending",
                            "correct?": "-" if a["outcome"] is None else
                            ("yes" if (a["outcome"] != "legit") == (a["decision"] == "block_and_review") else "NO"),
                            "reasons": "; ".join(a["reason_codes"]), "model": f"v{a['version']}", "source": a["source"]}
                           for a in alerts])
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Investigation outcomes come back from the twin; the governance layer uses them to track false "
                   "positives and false negatives.")
