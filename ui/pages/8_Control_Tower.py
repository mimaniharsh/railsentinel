"""Control Tower: the autonomous governance layer at work."""
import pandas as pd
import streamlit as st

import client
from components import call, fmt_ts, pill, setup

setup("Control Tower", "Every AI asset, its health, the incidents the supervisor agent is handling, and the "
      "decisions waiting for a human.", "🗼")

top = st.columns([1, 1, 4])
if top[0].button("Run supervisor cycle now", type="primary"):
    res = call(client.post, "/gov/cycle")
    if res:
        st.toast(f"Cycle done in {res['duration_ms']:.0f} ms, {len(res['events'])} event(s)")
if top[1].button("Refresh"):
    st.rerun()

fleet = call(client.get, "/gov/fleet")
if not fleet:
    st.stop()
cols = st.columns(len(fleet["assets"]))
for col, a in zip(cols, fleet["assets"]):
    h = a["health"]
    col.markdown(f"<div class='rs-card'><h4>{a['name'].replace('_', ' ')}</h4>{pill(h.get('status', 'unknown'))}"
                 f"<div class='rs-muted'>v{a['production_version']} in production"
                 f"{' · v' + str(a['shadow_version']) + ' shadow' if a['shadow_version'] else ''}<br>"
                 f"{a['kind']} · {a['risk_tier']} risk</div></div>", unsafe_allow_html=True)

approvals = call(client.get, "/gov/approvals", status="pending") or []
if approvals:
    st.subheader("Waiting for your decision")
    for ap in approvals:
        with st.container(border=True):
            st.markdown(f"**{ap['action'].replace('_', ' ')}** on **{ap['model']}** · {ap['id']} · incident "
                        f"{ap['incident_id']}  \n{ap['reason']}")
            a, b, _ = st.columns([1, 1, 5])
            if a.button("Approve", key=f"ap_{ap['id']}", type="primary"):
                r = call(client.post, f"/gov/approvals/{ap['id']}", {"approve": True})
                if r:
                    st.success("Approved and executed.")
                    st.rerun()
            if b.button("Reject", key=f"rj_{ap['id']}"):
                if call(client.post, f"/gov/approvals/{ap['id']}", {"approve": False}):
                    st.rerun()

st.subheader("Incidents")
show_all = st.toggle("Show resolved incidents too", value=False)
incidents = call(client.get, "/gov/incidents", active=not show_all, limit=40) or []
if not incidents:
    st.success("No open incidents.")
for inc in incidents:
    rca = inc["rca"]
    with st.expander(f"{inc['id']} · {inc['title']} · {inc['status']} · priority {inc['priority']}",
                     expanded=inc["status"] not in ("resolved", "closed_by_reset")):
        st.markdown(f"{pill(inc['severity'])} {pill(inc['status'])} <span class='rs-muted'>opened "
                    f"{fmt_ts(inc['opened_ts'])}{' · closed ' + fmt_ts(inc['closed_ts']) if inc['closed_ts'] else ''}"
                    f"{' · scenario ' + inc['scenario'] if inc.get('scenario') else ''}</span>", unsafe_allow_html=True)
        st.write(inc["narrative"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Root cause:** `{rca.get('root_cause')}` ({rca.get('cause_type', '').replace('_', ' ')}, "
                        f"confidence {rca.get('confidence', 0):.0%})")
            st.markdown("**Evidence chain**\n" + "\n".join(f"- {e}" for e in rca.get("evidence_chain", [])))
            if rca.get("impacted"):
                st.markdown("**Downstream impact:** " + ", ".join(rca["impacted"]))
        with c2:
            st.markdown("**Remediation timeline**")
            for a in inc["actions"]:
                res = a.get("result") or {}
                detail = ", ".join(f"{k}={v}" for k, v in res.items()
                                   if k in ("status", "version", "relative_gain", "candidate", "production",
                                            "new_threshold", "from", "to", "quarantined", "circuit_breakers", "labels"))
                st.markdown(f"{pill(a['status'])} **{a['action']}** <span class='rs-muted'>{a.get('reason', '')}"
                            f"{' · ' + detail if detail else ''}</span>", unsafe_allow_html=True)

st.subheader("Asset drill-down")
names = [a["name"] for a in fleet["assets"]]
name = st.selectbox("Asset", names)
detail = call(client.get, f"/gov/assets/{name}")
if detail:
    rep = detail["report"]
    checks = rep.get("checks", {})
    st.markdown(" ".join(f"{pill(v.get('status', 'unknown'), k)}" for k, v in checks.items()), unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Checks", "Versions", "Metric history", "Policy & dependencies"])
    with tab1:
        for k, v in checks.items():
            with st.expander(f"{k}: {v.get('status')}"):
                st.json(v, expanded=False)
    with tab2:
        vers = detail["versions"]
        st.dataframe(pd.DataFrame([{"version": v["version"], "stage": v["stage"], "created": fmt_ts(v["created_ts"]),
                                    "promoted": fmt_ts(v["promoted_ts"]), "notes": v["notes"],
                                    "key metric": next((f"{k}={v['metrics'][k]}" for k in
                                                        ("accuracy", "mae", "recall", "hit_at_1") if k in v["metrics"]), "")}
                                   for v in vers]), hide_index=True, width="stretch")
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Version", [v["version"] for v in vers])
        if c2.button("Promote to production"):
            if call(client.post, f"/gov/assets/{name}/promote", {"version": int(target)}):
                st.rerun()
        if c3.button("Roll back"):
            if call(client.post, f"/gov/assets/{name}/rollback", {"to_version": None}):
                st.rerun()
    with tab3:
        metric_names = {"confirmation_predictor": ["accuracy", "max_psi", "fairness_ratio", "blocked_rate"],
                        "delay_predictor": ["mae", "max_psi", "latency_p95_ms"],
                        "fare_forecaster": ["mae", "blocked_rate", "quote_anomaly_rate", "max_psi"],
                        "fraud_detector": ["recall", "fpr", "max_psi"],
                        "train_recommender": ["hit_at_1", "premium_exposure_ratio", "fairness_ratio"],
                        "support_assistant": ["hallucination_rate", "grounding_mean", "csat"],
                        "travel_planner_agent": ["violation_rate", "abnormal_rate", "tool_error_rate", "success_rate"]}
        for metric in metric_names.get(name, []):
            rows = call(client.get, "/gov/metrics", asset=name, metric=metric, hours=24) or []
            if rows:
                df = pd.DataFrame(rows)
                df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                st.caption(metric)
                st.line_chart(df, x="time", y="value", height=160)
    with tab4:
        st.json(detail["policy"], expanded=False)
        st.markdown("**Depends on:** " + ", ".join(detail["dependencies"]))
        st.markdown("**Used by:** " + (", ".join(detail["downstream"]) or "nothing downstream"))

with st.expander("Dependency graph used for root-cause analysis"):
    g = call(client.get, "/gov/graph")
    if g:
        st.graphviz_chart(g["dot"], width="stretch")
