"""RailSentinel home: what the platform does and the live state of the AI fleet."""
import pandas as pd
import streamlit as st

import client
from components import call, fmt_ts, pill, setup

setup("RailSentinel", "AI rail-travel services, governed in real time by an autonomous AI control tower.")

fleet = call(client.get, "/gov/fleet")
incidents = call(client.get, "/gov/incidents", active=True) or []
approvals = call(client.get, "/gov/approvals", status="pending") or []
audit = call(client.get, "/gov/audit/verify")
if fleet is None:
    st.stop()

assets = fleet["assets"]
healthy = sum(1 for a in assets if a["health"].get("status") == "healthy")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AI assets healthy", f"{healthy} / {len(assets)}")
c2.metric("Open incidents", len(incidents))
c3.metric("Awaiting human approval", len(approvals))
c4.metric("Audit chain", "intact" if audit and audit["valid"] else "BROKEN", f"{audit['records'] if audit else 0} records",
          delta_color="off")
c5.metric("Active chaos scenario", fleet["world"]["scenario"] or "none")

st.subheader("AI fleet")
rows = []
for a in assets:
    h = a["health"]
    rows.append({"asset": a["name"], "kind": a["kind"], "risk tier": a["risk_tier"],
                 "production": f"v{a['production_version']}", "shadow": f"v{a['shadow_version']}" if a["shadow_version"] else "",
                 "status": h.get("status", "unknown"), "open incidents": h.get("open_incidents", 0),
                 "checked": h.get("checked_at", "-")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

left, right = st.columns([3, 2])
with left:
    st.subheader("How it works")
    st.markdown("""
**Passenger side.** Seven AI services: a ticket-confirmation predictor, a train recommender, delay and congestion
prediction with proactive alerts, an agentic travel planner, a fare and sell-out forecaster, a grounded support
assistant and a fraud and bot detector. Every prediction carries a **signed explanation passport**.

**Governance side.** A supervisor agent watches all of them: data contracts, drift, accuracy, fairness, exposure
bias, hallucinations and agent behaviour. It finds the root cause on a dependency graph, then retrains and
shadow-tests, rolls back, quarantines a feed or opens a circuit breaker. Actions that affect passengers wait
for a human. Everything lands in a hash-chained audit log.

**Chaos Lab.** Break the system on purpose (fog, a corrupted fare feed, stealth bots, a bad release, an
ungrounded prompt, an API outage) and watch the control tower detect, diagnose and repair it.
""")
with right:
    st.subheader("Active incidents")
    if not incidents:
        st.success("No open incidents. Try the Chaos Lab to inject one.")
    for inc in incidents[:6]:
        st.markdown(f"{pill(inc['status'])} **{inc['title']}**  \n<span class='rs-muted'>{inc['id']} · opened "
                    f"{fmt_ts(inc['opened_ts'])} · priority {inc['priority']}</span>", unsafe_allow_html=True)

st.subheader("Try it")
cols = st.columns(4)
cols[0].page_link("pages/1_Travel_Planner.py", label="Plan a journey with the agent", icon="🧭")
cols[1].page_link("pages/6_Support_Assistant.py", label="Ask the support assistant", icon="💬")
cols[2].page_link("pages/9_Chaos_Lab.py", label="Break something in the Chaos Lab", icon="🧪")
cols[3].page_link("pages/8_Control_Tower.py", label="Open the Control Tower", icon="🗼")
if fleet.get("last_cycle"):
    res = fleet["last_cycle"].get("resources") or {}
    st.caption(f"Last supervisor cycle {fmt_ts(fleet['last_cycle']['ts'])} ({fleet['last_cycle']['duration_ms']:.0f} ms) · "
               f"API memory {res.get('process_rss_mb')} MB · database {res.get('db_size_mb')} MB · "
               f"model artifacts {res.get('model_artifacts_mb')} MB")
