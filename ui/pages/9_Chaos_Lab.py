"""Chaos Lab: break the platform on purpose and watch it heal."""
import streamlit as st

import client
from components import call, pill, setup

setup("Chaos Lab", "Inject a real failure. The supervisor is not told what you did; it has to detect, diagnose and "
      "repair it from evidence.", "🧪")

ICONS = {"incident_opened": "🚨", "remediation": "🛠️", "retrain_queued": "⏳", "promoted": "🚀",
         "shadow_rejected": "🧯", "incident_resolved": "✅", "approval_requested": "🙋", "incident_replanned": "🔁",
         "circuit_breaker_closed": "🔌", "feed_released": "📡"}


def describe(e: dict) -> str:
    t = e["type"]
    if t == "incident_opened":
        return f"Incident on **{e['asset']}**: {e['cause'].replace('_', ' ')} (root cause `{e['root_cause']}`)"
    if t == "remediation":
        r = e.get("result") or {}
        extra = ", ".join(f"{k}={v}" for k, v in r.items() if k in ("status", "version", "relative_gain", "from", "to",
                                                                   "quarantined", "circuit_breakers", "new_threshold"))
        return f"**{e['action']}** on {e['asset']} ({extra})"
    if t == "retrain_queued":
        r = e.get("result") or {}
        return f"Retrain of {e['asset']} queued: waiting for post-change labels ({r.get('labels')}/{r.get('needed')})"
    if t == "promoted":
        c = e.get("comparison") or {}
        return (f"Shadow v{e['version']} of {e['asset']} beat production on {c.get('labels')} live requests "
                f"({c.get('metric')} {c.get('production')} → {c.get('shadow')}): promoted")
    if t == "incident_resolved":
        return f"Incident on {e['asset']} verified fixed on fresh data and closed ({e['minutes']} min)"
    if t == "approval_requested":
        return f"**{e['action']}** on {e['asset']} needs a human: approve it in the Control Tower"
    if t == "incident_replanned":
        return f"New plan for {e['asset']}: {', '.join(e['actions'])}"
    return str({k: v for k, v in e.items() if k != "type"})


scenarios = call(client.get, "/chaos/scenarios") or []
active = next((s for s in scenarios if s["active"]), None)
if active:
    st.warning(f"Active scenario: **{active['title']}**")
c1, c2, c3, c4 = st.columns(4)
if c1.button("Generate passenger traffic"):
    with st.spinner("Simulating a batch of passengers..."):
        r = call(client.post, "/chaos/traffic", {"scale": 1.0})
    if r:
        st.toast(f"{sum(r['requests'].values())} requests in {r['duration_s']} s")
if c2.button("Run supervisor cycle"):
    r = call(client.post, "/gov/cycle")
    if r:
        for e in r["events"]:
            st.markdown(f"{ICONS.get(e['type'], '•')} {describe(e)}")
if c3.button("Clear scenario"):
    call(client.post, "/chaos/clear")
    st.rerun()
if c4.button("Reset demo to launch state"):
    call(client.post, "/chaos/reset")
    st.session_state.pop("autopilot", None)
    st.rerun()

st.subheader("Scenarios")
for i in range(0, len(scenarios), 3):
    cols = st.columns(3)
    for col, sc in zip(cols, scenarios[i:i + 3]):
        with col.container(border=True):
            st.markdown(f"**{sc['title']}** {pill('critical', 'active') if sc['active'] else ''}", unsafe_allow_html=True)
            st.caption(sc["story"])
            st.markdown(f"<span class='rs-muted'>Targets: {', '.join(sc['targets'])}"
                        f"{' · ships a bad release' if sc['deploys_release'] else ''}</span>", unsafe_allow_html=True)
            with st.popover("What should happen"):
                st.write(sc["expected"])
            a, b = st.columns(2)
            if a.button("Autopilot", key=f"auto_{sc['id']}", type="primary",
                        help="Reset to the launch state, activate, then alternate passenger traffic and "
                             "supervisor cycles until resolved"):
                with st.spinner("Running traffic and supervisor cycles (about 15-90 s)..."):
                    st.session_state["autopilot"] = call(client.post, "/chaos/autopilot",
                                                         {"scenario": sc["id"], "max_rounds": 8}, timeout=300)
            if b.button("Activate only", key=f"act_{sc['id']}",
                        help="Change the world now and keep the current models (use Reset for a clean start)"):
                call(client.post, "/chaos/activate", {"scenario": sc["id"]})
                st.rerun()

run = st.session_state.get("autopilot")
if run:
    st.subheader(f"Timeline: {run['scenario']}")
    outcome = {"resolved": "healthy", "awaiting_human_approval": "awaiting_approval", "monitoring": "monitoring"}
    st.markdown(f"Outcome {pill(outcome.get(run['outcome'], 'warning'), run['outcome'].replace('_', ' '))} after "
                f"{run['rounds']} round(s), {run['duration_s']} s", unsafe_allow_html=True)
    if run.get("activation", {}).get("deployed"):
        d = run["activation"]["deployed"]
        st.markdown(f"📦 Release pipeline deployed **{d['asset']} v{d['version']}** to production")
    for r in run["timeline"]:
        with st.container(border=True):
            status = ", ".join(f"{a}: {s}" for a, s in r["status"].items())
            st.markdown(f"**Round {r['round']}** · {sum(r['traffic']['requests'].values())} passenger requests · "
                        f"{status}")
            if not r["events"]:
                st.caption("No new events (collecting fresh data for verification).")
            for e in r["events"]:
                st.markdown(f"{ICONS.get(e['type'], '•')} {describe(e)}")
    if run["outcome"] == "awaiting_human_approval":
        st.page_link("pages/8_Control_Tower.py", label="Open the Control Tower to approve or reject", icon="🗼")
