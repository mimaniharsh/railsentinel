"""Audit trail, governance reports, ethics checks and conversation/agent logs."""
import json

import pandas as pd
import streamlit as st

import client
from components import call, fmt_ts, pill, setup

setup("Audit & Reports", "Tamper-evident audit trail, governance report, ethical-AI checks, and full logs of agent "
      "decisions and support conversations.", "📜")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Audit trail", "Governance report", "Ethics checks", "Agent & chat logs",
                                        "Verify a passport"])
with tab1:
    c1, c2 = st.columns([1, 3])
    if c1.button("Verify hash chain", type="primary"):
        v = call(client.get, "/gov/audit/verify")
        if v:
            if v["valid"]:
                c2.success(f"Chain intact across {v['records']} records. Head {v['head'][:24]}...")
            else:
                c2.error(f"Chain broken at record {v['broken_at_seq']}: {v['reason']}")
    rows = call(client.get, "/gov/audit", limit=300) or []
    st.dataframe(pd.DataFrame([{"#": r["seq"], "time": fmt_ts(r["ts"]), "actor": r["actor"], "action": r["action"],
                                "target": r["target"], "details": json.dumps(r["details"])[:160],
                                "hash": r["hash"][:16]} for r in rows]), hide_index=True, width="stretch", height=420)
    st.caption("Each record stores the SHA-256 of the previous one. Editing or deleting any record breaks every hash after it.")
with tab2:
    report = call(client.get, "/gov/report")
    if report:
        st.download_button("Download report (Markdown)", report, file_name="railsentinel_governance_report.md",
                           mime="text/markdown")
        st.markdown(report)
with tab3:
    checks = call(client.get, "/gov/ethics") or []
    for c in checks:
        st.markdown(f"{pill('ok' if c['passed'] else 'critical', 'pass' if c['passed'] else 'fail')} **{c['check']}**  \n"
                    f"<span class='rs-muted'>{c['detail']}</span>", unsafe_allow_html=True)
with tab4:
    st.markdown("**Planner agent sessions**")
    traces = call(client.get, "/gov/agent/traces", limit=40) or []
    st.dataframe(pd.DataFrame([{"time": fmt_ts(t["ts"]), "source": t["source"], "mode": t["mode"], "query": t["query"],
                                "status": t["plan"].get("status"), "valid": t["verdict"].get("valid"),
                                "abnormal": t["verdict"].get("abnormal"), "steps": t["verdict"].get("steps"),
                                "issues": "; ".join(t["verdict"].get("violations", []) + t["verdict"].get("abnormal_reasons", []))}
                               for t in traces]), hide_index=True, width="stretch")
    st.markdown("**Support conversations** (PII masked before storage)")
    chats = call(client.get, "/gov/support/interactions", limit=60) or []
    st.dataframe(pd.DataFrame([{"time": fmt_ts(c["ts"]), "source": c["source"], "prompt": f"v{c['version']}",
                                "intent": c["intent"], "query": c["query"], "answer": c["response"][:160],
                                "grounding": c["grounding"].get("grounding_score"),
                                "hallucination": bool(c["hallucination"]),
                                "feedback": {1: "👍", -1: "👎"}.get(c["feedback"], "")} for c in chats]),
                 hide_index=True, width="stretch")
with tab5:
    raw = st.text_area("Paste a passport JSON (downloaded from any prediction)", height=200)
    if st.button("Verify"):
        try:
            res = call(client.post, "/passports/verify", {"passport": json.loads(raw)})
            if res:
                (st.success if res["valid"] else st.error)("Valid: issued by this platform and unaltered."
                                                           if res["valid"] else "Invalid: altered or not issued here.")
        except json.JSONDecodeError:
            st.error("That is not valid JSON.")
