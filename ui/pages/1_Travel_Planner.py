"""Agentic travel planner: natural-language request in, audited itinerary out."""
import pandas as pd
import streamlit as st

import client
from components import call, fmt_dt, pill, setup

setup("Travel Planner Agent", "Describe your trip in plain English. Specialist agents check seats, delays and fares; "
      "an auditor validates the plan; nothing is booked until you approve.", "🧭")

examples = ["I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.",
            "Need 2 sleeper tickets Mumbai to Kolkata next sunday, arrive by 6 pm",
            "Cheapest way from Bengaluru to Chennai this friday",
            "Find the most reliable train from Delhi to Lucknow day after tomorrow"]
choice = st.selectbox("Example requests", examples)
query = st.text_area("Your request", value=choice, height=80)
if st.button("Plan my journey", type="primary"):
    with st.spinner("Agents are searching, scoring and checking options..."):
        st.session_state["plan"] = call(client.post, "/agent/plan", {"query": query})
        st.session_state.pop("booking", None)

plan = st.session_state.get("plan")
if plan:
    v = plan["verdict"]
    top = st.columns([2, 1, 1, 1])
    top[0].markdown(f"**Status:** {pill('ok' if plan['status'] == 'proposed' else 'warning', plan['status'])} "
                    f"&nbsp; **Engine:** {plan['mode']} &nbsp; **Agent config:** v{plan['agent_version']}",
                    unsafe_allow_html=True)
    top[1].metric("Auditor verdict", "valid" if v["valid"] else "VIOLATION")
    top[2].metric("Agent steps", v["steps"])
    top[3].metric("Latency", f"{plan['latency_ms']:.0f} ms")
    st.info(plan["message"].replace("\n", "\n\n"))
    if v["disclosed_deviations"]:
        st.warning("Disclosed trade-offs: " + "; ".join(v["disclosed_deviations"]))
    if v["violations"]:
        st.error("Auditor blocked booking: " + "; ".join(v["violations"]))

    best = plan.get("recommended")
    if best and best.get("legs"):
        st.subheader("Recommended itinerary")
        legs = pd.DataFrame([{"train": f"{l['train_number']} {l['train_name']}", "from": l["origin"], "to": l["destination"],
                              "departs": fmt_dt(l["departure"]), "arrives": fmt_dt(l["arrival"]), "class": l["travel_class"],
                              "fare": l["fare"], "availability": l["availability"],
                              "confirm chance": f"{l['confirm_prob']:.0%}", "expected delay (min)": l.get("delay_p50")}
                             for l in best["legs"]])
        st.dataframe(legs, hide_index=True, width="stretch")
        if best.get("rationale"):
            st.caption("Ranking rationale: " + "; ".join(r["factor"] for r in best["rationale"]))
        if plan.get("booking_intent_id") and not st.session_state.get("booking"):
            with st.form("approve"):
                name = st.text_input("Passenger name for the demo booking", "Demo Passenger")
                a, b = st.columns(2)
                ok = a.form_submit_button("Approve and book", type="primary")
                no = b.form_submit_button("Reject")
            if ok:
                st.session_state["booking"] = call(client.post, f"/agent/intents/{plan['booking_intent_id']}/approve",
                                                   {"passenger_name": name})
            if no:
                call(client.post, f"/agent/intents/{plan['booking_intent_id']}/reject")
                st.session_state["booking"] = {"rejected": True}
        bk = st.session_state.get("booking")
        if bk and bk.get("bookings"):
            st.success("Booked (demo PNRs in RailSentinel's own system): " +
                       ", ".join(f"PNR {b['pnr']} · {b['train']} · {b['status']}" for b in bk["bookings"]))
        elif bk and bk.get("rejected"):
            st.info("Booking intent rejected.")

    if plan.get("alternatives"):
        st.subheader("Alternatives")
        st.dataframe(pd.DataFrame([{"route": a["summary"], "class": a["travel_class"], "fare": a["fare"],
                                    "departs": fmt_dt(a["departure"]), "arrives": fmt_dt(a["arrival"]),
                                    "availability": a["availability"], "confirm chance": f"{a['confirm_prob']:.0%}",
                                    "on-time chance": "-" if a.get("on_time_prob") is None else f"{a['on_time_prob']:.0%}",
                                    "note": a.get("note", "")} for a in plan["alternatives"]]),
                     hide_index=True, width="stretch")

    with st.expander("Agent trace (every tool call, audited)", expanded=False):
        st.dataframe(pd.DataFrame([{"#": s["step"], "agent": s["agent"], "tool": s["tool"], "ok": s["ok"],
                                    "latency ms": s["latency_ms"], "result": s["summary"]} for s in plan["steps"]]),
                     hide_index=True, width="stretch")
        st.json(plan["constraints"], expanded=False)
