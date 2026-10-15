"""Smart train recommender personalised by travel history."""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import client
from components import call, fmt_dt, meta, passport_card, setup

setup("Smart Train Recommender", "Ranks every train and class for your journey with a choice model learned from what "
      "passengers actually pick, personalised by your travel history.", "🚄")

m = meta()
c1, c2, c3, c4 = st.columns(4)
origin = c1.selectbox("From (city)", m["cities"], index=m["cities"].index("Mumbai"))
dest = c2.selectbox("To (city)", m["cities"], index=m["cities"].index("Delhi"))
jd = c3.date_input("Journey date", date.fromisoformat(m["today"]) + timedelta(days=10))
budget = c4.number_input("Budget per passenger (Rs, 0 = none)", 0, 50000, 2500, step=100)
st.markdown("**Your travel history** (personalisation uses behaviour, never demographics)")
h1, h2, h3 = st.columns(3)
usual = h1.selectbox("Usual class", ["SL", "3A", "2A", "1A", "CC", "2S"])
night = h2.slider("Share of past trips taken overnight", 0.0, 1.0, 0.6, 0.05)
fare_km = {"2S": 0.3, "SL": 0.48, "CC": 1.05, "3A": 1.32, "2A": 1.9, "1A": 3.15}[usual]
h3.metric("Typical fare paid per km (from history)", f"Rs {fare_km:.2f}")
if st.button("Recommend", type="primary"):
    with st.spinner("Ranking options..."):
        st.session_state["rec"] = call(client.post, "/recommend", {
            "origin": origin, "destination": dest, "journey_date": jd.isoformat(), "budget": budget or None,
            "profile": {"hist_class_mode": usual, "hist_night_share": night, "hist_fare_per_km": fare_km}})
res = st.session_state.get("rec")
if res:
    if not res.get("options"):
        st.warning(res.get("message", "No options."))
    else:
        st.caption(f"{res['candidates']} train/class options scored · budget used Rs {res['budget']:,.0f}")
        st.dataframe(pd.DataFrame([{
            "rank": o["rank"], "route": o["summary"], "class": o["travel_class"], "fare": o["fare"],
            "departs": fmt_dt(o["departure"]), "arrives": fmt_dt(o["arrival"]),
            "hours": round(o["duration_min"] / 60, 1), "availability": o["availability"],
            "confirm chance": f"{o['confirm_prob']:.0%}", "punctuality": f"{o['punctuality']:.0%}",
            "match": f"{o['match']:.0%}", "why": "; ".join(r["factor"] for r in o["rationale"])}
            for o in res["options"]]), hide_index=True, width="stretch")
        passport_card(res.get("passport"), f"rec_{res['prediction_id'][:8]}")
