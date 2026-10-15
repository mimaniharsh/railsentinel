"""Dynamic fare and sell-out forecaster with booking-window advice."""
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import client
from components import call, meta, passport_card, setup, station_options

setup("Fare & Demand Forecaster", "How the fare and seat availability will move until departure, when to book, "
      "and cheaper options. Premium trains use flexi-style pricing in the twin.", "💸")
m = meta()
stations = station_options()
c1, c2 = st.columns(2)
origin = c1.selectbox("From", list(stations), index=list(stations).index("PUNE"), format_func=stations.get)
dest = c2.selectbox("To", list(stations), index=list(stations).index("NDLS"), format_func=stations.get)
trains = call(client.get, "/trains", origin=origin, destination=dest) or []
if not trains:
    st.info("No direct trains between these stations.")
    st.stop()
tlabels = {t["train_number"]: f"{t['train_number']} {t['train_name']}" for t in trains}
c1, c2, c3 = st.columns(3)
tn = c1.selectbox("Train", list(tlabels), format_func=tlabels.get)
tr = next(t for t in trains if t["train_number"] == tn)
cls = c2.selectbox("Class", tr["classes"])
# 45 days out: inside the 60-day booking window, early enough that fares and availability still move
jd = c3.date_input("Journey date", date.fromisoformat(m["today"]) + timedelta(days=45))
with st.expander("Test the pricing anomaly check"):
    override = st.number_input("Override the fare-feed quote (Rs, 0 = use feed)", 0, 500000, 0)
if st.button("Forecast", type="primary"):
    res = call(client.post, "/predict/fare", {"train_number": tn, "origin": origin, "destination": dest,
                                             "travel_class": cls, "journey_date": jd.isoformat(),
                                             "quoted_fare": override or None})
    if res:
        st.info(res["advice"])
        demand = res.get("demand_outlook")
        if demand:
            (st.warning if demand["surge"] else st.caption)(
                f"{'📈 ' if demand['surge'] else ''}{demand['message']} About {demand['seats_booked_next_7_days']} "
                "more seats are expected to be booked in the next 7 days.")
        a, b, c = st.columns(3)
        a.metric("Availability now", res["availability"])
        a2 = res["curve"][0] if res["curve"] and "fare_now" in res["curve"][0] else None
        b.metric("Fare now", f"Rs {a2['fare_now']:,}" if a2 else "-")
        c.metric("Pricing", "dynamic (flexi-style)" if res["dynamic_fare"] else "fixed")
        ok = [x for x in res["curve"] if "fare_forecast" in x]
        if ok:
            df = pd.DataFrame(ok)
            base = alt.Chart(df).encode(x=alt.X("days_before_departure:Q", scale=alt.Scale(reverse=True),
                                                title="days before departure"))
            fare = base.mark_line(color="#e8a33d", strokeWidth=3).encode(y=alt.Y("fare_forecast:Q", title="fare (Rs)"))
            sell =base.mark_line(color="#f85149", strokeDash=[4, 3]).encode(
                y=alt.Y("sellout_probability:Q", title="sell-out probability", scale=alt.Scale(domain=[0, 1])))
            st.altair_chart(alt.layer(fare, sell).resolve_scale(y="independent").properties(height=300), width="stretch")
            st.caption("Orange: forecast fare. Red dashed: probability this class is sold out by that day.")
        else:
            st.error("Quote rejected by the data contract; no forecast was produced.")
        if res["cost_saving_options"]:
            st.subheader("Cheaper options on the same route")
            st.dataframe(pd.DataFrame(res["cost_saving_options"]), hide_index=True, width="stretch")
        h = res["headline"]
        if h["status"] == "ok" and h["output"].get("quote_anomaly"):
            st.warning("Pricing anomaly: the fare-feed quote disagrees with the pricing rule for current occupancy.")
        passport_card(h.get("passport"), f"fare_{h['prediction_id'][:8]}")
