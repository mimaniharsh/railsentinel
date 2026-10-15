"""Delay prediction, station congestion and proactive journey alerts."""
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import client
from components import call, fmt_dt, meta, passport_card, setup, station_options

setup("Delays & Disruptions", "Arrival and departure delay with an 80% range, station congestion through the day, "
      "and proactive alerts with alternatives for your booking.", "⏱️")
m = meta()
today = date.fromisoformat(m["today"])
tab1, tab2, tab3 = st.tabs(["Train delay", "Station congestion", "My journey alerts"])

with tab1:
    stations = station_options()
    c1, c2 = st.columns(2)
    origin = c1.selectbox("From", list(stations), index=list(stations).index("NDLS"), format_func=stations.get, key="d_o")
    dest = c2.selectbox("To", list(stations), index=list(stations).index("HWH"), format_func=stations.get, key="d_d")
    trains = call(client.get, "/trains", origin=origin, destination=dest) or []
    if trains:
        tlabels = {t["train_number"]: f"{t['train_number']} {t['train_name']}" for t in trains}
        tn = st.selectbox("Train", list(tlabels), format_func=tlabels.get)
        detail = call(client.get, f"/trains/{tn}")
        stops = [s["code"] for s in detail["stops"]][1:] if detail else [dest]
        c1, c2 = st.columns(2)
        station = c1.selectbox("Predict arrival at", stops, index=len(stops) - 1)
        run_date = c2.date_input("Run date (train's start date)", today)
        if st.button("Predict delay", type="primary"):
            res = call(client.post, "/predict/delay", {"train_number": tn, "station": station,
                                                      "run_date": run_date.isoformat()})
            if res and res["status"] == "ok":
                out, ctx = res["output"], res["context"]
                a, b, c, d = st.columns(4)
                a.metric("Expected arrival delay", f"{out['arrival_delay_min']:.0f} min", out["risk_level"], delta_color="off")
                b.metric("80% range", f"{out['interval_p10_p90'][0]:.0f}–{out['interval_p10_p90'][1]:.0f} min")
                c.metric("Expected departure delay", f"{out['departure_delay_min']:.0f} min")
                eta = ctx.get("expected_arrival")
                d.metric("Expected arrival", eta[11:16] if eta else "-", fmt_dt(eta)[:10] if eta else None,
                         delta_color="off")
                rs = ctx["running_status"]
                st.caption(f"Scheduled {fmt_dt(ctx['scheduled_arrival'])} · weather severity {ctx['weather_severity']:.2f} · "
                           + (f"running {rs['delay_min']:.0f} min late, last reported at {rs['last_station']}"
                              if rs["started"] else "train has not started yet"))
                passport_card(res["passport"], f"delay_{res['prediction_id'][:8]}")
            elif res:
                st.error(f"Blocked by data contract: {res['output']}")
    else:
        st.info("No direct trains between these stations.")

with tab2:
    stations = station_options()
    c1, c2 = st.columns(2)
    code = c1.selectbox("Station", list(stations), index=list(stations).index("NDLS"), format_func=stations.get, key="cg")
    day = c2.date_input("Date", today, key="cg_day")
    if st.button("Show congestion", type="primary"):
        res = call(client.get, f"/stations/{code}/congestion", day=day.isoformat())
        if res:
            df = pd.DataFrame({"hour": list(range(24)), "utilisation": res["hourly_platform_utilisation"],
                               "level": res["hourly_level"]})
            st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=3).encode(
                x=alt.X("hour:O", title="hour of day"), y=alt.Y("utilisation:Q", title="platform utilisation"),
                color=alt.Color("level:N", scale=alt.Scale(domain=["low", "moderate", "high"],
                                                           range=["#3fb950", "#d29922", "#f85149"])),
                tooltip=["hour", "utilisation", "level"]).properties(height=260), width="stretch")
            st.caption(f"{res['trains']} trains call here · {res['station']['platforms']} platforms (twin) · peak at "
                       f"{res['peak_hour']:02d}:00 ({res['peak_level']}). Arrivals are shifted by each train's predicted delay.")
            if res["predicted_disruptions"]:
                st.warning("Trains predicted more than an hour late here:")
                st.dataframe(pd.DataFrame(res["predicted_disruptions"]), hide_index=True, width="stretch")

with tab3:
    pnrs = call(client.get, "/pnr", limit=100) or []
    active = [b for b in pnrs if b["current_status"] != "CANCELLED"]
    if active:
        labels = {b["pnr"]: f"{b['pnr']} · {b['train_number']} {b['origin']}→{b['destination']} · {b['journey_date']} · "
                            f"{b['current_status']}" for b in active}
        pnr = st.selectbox("Demo PNR", list(labels), format_func=labels.get)
        if st.button("Check my journey", type="primary"):
            res = call(client.get, f"/pnr/{pnr}/alerts")
            if res:
                if not res["alerts"]:
                    st.success("No alerts: on-time arrival expected and the ticket status is fine.")
                for a in res["alerts"]:
                    st.warning(f"⚠️ {a['message']}")
                if res["alternatives"]:
                    st.subheader("Suggested alternatives")
                    st.dataframe(pd.DataFrame([{**a, "departure": fmt_dt(a["departure"]), "arrival": fmt_dt(a["arrival"]),
                                                "confirm_prob": f"{a['confirm_prob']:.0%}"} for a in res["alternatives"]]),
                                 hide_index=True, width="stretch")
