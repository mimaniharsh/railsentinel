"""Ticket confirmation predictor for waitlisted and RAC tickets."""
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import client
from components import call, meta, passport_card, setup, station_options

setup("Ticket Confirmation Predictor", "Will a waitlisted or RAC ticket be confirmed by chart preparation? "
      "Probability, likely final status and the reasons behind it.", "🎫")


def show(res: dict):
    if res.get("status") == "not_applicable":
        st.info(res["message"])
        return
    if res.get("status") != "ok":
        st.error("Input blocked by the data contract: " +
                 "; ".join(f"{v['field']} {v['issue']}" for v in res["output"]["reasons"]))
        passport_card(res.get("passport"), "conf_blocked")
        return
    out = res["output"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Chance of confirmation (CNF)", f"{out['p_confirm']:.0%}")
    c2.metric("Chance of at least RAC", f"{out['p_at_least_rac']:.0%}")
    c3.metric("Most likely final status", out["predicted_final_status"])
    df = pd.DataFrame([{"status": k, "probability": v} for k, v in out["class_probs"].items()])
    st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=4).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("status:N", sort=["CNF", "RAC", "WL"]),
        color=alt.Color("status:N", scale=alt.Scale(domain=["CNF", "RAC", "WL"],
                                                    range=["#3fb950", "#d29922", "#f85149"]), legend=None)
    ).properties(height=130), width="stretch")
    passport_card(res.get("passport"), f"conf_{res['prediction_id'][:8]}")


tab1, tab2, tab3 = st.tabs(["My PNR", "What-if", "Historical trends"])
with tab1:
    pnrs = call(client.get, "/pnr", limit=100) or []
    waiting = [b for b in pnrs if b["current_status"].split()[0] in ("WL", "RAC")]
    if not waiting:
        st.info("No waitlisted demo PNRs right now.")
    else:
        labels = {b["pnr"]: f"{b['pnr']} · {b['train_number']} {b['origin']}→{b['destination']} · {b['journey_date']} · "
                            f"{b['travel_class']} · {b['current_status']}" for b in waiting}
        pnr = st.selectbox("Demo PNR", list(labels), format_func=labels.get)
        if st.button("Predict", type="primary", key="pnr_go"):
            res = call(client.post, "/predict/confirmation", {"pnr": pnr})
            if res:
                show(res)
with tab2:
    stations = station_options()
    c1, c2 = st.columns(2)
    origin = c1.selectbox("From", list(stations), index=list(stations).index("PUNE"), format_func=stations.get)
    dest = c2.selectbox("To", list(stations), index=list(stations).index("NZM"), format_func=stations.get)
    trains = call(client.get, "/trains", origin=origin, destination=dest) or []
    if not trains:
        st.info("No direct trains between these stations in the twin. Try another pair.")
    else:
        tlabels = {t["train_number"]: f"{t['train_number']} {t['train_name']} (dep {t['departure']})" for t in trains}
        tn = st.selectbox("Train", list(tlabels), format_func=tlabels.get)
        tr = next(t for t in trains if t["train_number"] == tn)
        classes = [c for c in tr["classes"] if c in ("SL", "CC", "3A", "2A", "1A")]
        c1, c2, c3, c4 = st.columns(4)
        cls = c1.selectbox("Class", classes or ["SL"])
        jd = c2.date_input("Journey date", date.fromisoformat(meta()["today"]) + timedelta(days=14))
        status = c3.selectbox("Current status", ["WL", "RAC"])
        pos = c4.number_input("Position", 1, 2000, 25)
        quota = st.radio("Quota", ["GN", "TQ"], horizontal=True, format_func=lambda q: {"GN": "General", "TQ": "Tatkal"}[q])
        if st.button("Predict", type="primary", key="whatif_go"):
            res = call(client.post, "/predict/confirmation", {"train_number": tn, "origin": origin, "destination": dest,
                                                             "travel_class": cls, "journey_date": jd.isoformat(),
                                                             "booking_status": status, "queue_position": int(pos),
                                                             "quota": quota})
            if res:
                show(res)
with tab3:
    st.caption("How waitlists on one train and class have cleared in the past: by waitlist position, by how early "
               "the ticket was booked and by month, plus how fast seats sell and the cancellation rate the operator "
               "reports.")
    stations = station_options()
    c1, c2 = st.columns(2)
    t_origin = c1.selectbox("From", list(stations), index=list(stations).index("PUNE"), format_func=stations.get,
                            key="tr_o")
    t_dest = c2.selectbox("To", list(stations), index=list(stations).index("NZM"), format_func=stations.get,
                          key="tr_d")
    t_trains = call(client.get, "/trains", origin=t_origin, destination=t_dest) or []
    if not t_trains:
        st.info("No direct trains between these stations in the twin. Try another pair.")
    else:
        t_labels = {t["train_number"]: f"{t['train_number']} {t['train_name']}" for t in t_trains}
        c1, c2 = st.columns(2)
        t_tn = c1.selectbox("Train", list(t_labels), format_func=t_labels.get, key="tr_t")
        t_tr = next(t for t in t_trains if t["train_number"] == t_tn)
        t_classes = [c for c in t_tr["classes"] if c in ("SL", "CC", "3A", "2A", "1A")]
        t_cls = c2.selectbox("Class", t_classes, key="tr_c")
        if t_classes and st.button("Analyse history", type="primary", key="tr_go"):
            with st.spinner("Analysing this train's booking history..."):
                st.session_state["trends"] = call(client.get, "/insights/confirmation-trends", train_number=t_tn,
                                                  travel_class=t_cls)
    trends = st.session_state.get("trends")
    if trends:
        a, b, c, d = st.columns(4)
        a.metric("Past WL/RAC bookings analysed", f"{trends['history_bookings']:,}")
        b.metric("Confirmed by chart time", f"{trends['overall']['confirmed_rate']:.0%}")
        c.metric("Usually confirmed up to", trends["usually_confirmed_up_to"] or "below WL 10",
                 help="Deepest starting waitlist band that ended confirmed at least half the time")
        d.metric("Usually RAC or better up to", trends["usually_rac_or_better_up_to"] or "below WL 10",
                 help="Deepest starting waitlist band that ended RAC or confirmed at least half the time")
        left, right = st.columns(2)
        with left:
            st.markdown("**Confirmation rate by starting waitlist position**")
            df = pd.DataFrame(trends["by_waitlist_position"])
            if not df.empty:
                st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=3, color="#e8a33d").encode(
                    x=alt.X("band:N", sort=None, title=None), y=alt.Y("confirmed_rate:Q", title="confirmed",
                                                                     axis=alt.Axis(format="%")),
                    tooltip=["band", "bookings", alt.Tooltip("confirmed_rate:Q", format=".0%")]).properties(height=240),
                    width="stretch")
            st.markdown("**Confirmation rate by booking lead time**")
            df = pd.DataFrame(trends["by_booking_lead"])
            if not df.empty:
                st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=3, color="#58a6ff").encode(
                    x=alt.X("band:N", sort=None, title="booked before departure"),
                    y=alt.Y("confirmed_rate:Q", title="confirmed", axis=alt.Axis(format="%")),
                    tooltip=["band", "bookings", alt.Tooltip("confirmed_rate:Q", format=".0%")]).properties(height=240),
                    width="stretch")
        with right:
            st.markdown("**Seasonality: confirmations and reported cancellations by month**")
            df = pd.DataFrame(trends["by_month"]).rename(columns={
                "confirmed_rate": "confirmed by chart time", "reported_cancellation_rate": "reported cancellation rate"})
            df = df.melt(id_vars=["month", "bookings"],
                         value_vars=["confirmed by chart time", "reported cancellation rate"],
                         var_name="series", value_name="rate").dropna()
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("month:O"), y=alt.Y("rate:Q", axis=alt.Axis(format="%")),
                color=alt.Color("series:N", scale=alt.Scale(range=["#3fb950", "#f85149"])),
                tooltip=["month", "series", alt.Tooltip("rate:Q", format=".1%")]).properties(height=240),
                width="stretch")
            st.markdown("**Booking curve: demand as a share of berths**")
            df = pd.DataFrame(trends["booking_curve"])
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("days_before_departure:Q", scale=alt.Scale(reverse=True), title="days before departure"),
                y=alt.Y("demand_vs_capacity:Q", title="demand / berths", axis=alt.Axis(format="%")),
                color="season:N", tooltip=["season", "days_before_departure",
                                           alt.Tooltip("demand_vs_capacity:Q", format=".0%")]).properties(height=240),
                width="stretch")
            st.caption("Above 100%, the extra demand is waitlisted.")
        st.caption(trends["data_note"])
