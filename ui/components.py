"""Shared layout, styling and widgets for the Streamlit pages."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

import client

IST = timezone(timedelta(hours=5, minutes=30))
STATUS_COLORS = {"healthy": "#3fb950", "ok": "#3fb950", "resolved": "#3fb950", "warning": "#d29922",
                 "monitoring": "#d29922", "mitigating": "#d29922", "awaiting_approval": "#a371f7",
                 "critical": "#f85149", "breach": "#f85149", "open": "#f85149", "unknown": "#8b949e",
                 "insufficient": "#8b949e", "closed_by_reset": "#8b949e", "pending": "#d29922"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.rs-hero {border: 1px solid #263241; border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
  background: linear-gradient(120deg, #141c26 0%, #18202a 60%, #1d1a12 100%);}
.rs-hero h1 {font-size: 1.7rem; margin: 0 0 4px 0;}
.rs-hero p {color: #9fb0c3; margin: 0;}
.rs-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  color: #0f1419; text-transform: uppercase; letter-spacing: 0.03em;}
.rs-card {border: 1px solid #263241; border-radius: 12px; padding: 12px 14px; background: #151c25; height: 100%;}
.rs-card h4 {margin: 0 0 6px 0; font-size: 0.95rem;}
.rs-muted {color: #8b98a8; font-size: 0.85rem;}
.rs-mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; color: #9fb0c3;
  word-break: break-all;}
.rs-signal {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;}
</style>
"""


def setup(title: str, subtitle: str, icon: str = "🚆") -> dict | None:
    st.set_page_config(page_title=f"{title} · RailSentinel", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    health = client.health()
    with st.sidebar:
        st.markdown("### 🚆 RailSentinel")
        if health:
            mode = f"Gemini ({health['llm']['model']})" if health["llm"]["enabled"] else "Offline (no API key)"
            st.markdown(f"{signal('healthy')} API connected", unsafe_allow_html=True)
            st.caption(f"AI mode: {mode}")
            if health.get("scenario"):
                st.markdown(f"{signal('critical')} Chaos scenario: **{health['scenario']}**", unsafe_allow_html=True)
        else:
            st.markdown(f"{signal('critical')} API not reachable", unsafe_allow_html=True)
        st.caption("Data: synthetic passengers on the real Indian Railways station network (digital twin).")
    st.markdown(f"<div class='rs-hero'><h1>{icon} {title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)
    if not health:
        st.error(f"Cannot reach the API at {client.API_URL}. Start it with "
                 "`uvicorn railsentinel.api.main:app --port 8000` (see README), then refresh.")
        st.stop()
    return health


def signal(status: str) -> str:
    color = STATUS_COLORS.get(status, "#8b949e")
    return f"<span class='rs-signal' style='background:{color}'></span>"


def pill(status: str, text: str | None = None) -> str:
    color = STATUS_COLORS.get(status, "#8b949e")
    return f"<span class='rs-pill' style='background:{color}'>{text or status}</span>"


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, IST).strftime("%d %b %H:%M:%S")


def fmt_dt(iso: str | None) -> str:
    if not iso:
        return "-"
    return datetime.fromisoformat(iso).strftime("%a %d %b %H:%M")


def call(fn, *args, **kwargs):
    """Run an API call and show a readable error instead of a stack trace."""
    try:
        return fn(*args, **kwargs)
    except client.ApiError as exc:
        st.error(f"API error {exc.status}: {exc.detail}")
    except Exception as exc:  # network problems
        st.error(f"Request failed: {exc}")
    return None


@st.cache_data(ttl=300)
def meta() -> dict:
    return client.get("/meta")


def factor_chart(factors: list[dict], unit: str = ""):
    if not factors:
        return
    df = pd.DataFrame([{"factor": f"{f['label']} = {f['value']}", "impact": f["impact"]} for f in factors])
    chart = alt.Chart(df).mark_bar(cornerRadius=3).encode(
        x=alt.X("impact:Q", title=f"contribution ({unit})" if unit else "contribution"),
        y=alt.Y("factor:N", sort=None, title=None),
        color=alt.condition("datum.impact > 0", alt.value("#3fb950"), alt.value("#f85149")),
        tooltip=["factor", alt.Tooltip("impact:Q", format=".3f")],
    ).properties(height=max(120, 34 * len(df)))
    st.altair_chart(chart, width="stretch")


def passport_card(passport: dict | None, key: str):
    if not passport:
        return
    with st.expander("🛂 Explanation passport (signed)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", f"{passport['model']} v{passport['model_version']}")
        conf = passport.get("confidence")
        c2.metric("Confidence", "-" if conf is None else f"{conf:.0%}", passport.get("confidence_label"),
                  delta_color="off")
        gov = passport["governance"]
        c3.markdown(f"**Model health**<br>{pill(gov['model_health'])}<br><span class='rs-muted'>"
                    f"{gov['open_incidents']} open incident(s)</span>", unsafe_allow_html=True)
        dq = passport["data_quality"]
        c4.markdown(f"**Data contract**<br>{pill('ok' if dq['contract'] == 'pass' else 'breach', dq['contract'])}",
                    unsafe_allow_html=True)
        if passport.get("top_factors"):
            st.caption("Why: Shapley attribution of the prediction to its inputs "
                       f"(baseline {passport['top_factors'][0].get('base_value')})")
            factor_chart(passport["top_factors"], passport["top_factors"][0].get("unit", ""))
        st.markdown(f"<div class='rs-mono'>{passport['signature']}</div>", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 3])
        if b1.button("Verify signature", key=f"verify_{key}"):
            res = call(client.post, "/passports/verify", {"passport": passport})
            if res is not None:
                (b2.success if res["valid"] else b2.error)("Signature valid: passport is unaltered."
                                                           if res["valid"] else "Signature INVALID.")
        st.download_button("Download passport JSON", json.dumps(passport, indent=2), file_name=f"passport_{key}.json",
                           mime="application/json", key=f"dl_{key}")


def station_options() -> dict[str, str]:
    m = meta()
    return {s["code"]: f"{s['name']} ({s['code']}) - {s['city']}" for s in m["stations"]}
