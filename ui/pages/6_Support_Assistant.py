"""Grounded customer-support assistant with live hallucination checks."""
import streamlit as st

import client
from components import call, pill, setup

setup("Support Assistant", "PNR status, refunds, cancellation charges, Tatkal, chart timing and journey planning. "
      "Every answer is checked against its evidence before you see it.", "💬")

if "chat" not in st.session_state:
    st.session_state["chat"] = []
pnrs = call(client.get, "/pnr", limit=50) or []
sample_pnr = next((b["pnr"] for b in pnrs if b["current_status"].startswith(("WL", "RAC"))), pnrs[0]["pnr"] if pnrs else "")
cancelled = next((b["pnr"] for b in pnrs if b["current_status"] == "CANCELLED"), sample_pnr)
suggestions = [f"Will my ticket {sample_pnr} get confirmed?", f"How much refund if I cancel PNR {sample_pnr}?",
               f"What is the refund status for PNR {cancelled}?", "Can I get a refund on a confirmed Tatkal ticket?",
               "When is the reservation chart prepared?", "What does GNWL mean?",
               "Find me a train from Mumbai to Goa tomorrow"]
cols = st.columns(4)
picked = None
for i, s in enumerate(suggestions):
    if cols[i % 4].button(s, key=f"sugg{i}"):
        picked = s
message = st.chat_input("Ask about a PNR, refund, cancellation, Tatkal, chart timing or a journey") or picked
if message:
    res = call(client.post, "/support/chat", {"message": message, "session_id": "ui"})
    if res:
        st.session_state["chat"].append({"q": message, "r": res})

for i, turn in enumerate(st.session_state["chat"]):
    with st.chat_message("user"):
        st.write(turn["q"])
    r = turn["r"]
    with st.chat_message("assistant"):
        st.write(r["answer"])
        g = r["grounding"]
        status = "breach" if g["hallucination"] else "ok"
        st.markdown(f"{pill(status, 'unsupported claims flagged' if g['hallucination'] else 'grounded')} "
                    f"<span class='rs-muted'>grounding {g['grounding_score']:.2f} · intent {r['intent']} · "
                    f"{r['mode']} engine · prompt v{r['prompt_version']} · {r['latency_ms']:.0f} ms</span>",
                    unsafe_allow_html=True)
        if g.get("unsupported_numbers") or g.get("unsupported_sentences"):
            st.caption(f"Unsupported: {g.get('unsupported_numbers') or ''} {g.get('unsupported_sentences') or ''}")
        if r["sources"]:
            st.caption("Sources: " + " · ".join(f"[{s['title']}]({s['url']})" for s in r["sources"] if s.get("url")))
        a, b, _ = st.columns([1, 1, 6])
        if a.button("👍", key=f"up{i}"):
            call(client.post, "/support/feedback", {"interaction_id": r["interaction_id"], "helpful": True})
            st.toast("Thanks for the feedback")
        if b.button("👎", key=f"down{i}"):
            call(client.post, "/support/feedback", {"interaction_id": r["interaction_id"], "helpful": False})
            st.toast("Feedback recorded; it feeds the satisfaction monitor")
