from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from chart_review import analyze_screenshot, annotate_image
from storage import init_db, latest_scan, recent_scans

st.set_page_config(page_title="CRT Scanner V1", page_icon="📊", layout="wide")
init_db()

st.title("CRT Scanner V1")
st.caption("TradingView / GitHub Edition · analysis only · no automatic execution")

latest = latest_scan()
if latest is None:
    st.info("No TradingView scan has been received yet. Configure the webhook alert, then the latest AM/PM scan will appear here.")
    st.stop()

received = latest["received_at"]
if isinstance(received, datetime):
    received_text = received.strftime("%Y-%m-%d %H:%M:%S UTC")
else:
    received_text = str(received)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest scan", latest["scan_type"])
m2.metric("TradingView time", latest["time_ny"])
m3.metric("Payload version", latest["version"])
m4.metric("Received", received_text)

rows = []
for item in latest["symbols"]:
    rows.append({
        "Symbol": item["label"],
        "Daily bias": item["bias"],
        "Trend score": item["trend_score"],
        "C2 sweep": item["sweep"],
        "C2 close inside": "YES" if item["close_inside"] else "NO",
        "CRT": "VALID" if item["valid_crt"] else "INVALID",
        "CRT direction": item["direction"],
        "Bias alignment": "YES" if item["valid_crt"] and item["aligned"] else ("NO" if item["valid_crt"] else "-"),
    })

df = pd.DataFrame(rows)
st.subheader("7-symbol scanner")
st.dataframe(df, use_container_width=True, hide_index=True)

valid = [x for x in latest["symbols"] if x["valid_crt"]]
col_a, col_b = st.columns([1, 1])
with col_a:
    st.metric("Valid CRT setups", len(valid))
with col_b:
    st.metric("Bias-aligned valid setups", sum(1 for x in valid if x["aligned"]))

st.divider()
st.subheader("M15 / M5 manual-review workspace")
if not valid:
    st.info("No valid CRT setup exists in the latest scan, so screenshot review is disabled for this scan.")
else:
    labels = [x["label"] for x in valid]
    chosen = st.selectbox("Valid setup", labels)
    setup = next(x for x in valid if x["label"] == chosen)
    st.write(
        f"**{chosen}** · CRT {setup['direction']} · Daily bias {setup['bias']} · "
        f"Alignment {'YES' if setup['aligned'] else 'NO'} · C2 sweep {setup['sweep']}"
    )

    timeframe = st.radio("Screenshot timeframe", ["M15", "M5"], horizontal=True)
    upload = st.file_uploader("Upload TradingView screenshot", type=["png", "jpg", "jpeg", "webp"])

    if upload is not None:
        image_bytes = upload.getvalue()
        st.image(image_bytes, caption=f"{chosen} {timeframe} uploaded chart", use_container_width=True)
        ai_ready = bool(os.getenv("OPENAI_API_KEY"))
        if not ai_ready:
            st.warning("OPENAI_API_KEY is not configured. Scanner data still works; AI MSS/OB/FVG overlay is unavailable until the key is added to deployment secrets.")

        if st.button("Analyze MSS / Order Block / FVG", type="primary", disabled=not ai_ready):
            context = (
                f"symbol={chosen}, timeframe={timeframe}, crt_direction={setup['direction']}, "
                f"daily_bias={setup['bias']}, c2_sweep={setup['sweep']}, aligned={setup['aligned']}"
            )
            with st.spinner("Reviewing visible chart structure..."):
                try:
                    analysis = analyze_screenshot(image_bytes, upload.type or "image/png", context)
                    annotated = annotate_image(image_bytes, analysis)
                    st.image(annotated, caption="Annotated review mock-up", use_container_width=True)
                    st.metric("Review confidence", f"{analysis.confidence:.0%}")
                    st.write("**MSS:**", analysis.mss.model_dump())
                    if analysis.zones:
                        st.write("**Detected zones:**")
                        st.dataframe(pd.DataFrame([z.model_dump() for z in analysis.zones]), hide_index=True, use_container_width=True)
                    if analysis.notes:
                        st.write("**Review notes:**")
                        for note in analysis.notes:
                            st.write(f"- {note}")
                    st.caption("AI chart annotations are approximate visual review aids. Confirm structure directly on TradingView before using them in a trading decision.")
                except Exception as exc:
                    st.error(f"Chart review failed: {exc}")

st.divider()
with st.expander("Recent scanner runs"):
    history = recent_scans(30)
    if history:
        hist_df = pd.DataFrame([
            {
                "ID": x["id"],
                "Scan": x["scan_type"],
                "TradingView time": x["time_ny"],
                "Received": x["received_at"],
                "Version": x["version"],
            }
            for x in history
        ])
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
