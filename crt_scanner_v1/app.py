from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from chart_review import analyze_screenshot, annotate_image
from storage import init_db, latest_scan, recent_scans, recent_shadow_setups, update_shadow_outcome
from v2_policy import PRIMARY_COHORT, classify_model_eligibility

st.set_page_config(page_title="CRT Scanner V2", page_icon="📊", layout="wide")
init_db()

st.title("CRT Scanner V2")
st.caption("Bias-Aligned shadow validation · V1 control retained · analysis only · no automatic execution")

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
    policy = classify_model_eligibility(item)
    rows.append({
        "Symbol": item["label"],
        "Daily bias": item["bias"],
        "Trend score": item["trend_score"],
        "C1 sweep": item["sweep"],
        "C2 close inside": "YES" if item["close_inside"] else "NO",
        "CRT": "VALID" if item["valid_crt"] else "INVALID",
        "CRT direction": item["direction"],
        "Bias alignment": "YES" if item["valid_crt"] and item["aligned"] else ("NO" if item["valid_crt"] else "-"),
        "V1 control": "YES" if policy["v1_eligible"] else "-",
        "V2 candidate": "YES" if policy["v2_eligible"] else "-",
    })

df = pd.DataFrame(rows)
st.subheader("10-market scanner")
st.dataframe(df, use_container_width=True, hide_index=True)

valid = [x for x in latest["symbols"] if x["valid_crt"]]
primary_valid = [x for x in valid if x["label"] in PRIMARY_COHORT]
v2_candidates = [x for x in primary_valid if x["aligned"]]
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("All valid CRT setups", len(valid))
with col_b:
    st.metric("V1 control candidates", len(primary_valid), help="Valid CRTs in XAUUSD, US500 and US30")
with col_c:
    st.metric("V2 bias-aligned candidates", len(v2_candidates), help="V1 control candidate plus frozen D1/H4 bias alignment")

st.divider()
st.subheader("M15 / M5 manual-review workspace")
if not valid:
    st.info("No valid CRT setup exists in the latest scan, so screenshot review is disabled for this scan.")
else:
    labels = [x["label"] for x in valid]
    chosen = st.selectbox("Valid setup", labels)
    setup = next(x for x in valid if x["label"] == chosen)
    policy = classify_model_eligibility(setup)
    model_tag = "V2 CANDIDATE" if policy["v2_eligible"] else ("V1 CONTROL" if policy["v1_eligible"] else "OUTSIDE PRIMARY COHORT")
    st.write(
        f"**{chosen}** · CRT {setup['direction']} · Daily bias {setup['bias']} · "
        f"Alignment {'YES' if setup['aligned'] else 'NO'} · C1 sweep {setup['sweep']} · **{model_tag}**"
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
                f"daily_bias={setup['bias']}, c1_sweep={setup['sweep']}, aligned={setup['aligned']}, model_tag={model_tag}"
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
st.subheader("Prospective V1 vs V2 shadow ledger")
shadow = recent_shadow_setups(250)
if not shadow:
    st.info("No primary-cohort valid CRT has been recorded since the V2 shadow ledger was enabled.")
else:
    shadow_df = pd.DataFrame(shadow)
    resolved = shadow_df[shadow_df["model_r"].notna()].copy()
    v1_r = float(resolved["model_r"].sum()) if not resolved.empty else 0.0
    v2_resolved = resolved[resolved["v2_eligible"] == True].copy()  # noqa: E712
    v2_r = float(v2_resolved["model_r"].sum()) if not v2_resolved.empty else 0.0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tracked V1 setups", len(shadow_df))
    s2.metric("V2 candidates", int(shadow_df["v2_eligible"].sum()))
    s3.metric("Resolved clean R · V1", f"{v1_r:+.2f}R")
    s4.metric("Resolved clean R · V2", f"{v2_r:+.2f}R")

    ledger_cols = [
        "id", "time_ny", "scan_type", "label", "direction", "bias", "trend_score",
        "v1_eligible", "v2_eligible", "outcome", "model_r", "resolved_at",
    ]
    st.dataframe(shadow_df[ledger_cols], hide_index=True, use_container_width=True)

    unresolved = shadow_df[shadow_df["outcome"] == "PENDING"].copy()
    with st.expander("Resolve a shadow setup manually"):
        st.caption("This is a research/audit input only. It does not place or manage any broker order.")
        if unresolved.empty:
            st.success("No pending shadow setups.")
        else:
            options = {
                f"#{int(row['id'])} · {row['time_ny']} · {row['label']} · {row['direction']} · {'V2' if row['v2_eligible'] else 'V1 only'}": int(row["id"])
                for _, row in unresolved.iterrows()
            }
            with st.form("shadow_resolution"):
                display = st.selectbox("Pending setup", list(options.keys()))
                outcome = st.selectbox("Outcome", ["NO_ENTRY", "STOP", "BE", "TP2", "TIMEOUT", "AMBIGUOUS"])
                timeout_r = st.number_input("TIMEOUT R only", min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
                notes = st.text_area("Notes", max_chars=4000)
                submitted = st.form_submit_button("Save shadow outcome")
                if submitted:
                    model_r = float(timeout_r) if outcome == "TIMEOUT" else None
                    ok = update_shadow_outcome(options[display], outcome, model_r=model_r, notes=notes or None)
                    if ok:
                        st.success("Shadow outcome saved. Refresh the page to update the ledger metrics.")
                    else:
                        st.error("Could not find that shadow setup.")

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
