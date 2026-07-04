"""Risk dashboard for the authenticated account."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client
from config import SIMILARITY_THRESHOLD

st.set_page_config(page_title="PhaseGuard Risk", page_icon="RD", layout="wide")
st.title("Risk Dashboard")

user = st.session_state.get("current_user")
if not st.session_state.get("session_token") or not user:
    st.warning("Please log in from the main page first.")
    st.stop()

ok, profile = api_client.get_me()
if ok:
    user.update(profile)

c1, c2, c3 = st.columns(3)
c1.metric("Customer", user.get("full_name", "-"))
c2.metric("Account", user.get("account_number", "-"))
c3.metric("Voice enrolled", "Yes" if user.get("is_enrolled") else "No")

ok_risk, risk_payload = api_client.get_risk_history(limit=100)
ok_ver, ver_payload = api_client.get_verification_history(limit=100)

if not ok_risk:
    st.error(risk_payload)
    st.stop()
if not risk_payload:
    st.info("No risk evaluations recorded yet.")
    st.stop()

risk_df = pd.DataFrame(risk_payload)
risk_df["created_at"] = pd.to_datetime(risk_df["created_at"])
risk_df = risk_df.sort_values("created_at")
latest = risk_df.iloc[-1]

left, right = st.columns(2)
with left:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=latest["risk_score"],
            number={"suffix": "%"},
            title={"text": f"Risk Score - {latest['risk_level']}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#c62828" if latest["risk_level"] == "FRAUD_ALERT" else "#2e7d32"}},
        )
    )
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    counts = risk_df["risk_level"].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]
    st.plotly_chart(px.pie(counts, names="risk_level", values="count"), use_container_width=True)

st.plotly_chart(px.line(risk_df, x="created_at", y="risk_score", markers=True), use_container_width=True)

if ok_ver and ver_payload:
    ver_df = pd.DataFrame(ver_payload)
    ver_df["created_at"] = pd.to_datetime(ver_df["created_at"])
    fig_sim = px.line(ver_df.sort_values("created_at"), x="created_at", y="similarity_score", markers=True)
    fig_sim.add_hline(y=SIMILARITY_THRESHOLD, line_dash="dash", line_color="red")
    st.plotly_chart(fig_sim, use_container_width=True)
