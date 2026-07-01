"""
streamlit_app/pages/4_Risk_Dashboard.py

Risk Dashboard page: a visual summary of a user's most recent risk
evaluation plus distribution/trends across their history, combining
GET /api/v1/users/{id}, GET /api/v1/verification-history/{id}, and
GET /api/v1/risk-history/{id}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client
from config import LAYER1_FRAUD_THRESHOLD, SIMILARITY_THRESHOLD

st.set_page_config(page_title="PhaseGuard - Risk Dashboard", page_icon="📊", layout="wide")

st.title("📊 Risk Dashboard")
st.caption("Live overview of Layer 2 risk evaluations for an enrolled user")

default_user_id = st.session_state.get("enroll_user_id", "")
user_id = st.text_input("User ID (UUID)", value=default_user_id, key="dashboard_user_id")

if not user_id:
    st.info("Enter a User ID above to view their risk dashboard.")
    st.stop()

ok_user, user_payload = api_client.get_user(user_id)
if not ok_user:
    st.error(f"Could not find user: {user_payload}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Name", user_payload["name"])
col2.metric("Email", user_payload["email"])
col3.metric("Enrolled?", "Yes" if user_payload["is_enrolled"] else "No")
col4.metric("Recordings on file", user_payload.get("recording_count", 0))

st.divider()

ok_risk, risk_payload = api_client.get_risk_history(user_id, limit=100)
ok_ver, ver_payload = api_client.get_verification_history(user_id, limit=100)

if not ok_risk or not risk_payload:
    st.info("No risk evaluations recorded yet. Run a verification first.")
    st.stop()

risk_df = pd.DataFrame(risk_payload)
risk_df["created_at"] = pd.to_datetime(risk_df["created_at"])
risk_df = risk_df.sort_values("created_at")

latest = risk_df.iloc[-1]

# ----------------------------------------------------------------------
# Latest risk snapshot
# ----------------------------------------------------------------------
st.subheader("Latest evaluation")

snap_col1, snap_col2 = st.columns(2)

with snap_col1:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=latest["risk_score"],
            number={"suffix": "%"},
            title={"text": f"Risk Score — {latest['risk_level']}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "#f44336"
                    if latest["risk_level"] == "FRAUD_ALERT"
                    else "#4caf50"
                },
                "steps": [
                    {"range": [0, 50], "color": "#e8f5e9"},
                    {"range": [50, 100], "color": "#ffebee"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(fig_gauge, width='stretch')

with snap_col2:
    level_counts = risk_df["risk_level"].value_counts().reset_index()
    level_counts.columns = ["risk_level", "count"]

    fig_pie = px.pie(
        level_counts,
        names="risk_level",
        values="count",
        title="Risk Level Distribution (all-time)",
        color="risk_level",
        color_discrete_map={"CLEAN": "#4caf50", "FRAUD_ALERT": "#f44336"},
    )
    fig_pie.update_layout(height=300, margin=dict(t=50, b=10))
    st.plotly_chart(fig_pie, width='stretch')

st.divider()

# ----------------------------------------------------------------------
# Trend over time
# ----------------------------------------------------------------------
st.subheader("Risk score trend")

fig_trend = px.line(
    risk_df,
    x="created_at",
    y="risk_score",
    markers=True,
    title="Risk Score Over Time",
)
fig_trend.add_hline(
    y=50,
    line_dash="dash",
    line_color="orange",
    annotation_text="Mid-risk reference (50%)",
)
st.plotly_chart(fig_trend, width='stretch')

# ----------------------------------------------------------------------
# Verification similarity trend (if available)
# ----------------------------------------------------------------------
if ok_ver and ver_payload:
    st.subheader("Speaker similarity trend (Layer 2)")

    ver_df = pd.DataFrame(ver_payload)
    ver_df["created_at"] = pd.to_datetime(ver_df["created_at"])
    ver_df = ver_df.sort_values("created_at")

    fig_sim = px.line(
        ver_df,
        x="created_at",
        y="similarity_score",
        markers=True,
        title="Cosine Similarity Over Time",
    )
    fig_sim.add_hline(
        y=SIMILARITY_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Verification threshold ({SIMILARITY_THRESHOLD:.2f})",
    )
    st.plotly_chart(fig_sim, width='stretch')

st.divider()
st.caption(
    f"Risk Engine rules: layer1_score > {LAYER1_FRAUD_THRESHOLD:.2f} → FRAUD_ALERT  |  "
    f"speaker_similarity < {SIMILARITY_THRESHOLD:.2f} → FRAUD_ALERT  |  otherwise → CLEAN"
)
