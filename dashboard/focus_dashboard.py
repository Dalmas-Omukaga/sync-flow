import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import time

# -------- CONFIGURATION --------
st.set_page_config(
    page_title="Sync-Flow | Cloud Analytics",
    layout="wide"
)

# Your actual GitHub Raw URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/Dalmas-Omukaga/sync-flow/main/data/live/focus_predictions_live.csv"

def load_cloud_data():
    try:
        # Cache-buster to get fresh data from GitHub
        url = f"{GITHUB_RAW_URL}?t={int(time.time())}"
        
        # DEFINING COLUMN NAMES: This is what fixes the "stuck at 50" issue
        # Your tail output shows 6 columns: timestamp, gaze, blink, head, focus, state
        column_names = ["timestamp", "gaze_score", "blink_rate", "head_deviation", "focus_score", "state"]
        
        df = pd.read_csv(url, names=column_names)
        return df if not df.empty else None
    except Exception as e:
        st.toast(f"GitHub Fetching... {e}")
        return None

st.title("🧠 Sync-Flow: Real-Time Cloud Monitoring")

# Refresh the dashboard every 5 seconds to catch new Git pushes
st_autorefresh(interval=5000, key="datarefresh")

df = load_cloud_data()

# -------- UI LOGIC --------
if df is None:
    st.warning("📡 Connecting to GitHub Data Stream... Ensure your laptop is pushing logs.")
    st.stop()

# Get latest metrics
latest = df.iloc[-1]
focus_val = latest.get("focus_score", 0)

# -------- Status Indicator --------
if "distracted" in str(latest.get("state", "")).lower():
    st.error(f"### Status: Distracted ⚠️ ({focus_val:.1f}%)")
elif focus_val > 70:
    st.success(f"### Status: Highly Focused 🚀 ({focus_val:.1f}%)")
else:
    st.info(f"### Status: Neutral ⚖️ ({focus_val:.1f}%)")

# -------- Metrics --------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Focus Score", round(focus_val, 2))
col2.metric("Blink Rate", round(latest.get("blink_rate", 0), 2))
col3.metric("Gaze Score", round(latest.get("gaze_score", 0), 2))
col4.metric("Head Deviation", round(latest.get("head_deviation", 0), 2))

# -------- Charts --------
df_display = df.tail(100) # Last 100 points for a clean graph

st.subheader("Cloud Telemetry: Focus Trend")
fig_focus = px.line(df_display, x="timestamp", y="focus_score", template="plotly_dark", color_discrete_sequence=['#10b981'])
st.plotly_chart(fig_focus, use_container_width=True)

st.subheader("Environmental Metrics: Gaze & Stability")
fig_gaze = px.area(df_display, x="timestamp", y="gaze_score", template="plotly_dark", color_discrete_sequence=['#3b82f6'])
st.plotly_chart(fig_gaze, use_container_width=True)