import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import time  # Added this for the timestamp trick

# -------- CONFIGURATION --------
st.set_page_config(
    page_title="Focus Monitoring Dashboard",
    layout="wide"
)


GITHUB_RAW_URL = "https://raw.githubusercontent.com/Dalmas-Omukaga/sync-flow/main/data/live/focus_predictions_live.csv"
# -------- DATA LOADING --------
def load_cloud_data():
    try:
        # Use a timestamp 't' to force GitHub to serve fresh data, not a cached version
        url = f"{GITHUB_RAW_URL}?t={int(time.time())}"
        df = pd.read_csv(url)
        return df if not df.empty else None
    except Exception as e:
        # If the file doesn't exist on GitHub yet, show a friendly message
        return None

st.title("🧠 Real-Time Focus Monitoring")

# Refresh the dashboard every 5 seconds to check for new GitHub pushes
st_autorefresh(interval=5000, key="datarefresh")

df = load_cloud_data()

# -------- UI LOGIC --------
if df is None:
    st.warning("📡 Waiting for live data from GitHub... Ensure your laptop is pushing logs.")
    st.stop()

# Get latest metrics
latest = df.iloc[-1]
# Make sure your CSV column names match exactly (e.g., 'focus_score' or 'gaze_score')
focus_val = latest.get("focus_score", 0)

# -------- Status Indicator --------
if focus_val > 75:
    st.success(f"### Status: Highly Focused 🚀 ({focus_val:.1f}%)")
elif focus_val > 45:
    st.info(f"### Status: Neutral / Calm ⚖️ ({focus_val:.1f}%)")
else:
    st.error(f"### Status: Distracted / Tired ⚠️ ({focus_val:.1f}%)")

# -------- Metrics --------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Focus Score", round(focus_val, 2))
col2.metric("Blink Rate", round(latest.get("blink_rate", 0), 2))
col3.metric("Gaze Score", round(latest.get("gaze_score", 0), 2))
col4.metric("Head Deviation", round(latest.get("head_deviation", 0), 2))

# -------- Charts --------
# Only show the last 200 data points for performance
df_display = df.tail(200)

st.subheader("Focus Trend")
fig_focus = px.line(df_display, y="focus_score", template="plotly_dark")
st.plotly_chart(fig_focus, use_container_width=True)

st.subheader("Gaze Stability")
fig_gaze = px.line(df_display, y="gaze_score", template="plotly_dark")
st.plotly_chart(fig_gaze, use_container_width=True)