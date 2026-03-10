import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from streamlit_autorefresh import st_autorefresh


DATA_PATH = "data/live/focus_predictions_live.csv"

st.set_page_config(
    page_title="Focus Monitoring Dashboard",
    layout="wide"
)

st.title("🧠 Real-Time Focus Monitoring")

# refresh every 2 seconds
st_autorefresh(interval=2000, key="datarefresh")


def load_data():

    if not Path(DATA_PATH).exists():
        return None

    try:
        df = pd.read_csv(DATA_PATH)
        return df if not df.empty else None
    except Exception:
        return None


df = load_data()

if df is None:

    st.warning("Waiting for focus data...")
    st.stop()

latest = df.iloc[-1]
focus_val = latest["focus_score"]


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
col2.metric("Blink Rate", round(latest["blink_rate"], 2))
col3.metric("Gaze Score", round(latest["gaze_score"], 2))
col4.metric("Head Deviation", round(latest["head_deviation"], 2))


# -------- Charts --------

df = df.tail(200)


st.subheader("Focus Trend")

fig_focus = px.line(
    df,
    y="focus_score",
    template="plotly_dark"
)

st.plotly_chart(fig_focus, use_container_width=True)


st.subheader("Blink Rate")

fig_blink = px.line(
    df,
    y="blink_rate",
    template="plotly_dark"
)

st.plotly_chart(fig_blink, use_container_width=True)


st.subheader("Gaze Stability")

fig_gaze = px.line(
    df,
    y="gaze_score",
    template="plotly_dark"
)

st.plotly_chart(fig_gaze, use_container_width=True)