import streamlit as st

st.title("Sync-Flow Monitor")

focus = st.slider("Focus Score", 0, 100, 50)

st.metric("Current Focus", focus)