import streamlit as st

st.title("SyncFlow Real-Time Focus Monitor")

blink_rate = st.slider("Blink Rate")
typing_speed = st.slider("Typing Speed")

if st.button("Predict Focus"):
    st.success("Focused")