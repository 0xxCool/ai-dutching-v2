import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="⚽ Football Dashboard", layout="wide", page_icon="⚽")

st.title("⚽ AI Football Betting Dashboard")
st.success("✅ Dashboard is running successfully!")

# Simple metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Portfolio", "€10,000", "+12%")
with col2:
    st.metric("Profit", "€1,250", "+8%")
with col3:
    st.metric("Win Rate", "58%", "+2%")
with col4:
    st.metric("Status", "LIVE", "Active")

# Simple data
st.subheader("📊 Live Matches")
df = pd.DataFrame({
    'Match': ['Liverpool vs Arsenal', 'Real vs Barcelona', 'Bayern vs Dortmund'],
    'Score': ['2-1', '0-0', '3-2'],
    'Time': ['67 min', '23 min', 'FT'],
    'Best Odds': [2.10, 2.35, 1.85]
})
st.dataframe(df, use_container_width=True)

# Settings
with st.sidebar:
    st.header("⚙️ Settings")
    bankroll = st.slider("Bankroll (€)", 100, 10000, 1000)
    min_edge = st.slider("Min Edge (%)", -20, 10, -8)
    st.button("Apply Settings")
    
    st.info(f"""
    **Connection Info:**
    - Container: {st.secrets.get('hostname', 'unknown')}
    - Time: {datetime.now().strftime('%H:%M:%S')}
    - Port: 8501
    """)

st.balloons()
