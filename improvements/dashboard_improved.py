"""
⚽ ADVANCED AI FOOTBALL BETTING DASHBOARD v5.1 - IMPROVED UI
============================================================
Verbesserte Übersichtlichkeit - STANDALONE VERSION
Keine Dependencies zu dashboard.py!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Football Betting System v5.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'portfolio_stats' not in st.session_state:
    st.session_state.portfolio_stats = {
        'total_value': 1000.0,
        'total_profit': 0.0,
        'roi': 0.0,
        'active_bets': 0
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Formatiert DataFrame"""
    if df.empty:
        return df
    
    styled = df.copy()
    
    for col in styled.columns:
        if '%' in col or 'edge' in col.lower():
            if styled[col].dtype in [np.float64, np.float32]:
                styled[col] = styled[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        
        elif 'stake' in col.lower() or 'profit' in col.lower():
            if styled[col].dtype in [np.float64, np.float32]:
                styled[col] = styled[col].apply(lambda x: f"€{x:,.2f}" if pd.notna(x) else "-")
        
        elif 'odds' in col.lower():
            if styled[col].dtype in [np.float64, np.float32]:
                styled[col] = styled[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    return styled


def display_results(df: pd.DataFrame):
    """Zeigt Ergebnisse mit Tabs"""
    if df.empty:
        st.info("📊 Keine Ergebnisse verfügbar")
        return
    
    tab1, tab2, tab3 = st.tabs(["📋 Alle", "⭐ Top 10", "📊 Stats"])
    
    with tab1:
        st.markdown("#### Alle Wetten")
        
        col1, col2 = st.columns(2)
        with col1:
            min_ev = st.slider("Min EV (%)", -20, 50, 0)
        with col2:
            max_odds = st.slider("Max Odds", 1.5, 20.0, 20.0, 0.5)
        
        filtered = df.copy()
        if 'expected_value' in filtered.columns:
            filtered = filtered[filtered['expected_value'] >= min_ev]
        if 'odds' in filtered.columns:
            filtered = filtered[filtered['odds'] <= max_odds]
        
        if not filtered.empty:
            st.dataframe(format_dataframe(filtered), use_container_width=True, height=500)
            
            csv = filtered.to_csv(index=False)
            st.download_button("📥 CSV Download", csv, "results.csv", "text/csv")
        else:
            st.warning("Keine Wetten gefunden")
    
    with tab2:
        st.markdown("#### Top 10 Value Wetten")
        
        if 'expected_value' in df.columns:
            top10 = df.nlargest(10, 'expected_value')
            
            for idx, row in top10.iterrows():
                with st.expander(f"{idx+1}. {row.get('home_team', 'Match')} - EV: {row.get('expected_value', 0):+.2f}%"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("EV", f"{row.get('expected_value', 0):.2f}%")
                    with col2:
                        st.metric("Stake", f"€{row.get('stake', 0):.2f}")
                    with col3:
                        st.metric("Odds", f"{row.get('odds', 0):.2f}")
    
    with tab3:
        st.markdown("#### Statistiken")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gesamt", len(df))
        
        with col2:
            if 'stake' in df.columns:
                st.metric("Total Stake", f"€{df['stake'].sum():,.2f}")
        
        with col3:
            if 'expected_value' in df.columns:
                st.metric("Avg EV", f"{df['expected_value'].mean():+.2f}%")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Hauptfunktion"""
    
    st.session_state.last_refresh = datetime.now()
    
    # Header
    st.markdown("""
    <div class="highlight-box">
        <h3>⚽ AI Football Betting System v5.1</h3>
        <div>Verbesserte Übersichtlichkeit & Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.portfolio_stats
    
    with col1:
        st.metric("Balance", f"€{stats['total_value']:,.2f}")
    with col2:
        st.metric("Profit", f"€{stats['total_profit']:,.2f}")
    with col3:
        st.metric("ROI", f"{stats['roi']:+.2f}%")
    with col4:
        st.metric("Active Bets", stats['active_bets'])
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2 = st.tabs(["⚽ Dutching", "🎯 Correct Score"])
    
    with tab1:
        st.markdown("### Dutching System")
        
        results_path = Path('results')
        if results_path.exists():
            files = list(results_path.glob('*dutching*.csv'))
            
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                try:
                    df = pd.read_csv(latest)
                    display_results(df)
                except Exception as e:
                    st.error(f"Fehler: {e}")
            else:
                st.info("Keine Dutching-Ergebnisse gefunden")
        else:
            st.info("Results-Ordner nicht gefunden")
    
    with tab2:
        st.markdown("### Correct Score")
        
        results_path = Path('results')
        if results_path.exists():
            files = list(results_path.glob('*correct_score*.csv'))
            
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                try:
                    df = pd.read_csv(latest)
                    display_results(df)
                except Exception as e:
                    st.error(f"Fehler: {e}")
            else:
                st.info("Keine Correct Score-Ergebnisse gefunden")
        else:
            st.info("Results-Ordner nicht gefunden")

# Sidebar
with st.sidebar:
    st.markdown('## ⚽ AI Betting System')
    st.markdown("### v5.1")
    st.markdown("---")
    
    stats = st.session_state.portfolio_stats
    st.metric("Balance", f"€{stats['total_value']:,.2f}")
    st.metric("ROI", f"{stats['roi']:.2f}%")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    st.markdown("---")
    st.info(f"**Last Update**: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
