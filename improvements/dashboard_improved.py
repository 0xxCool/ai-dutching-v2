"""
⚽ ADVANCED AI FOOTBALL BETTING DASHBOARD v5.1 - IMPROVED UI
============================================================
Verbesserte Übersichtlichkeit mit:
- Farbcodierten Tabellen
- Interaktiven Filtern
- Kompakter Log-Anzeige
- Besseren KPI-Cards
- Sortieroptionen
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys
import queue
from pathlib import Path
import json
import subprocess
import threading
import time
import asyncio
from dataclasses import dataclass
import requests
from PIL import Image
import base64
from io import BytesIO, StringIO
import signal
import logging
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
import streamlit_shadcn_ui as ui
from live_match_loader import load_live_matches, get_available_leagues
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path('logs').mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)

# GPU Monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    import torch
    
    GPU_AVAILABLE = True
    CUDA_VERSION_STR = torch.version.cuda if torch.cuda.is_available() else "N/A (PyTorch)"
    PYTORCH_VERSION_STR = torch.__version__

except ImportError:
    GPU_AVAILABLE = False
    CUDA_VERSION_STR = "N/A"
    PYTORCH_VERSION_STR = "N/A"

# System Imports
sys.path.append(str(Path.cwd()))
from unified_config import get_config, ConfigManager
from sportmonks_dutching_system import SportmonksFixtureClient as SportmonksClient, OptimizedDutchingCalculator, Config as DutchingConfig
from sportmonks_correct_score_system import CorrectScoreConfig, CorrectScorePoissonModel, CorrectScoreValueCalculator
from alert_system import AlertManager, AlertConfig, Alert, AlertLevel, AlertType
from portfolio_manager import PortfolioManager
from api_cache_system import FileCache as APICache, CacheConfig
from continuous_training_system import ModelRegistry, ContinuousTrainingEngine
from backtesting_framework import Backtester, BacktestConfig

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
# CUSTOM CSS FOR BETTER STYLING
# =============================================================================
st.markdown("""
<style>
    /* Kompaktere Metriken */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Highlight Boxes */
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .highlight-box h3 {
        margin: 0 0 10px 0;
        font-size: 20px;
    }
    
    .highlight-box .value {
        font-size: 32px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    .kpi-card .label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-card .value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin: 5px 0;
    }
    
    .kpi-card .change {
        font-size: 14px;
        color: #28a745;
    }
    
    .kpi-card .change.negative {
        color: #dc3545;
    }
    
    /* Kompakte Logs */
    .log-container {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    .log-line {
        padding: 3px 0;
        border-bottom: 1px solid #2d2d2d;
        color: #d4d4d4;
    }
    
    .log-line.error {
        color: #f48771;
    }
    
    .log-line.success {
        color: #89d185;
    }
    
    .log-line.warning {
        color: #e5c07b;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-badge.running {
        background: #d4edda;
        color: #155724;
    }
    
    .status-badge.stopped {
        background: #f8d7da;
        color: #721c24;
    }
    
    .status-badge.idle {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Dataframe Styling */
    .dataframe {
        font-size: 12px !important;
    }
    
    /* Compact Tab Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 14px;
    }
    
    /* Match Cards */
    .match-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .match-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .match-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .match-time {
        background: #667eea;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .teams {
        font-size: 16px;
        font-weight: 600;
        color: #333;
    }
    
    .odds-display {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    
    .odd-box {
        flex: 1;
        background: #f8f9fa;
        padding: 8px;
        border-radius: 6px;
        text-align: center;
    }
    
    .odd-box .label {
        font-size: 10px;
        color: #666;
        text-transform: uppercase;
    }
    
    .odd-box .value {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-top: 2px;
    }
    
    .odd-box.best {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .odd-box.best .label,
    .odd-box.best .value {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS FOR IMPROVED UI
# =============================================================================

def create_kpi_card(label: str, value: str, change: Optional[str] = None, change_positive: bool = True):
    """Erstellt eine kompakte KPI-Karte"""
    change_class = "" if change is None else ("change" if change_positive else "change negative")
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""
    
    return f"""
    <div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {change_html}
    </div>
    """

def format_dataframe_with_styling(df: pd.DataFrame, value_columns: List[str] = []) -> pd.DataFrame:
    """
    Formatiert einen DataFrame mit bedingter Formatierung
    
    Args:
        df: Der zu formatierende DataFrame
        value_columns: Liste der Spalten, die farblich hervorgehoben werden sollen
    """
    if df.empty:
        return df
    
    # Kopiere DataFrame
    styled_df = df.copy()
    
    # Formatierung für verschiedene Spaltentypen
    for col in styled_df.columns:
        # Prozentangaben
        if 'percent' in col.lower() or 'edge' in col.lower() or '%' in col:
            if styled_df[col].dtype in [np.float64, np.float32]:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        
        # Geldbeträge
        elif 'stake' in col.lower() or 'profit' in col.lower() or 'balance' in col.lower():
            if styled_df[col].dtype in [np.float64, np.float32]:
                styled_df[col] = styled_df[col].apply(lambda x: f"€{x:,.2f}" if pd.notna(x) else "-")
        
        # Odds
        elif 'odds' in col.lower():
            if styled_df[col].dtype in [np.float64, np.float32]:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    return styled_df

def display_compact_logs(logs: List[str], max_lines: int = 50, container=None):
    """
    Zeigt Logs in einer kompakten, farbcodierten Ansicht
    
    Args:
        logs: Liste der Log-Einträge
        max_lines: Maximale Anzahl anzuzeigender Zeilen
        container: Streamlit Container für die Ausgabe
    """
    if not logs:
        if container:
            container.info("📝 Keine Logs verfügbar")
        else:
            st.info("📝 Keine Logs verfügbar")
        return
    
    # Begrenze auf letzte N Zeilen
    display_logs = logs[-max_lines:]
    
    # Erstelle HTML für Logs
    log_html = '<div class="log-container">'
    
    for log in display_logs:
        # Bestimme Log-Typ
        log_class = "log-line"
        if "ERROR" in log or "FEHLER" in log.upper():
            log_class += " error"
        elif "SUCCESS" in log or "ERFOLG" in log.upper() or "✅" in log:
            log_class += " success"
        elif "WARNING" in log or "WARNUNG" in log.upper() or "⚠️" in log:
            log_class += " warning"
        
        log_html += f'<div class="{log_class}">{log}</div>'
    
    log_html += '</div>'
    
    if container:
        container.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(log_html, unsafe_allow_html=True)

def create_match_card(match_data: Dict) -> str:
    """
    Erstellt eine hübsche Match-Karte als HTML
    
    Args:
        match_data: Dictionary mit Match-Informationen
    """
    time_str = match_data.get('time', 'TBD')
    home_team = match_data.get('home_team', 'Home')
    away_team = match_data.get('away_team', 'Away')
    league = match_data.get('league', 'Unknown League')
    
    odds_1 = match_data.get('odds_1', '-')
    odds_x = match_data.get('odds_x', '-')
    odds_2 = match_data.get('odds_2', '-')
    
    # Finde beste Quote
    best_market = match_data.get('best_market', None)
    
    return f"""
    <div class="match-card">
        <div class="match-header">
            <span class="match-time">{time_str}</span>
            <span style="font-size: 12px; color: #666;">{league}</span>
        </div>
        <div class="teams">{home_team} vs {away_team}</div>
        <div class="odds-display">
            <div class="odd-box {'best' if best_market == '1' else ''}">
                <div class="label">Home</div>
                <div class="value">{odds_1}</div>
            </div>
            <div class="odd-box {'best' if best_market == 'X' else ''}">
                <div class="label">Draw</div>
                <div class="value">{odds_x}</div>
            </div>
            <div class="odd-box {'best' if best_market == '2' else ''}">
                <div class="label">Away</div>
                <div class="value">{odds_2}</div>
            </div>
        </div>
    </div>
    """

def create_status_badge(status: str) -> str:
    """Erstellt ein Status-Badge"""
    status_lower = status.lower()
    return f'<span class="status-badge {status_lower}">{status}</span>'

def display_dutching_results_improved(df: pd.DataFrame):
    """
    Zeigt Dutching-Ergebnisse mit verbesserter Übersichtlichkeit
    
    Args:
        df: DataFrame mit Dutching-Ergebnissen
    """
    if df.empty:
        st.info("📊 Noch keine Dutching-Ergebnisse verfügbar")
        return
    
    # Erstelle Tabs für verschiedene Ansichten
    tab1, tab2, tab3 = st.tabs(["📋 Alle Wetten", "⭐ Top Value", "📊 Statistiken"])
    
    with tab1:
        st.markdown("#### Alle verfügbaren Wetten")
        
        # Filter-Optionen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_ev = st.slider("Min. Expected Value (%)", -20, 50, 0, key="filter_ev")
        
        with col2:
            max_odds = st.slider("Max. Odds", 1.5, 20.0, 20.0, 0.5, key="filter_odds")
        
        with col3:
            min_stake = st.number_input("Min. Stake (€)", 0.0, 100.0, 0.0, key="filter_stake")
        
        # Filtern
        filtered_df = df.copy()
        if 'expected_value' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['expected_value'] >= min_ev]
        if 'odds' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['odds'] <= max_odds]
        if 'stake' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['stake'] >= min_stake]
        
        # Sortieroptionen
        sort_col = st.selectbox("Sortieren nach:", 
                                filtered_df.columns.tolist() if not filtered_df.empty else ['expected_value'],
                                key="sort_dutching")
        sort_order = st.radio("Reihenfolge:", ["Absteigend", "Aufsteigend"], 
                             horizontal=True, key="order_dutching")
        
        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values(
                by=sort_col, 
                ascending=(sort_order == "Aufsteigend")
            )
            
            # Zeige gefilterte und formatierte Daten
            styled_df = format_dataframe_with_styling(filtered_df)
            st.dataframe(styled_df, use_container_width=True, height=500)
            
            # Download-Button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Als CSV herunterladen",
                data=csv,
                file_name=f"dutching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Keine Wetten entsprechen den Filterkriterien")
    
    with tab2:
        st.markdown("#### Top Value Wetten (Best Expected Value)")
        
        # Top 10 nach Expected Value
        if 'expected_value' in df.columns:
            top_value = df.nlargest(10, 'expected_value')
            
            if not top_value.empty:
                # Zeige als Match-Karten wenn möglich
                for idx, row in top_value.iterrows():
                    match_info = {
                        'time': row.get('datetime', 'TBD'),
                        'home_team': row.get('home_team', 'Home'),
                        'away_team': row.get('away_team', 'Away'),
                        'league': row.get('league', 'Unknown'),
                        'odds_1': row.get('odds', '-'),
                        'odds_x': '-',
                        'odds_2': '-',
                        'best_market': '1'
                    }
                    
                    st.markdown(create_match_card(match_info), unsafe_allow_html=True)
                    
                    # Zeige zusätzliche Details
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Expected Value", f"{row.get('expected_value', 0):.2f}%")
                    with col2:
                        st.metric("Empfohlener Einsatz", f"€{row.get('stake', 0):.2f}")
                    with col3:
                        st.metric("Potentieller Gewinn", f"€{row.get('potential_profit', 0):.2f}")
                    with col4:
                        st.metric("Wahrscheinlichkeit", f"{row.get('probability', 0)*100:.1f}%")
                    
                    st.markdown("---")
            else:
                st.info("Keine Top Value Wetten gefunden")
        else:
            st.warning("Expected Value Spalte nicht gefunden")
    
    with tab3:
        st.markdown("#### Statistik-Übersicht")
        
        # Berechne Statistiken
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_kpi_card(
                "Gesamt Wetten",
                str(len(df)),
                f"+{len(df)} heute"
            ), unsafe_allow_html=True)
        
        with col2:
            if 'stake' in df.columns:
                total_stake = df['stake'].sum()
                st.markdown(create_kpi_card(
                    "Gesamteinsatz",
                    f"€{total_stake:,.2f}",
                    f"Ø €{total_stake/len(df):.2f} pro Wette"
                ), unsafe_allow_html=True)
        
        with col3:
            if 'expected_value' in df.columns:
                avg_ev = df['expected_value'].mean()
                st.markdown(create_kpi_card(
                    "Durchschn. EV",
                    f"{avg_ev:+.2f}%",
                    None,
                    avg_ev > 0
                ), unsafe_allow_html=True)
        
        # Verteilungsdiagramme
        if 'expected_value' in df.columns and 'odds' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # EV Distribution
                fig_ev = px.histogram(
                    df, 
                    x='expected_value',
                    nbins=30,
                    title="Expected Value Verteilung",
                    labels={'expected_value': 'Expected Value (%)'},
                    color_discrete_sequence=['#667eea']
                )
                fig_ev.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_ev, use_container_width=True)
            
            with col2:
                # Odds Distribution
                fig_odds = px.histogram(
                    df,
                    x='odds',
                    nbins=30,
                    title="Odds Verteilung",
                    labels={'odds': 'Odds'},
                    color_discrete_sequence=['#764ba2']
                )
                fig_odds.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_odds, use_container_width=True)

# =============================================================================
# HAUPTFUNKTION (Angepasst für verbesserte UI)
# =============================================================================

def main():
    """Hauptfunktion mit verbesserter UI"""
    
    # Initialize session state
    if 'log_manager' not in st.session_state:
        from dashboard import LogStreamManager, initialize_session_state
        st.session_state.log_manager = LogStreamManager()
        initialize_session_state()
    
    # Auto-refresh
    st_autorefresh(interval=5000, key="datarefresh")
    st.session_state.last_refresh = datetime.now()
    
    # Header mit Highlight
    st.markdown("""
    <div class="highlight-box">
        <h3>⚽ AI Football Betting System v5.1</h3>
        <div style="font-size: 14px; opacity: 0.9;">Verbesserte Übersichtlichkeit & Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stats = st.session_state.get('portfolio_stats', {})
    
    with col1:
        st.markdown(create_kpi_card(
            "Portfolio Balance",
            f"€{stats.get('total_value', 0):,.2f}",
            f"{stats.get('roi', 0):+.2f}% ROI",
            stats.get('roi', 0) > 0
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card(
            "Total Profit",
            f"€{stats.get('total_profit', 0):,.2f}",
            None
        ), unsafe_allow_html=True)
    
    with col3:
        running_count = sum(1 for p in st.session_state.process_states.keys()
                           if st.session_state.log_manager.is_running(p))
        st.markdown(create_kpi_card(
            "Systems Online",
            f"{running_count}/6",
            None
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_kpi_card(
            "Active Bets",
            f"{stats.get('active_bets', 0)}",
            None
        ), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_kpi_card(
            "Last Update",
            st.session_state.last_refresh.strftime('%H:%M:%S'),
            None
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Tabs (gleiche Struktur wie vorher, aber mit verbesserter Darstellung)
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚽ Live Matches & Dutching",
        "⚙️ System Control",
        "🎯 Correct Score",
        "💼 Portfolio & Analytics"
    ])
    
    with tab1:
        st.markdown("### ⚽ Live Matches & Dutching System")
        
        # Dutching Results mit verbesserter Ansicht
        dutching_file = Path.cwd() / "results" / "dutching_results.csv"
        if dutching_file.exists():
            try:
                df = pd.read_csv(dutching_file)
                display_dutching_results_improved(df)
            except Exception as e:
                st.error(f"Fehler beim Laden: {e}")
        else:
            st.info("📊 Noch keine Dutching-Ergebnisse verfügbar")
    
    with tab2:
        st.markdown("### ⚙️ System Control Center")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### System Status")
            
            # Zeige alle Prozess-Status mit Badges
            for process, state in st.session_state.process_states.items():
                is_running = st.session_state.log_manager.is_running(process)
                status = "RUNNING" if is_running else "STOPPED"
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{process.replace('_', ' ').title()}**")
                with col_b:
                    st.markdown(create_status_badge(status), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Unified Logs")
            
            # Alle Logs zusammenführen
            all_logs = []
            for process in st.session_state.process_states.keys():
                logs = st.session_state.log_manager.get_logs(process)
                all_logs.extend(logs)
            
            # Sortiere nach Zeitstempel
            all_logs.sort()
            
            # Zeige kompakte Logs
            display_compact_logs(all_logs, max_lines=100)
    
    with tab3:
        st.markdown("### 🎯 Correct Score Predictions")
        
        cs_file = Path.cwd() / "results" / "correct_score_results.csv"
        if cs_file.exists():
            try:
                cs_df = pd.read_csv(cs_file)
                display_dutching_results_improved(cs_df)  # Gleiche verbesserte Ansicht
            except Exception as e:
                st.error(f"Fehler: {e}")
        else:
            st.info("📊 Noch keine Correct Score Predictions verfügbar")
    
    with tab4:
        st.markdown("### 💼 Portfolio & Analytics")
        
        # Portfolio Übersicht mit Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Entwicklung")
            # Hier könnte ein Chart kommen
            st.info("Portfolio Chart wird geladen...")
        
        with col2:
            st.markdown("#### Performance Metriken")
            # Performance Metriken
            st.info("Performance Metriken werden geladen...")

# =============================================================================
# SIDEBAR (Kompakter)
# =============================================================================
with st.sidebar:
    st.markdown('## ⚽ AI Betting System')
    st.markdown("### v5.1 - Improved UI")
    st.markdown("---")
    
    # Kompakte System-Übersicht
    stats = st.session_state.get('portfolio_stats', {})
    
    st.metric("Balance", f"€{stats.get('total_value', 0):,.2f}")
    st.metric("ROI", f"{stats.get('roi', 0):.2f}%")
    
    st.markdown("---")
    
    # Quick Actions
    if ui.button(text="🔄 Refresh", variant="secondary", className="w-full"):
        st.rerun()
    
    if ui.button(text="▶️ Start All", variant="default", className="w-full btn-green"):
        st.info("Starting all systems...")
        st.rerun()
    
    if ui.button(text="⏸️ Stop All", variant="destructive", className="w-full"):
        if 'log_manager' in st.session_state:
            st.session_state.log_manager.stop_all()
        st.success("All systems stopped!")

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()