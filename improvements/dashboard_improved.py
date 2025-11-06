"""
⚽ AI FOOTBALL BETTING DASHBOARD v5.1 - IMPROVED EDITION
=========================================================
Komplett eigenständig - Keine externen Dependencies
Funktioniert mit ALLEN CSV-Dateinamen
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# PAGE CONFIG
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
    
    .stDataFrame {
        font-size: 14px;
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

def format_value(value: any, col_name: str) -> str:
    """Formatiert Werte basierend auf Spaltenname"""
    if pd.isna(value):
        return "-"
    
    col_lower = col_name.lower()
    
    # Prozente
    if any(x in col_lower for x in ['percent', 'edge', '%', '_ev', 'value']):
        try:
            return f"{float(value):+.2f}%"
        except:
            return str(value)
    
    # Währung
    elif any(x in col_lower for x in ['stake', 'profit', 'balance', 'euro', '€']):
        try:
            return f"€{float(value):,.2f}"
        except:
            return str(value)
    
    # Odds
    elif 'odds' in col_lower:
        try:
            return f"{float(value):.2f}"
        except:
            return str(value)
    
    return str(value)


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Formatiert gesamten DataFrame"""
    if df.empty:
        return df
    
    styled = df.copy()
    
    for col in styled.columns:
        if styled[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            styled[col] = styled[col].apply(lambda x: format_value(x, col))
    
    return styled


def load_latest_results(results_dir: Path, patterns: list) -> Optional[pd.DataFrame]:
    """
    Lädt die neueste CSV-Datei basierend auf mehreren Patterns
    
    Args:
        results_dir: Pfad zum results/ Ordner
        patterns: Liste von Glob-Patterns (z.B. ['*dutching*.csv', 'sportmonks_results_*.csv'])
    
    Returns:
        DataFrame oder None
    """
    if not results_dir.exists():
        return None
    
    # Sammle alle Dateien die irgendeinem Pattern entsprechen
    all_files = []
    for pattern in patterns:
        all_files.extend(list(results_dir.glob(pattern)))
    
    if not all_files:
        return None
    
    # Sortiere nach Änderungsdatum (neueste zuerst)
    all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Lade neueste Datei
    try:
        df = pd.read_csv(all_files[0])
        st.info(f"📊 Geladene Datei: `{all_files[0].name}` ({len(df)} Zeilen)")
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden von {all_files[0].name}: {e}")
        return None


def display_results_with_tabs(df: pd.DataFrame, title: str):
    """Zeigt Ergebnisse mit Tab-System"""
    if df.empty:
        st.info(f"📊 Keine {title} verfügbar")
        return
    
    tab1, tab2, tab3 = st.tabs(["📋 Alle Wetten", "⭐ Top 10", "📊 Statistiken"])
    
    # TAB 1: Alle Wetten mit Filtern
    with tab1:
        st.markdown(f"#### Alle {title}")
        
        col1, col2, col3 = st.columns(3)
        
        # Filter nur anzeigen wenn Spalten existieren
        with col1:
            if 'expected_value' in df.columns:
                min_ev = st.slider("Min. Expected Value (%)", -20, 50, 0, key=f"ev_{title}")
            else:
                min_ev = None
        
        with col2:
            if 'odds' in df.columns:
                max_odds = st.slider("Max. Odds", 1.5, 20.0, 20.0, 0.5, key=f"odds_{title}")
            else:
                max_odds = None
        
        with col3:
            if 'stake' in df.columns:
                min_stake = st.number_input("Min. Stake (€)", 0.0, 100.0, 0.0, key=f"stake_{title}")
            else:
                min_stake = None
        
        # Anwenden der Filter
        filtered = df.copy()
        
        if min_ev is not None and 'expected_value' in df.columns:
            filtered = filtered[filtered['expected_value'] >= min_ev]
        
        if max_odds is not None and 'odds' in df.columns:
            filtered = filtered[filtered['odds'] <= max_odds]
        
        if min_stake is not None and 'stake' in df.columns:
            filtered = filtered[filtered['stake'] >= min_stake]
        
        # Sortierung
        if not filtered.empty:
            sort_col = st.selectbox(
                "Sortieren nach:",
                filtered.columns.tolist(),
                key=f"sort_{title}"
            )
            
            sort_order = st.radio(
                "Reihenfolge:",
                ["Absteigend", "Aufsteigend"],
                horizontal=True,
                key=f"order_{title}"
            )
            
            filtered = filtered.sort_values(
                by=sort_col,
                ascending=(sort_order == "Aufsteigend")
            )
            
            # Zeige formatierte Daten
            styled = format_dataframe(filtered)
            st.dataframe(styled, use_container_width=True, height=500)
            
            # Download-Button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Als CSV herunterladen",
                data=csv,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{title}"
            )
        else:
            st.warning("⚠️ Keine Wetten entsprechen den Filterkriterien")
    
    # TAB 2: Top 10
    with tab2:
        st.markdown(f"#### Top 10 {title}")
        
        # Bestimme Sortier-Spalte
        sort_by = None
        if 'expected_value' in df.columns:
            sort_by = 'expected_value'
        elif 'value' in df.columns:
            sort_by = 'value'
        elif 'profit' in df.columns:
            sort_by = 'profit'
        
        if sort_by:
            top10 = df.nlargest(10, sort_by)
            
            for idx, row in top10.iterrows():
                # Erstelle dynamischen Titel
                match_title = "Match"
                if 'home_team' in row and 'away_team' in row:
                    match_title = f"{row['home_team']} vs {row['away_team']}"
                elif 'match' in row:
                    match_title = row['match']
                
                ev_text = ""
                if sort_by in row:
                    ev_text = f" - {sort_by}: {row[sort_by]:+.2f}"
                
                with st.expander(f"{match_title}{ev_text}"):
                    # Zeige wichtigste Metriken
                    cols = st.columns(4)
                    
                    metrics_to_show = []
                    if 'expected_value' in row:
                        metrics_to_show.append(('Expected Value', f"{row['expected_value']:.2f}%"))
                    if 'stake' in row:
                        metrics_to_show.append(('Stake', f"€{row['stake']:.2f}"))
                    if 'odds' in row:
                        metrics_to_show.append(('Odds', f"{row['odds']:.2f}"))
                    if 'potential_profit' in row:
                        metrics_to_show.append(('Pot. Profit', f"€{row['potential_profit']:.2f}"))
                    
                    for i, (label, value) in enumerate(metrics_to_show[:4]):
                        with cols[i]:
                            st.metric(label, value)
        else:
            st.info("Keine geeignete Sortier-Spalte gefunden")
    
    # TAB 3: Statistiken
    with tab3:
        st.markdown(f"#### Statistik-Übersicht {title}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gesamt Wetten", len(df))
        
        with col2:
            if 'stake' in df.columns:
                total_stake = df['stake'].sum()
                avg_stake = total_stake / len(df) if len(df) > 0 else 0
                st.metric("Gesamteinsatz", f"€{total_stake:,.2f}", f"Ø €{avg_stake:.2f}")
        
        with col3:
            if 'expected_value' in df.columns:
                avg_ev = df['expected_value'].mean()
                st.metric("Durchschn. EV", f"{avg_ev:+.2f}%")
        
        # Charts wenn möglich
        if 'expected_value' in df.columns:
            st.markdown("##### Expected Value Verteilung")
            fig = px.histogram(
                df,
                x='expected_value',
                nbins=30,
                labels={'expected_value': 'Expected Value (%)'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'odds' in df.columns:
            st.markdown("##### Odds Verteilung")
            fig = px.histogram(
                df,
                x='odds',
                nbins=30,
                labels={'odds': 'Odds'},
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Hauptfunktion"""
    
    st.session_state.last_refresh = datetime.now()
    
    # Header
    st.markdown("""
    <div class="highlight-box">
        <h3>⚽ AI Football Betting System v5.1</h3>
        <div style="font-size: 14px; opacity: 0.9;">Verbesserte Übersichtlichkeit & Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
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
    
    # Main Tabs
    tab1, tab2 = st.tabs(["⚽ Dutching System", "🎯 Correct Score"])
    
    results_path = Path('results')
    
    # TAB 1: Dutching
    with tab1:
        st.markdown("### ⚽ Dutching System")
        
        # WICHTIG: Suche nach ALLEN möglichen Dateinamen!
        dutching_patterns = [
            '*dutching*.csv',
            'sportmonks_results_*.csv',
            'sportmonks_dutching_*.csv',
            'results_*.csv'
        ]
        
        df = load_latest_results(results_path, dutching_patterns)
        
        if df is not None and not df.empty:
            display_results_with_tabs(df, "Dutching Wetten")
        else:
            st.info("📊 Noch keine Dutching-Ergebnisse verfügbar")
            
            with st.expander("💡 Wie bekomme ich Ergebnisse?"):
                st.markdown("""
                **Führe das Dutching-System aus:**
                
                ```python
                !python sportmonks_dutching_system.py
                ```
                
                **Oder verwende das Test-Skript:**
                
                ```python
                !python test_dutching.py
                ```
                
                Dann diese Seite neu laden (F5).
                """)
    
    # TAB 2: Correct Score
    with tab2:
        st.markdown("### 🎯 Correct Score System")
        
        # Suche nach Correct Score Dateien
        cs_patterns = [
            '*correct_score*.csv',
            'sportmonks_correct_score_*.csv',
            'cs_results_*.csv'
        ]
        
        df = load_latest_results(results_path, cs_patterns)
        
        if df is not None and not df.empty:
            display_results_with_tabs(df, "Correct Score Predictions")
        else:
            st.info("📊 Noch keine Correct Score-Ergebnisse verfügbar")
            
            with st.expander("💡 Wie bekomme ich Ergebnisse?"):
                st.markdown("""
                **Führe das Correct Score System aus:**
                
                ```python
                !python sportmonks_correct_score_system.py
                ```
                
                Dann diese Seite neu laden (F5).
                """)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('## ⚽ AI Betting System')
    st.markdown("### v5.1 - Improved")
    st.markdown("---")
    
    stats = st.session_state.portfolio_stats
    st.metric("Balance", f"€{stats['total_value']:,.2f}")
    st.metric("ROI", f"{stats['roi']:.2f}%")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    results_path = Path('results')
    if results_path.exists():
        csv_count = len(list(results_path.glob('*.csv')))
        st.info(f"📊 {csv_count} Result-Datei(en) gefunden")
    
    st.markdown("---")
    
    st.caption(f"**Last Update**: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    st.caption("**Version**: 5.1.0 Final")

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
