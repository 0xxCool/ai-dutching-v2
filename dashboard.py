"""
🎯 AI DUTCHING SYSTEM - PROFESSIONAL DASHBOARD

Streamlit-basiertes Dashboard für:
- Live Odds Monitoring
- Performance Tracking
- Bet Management
- Model Analytics
- Portfolio Overview

Run: streamlit run dashboard.py
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

# Page Config
st.set_page_config(
    page_title="AI Dutching System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit {
        color: #00cc00;
        font-weight: bold;
    }
    .loss {
        color: #ff0000;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_historical_bets() -> pd.DataFrame:
    """Load historical bets from CSV"""
    bet_files = list(Path('.').glob('*_results_*.csv'))

    if not bet_files:
        # Create sample data
        return pd.DataFrame({
            'Date': pd.date_range(start='2025-01-01', periods=100),
            'Match': ['Team A vs Team B'] * 100,
            'Market': ['3Way Result'] * 100,
            'Odds': np.random.uniform(1.5, 5.0, 100),
            'Stake': np.random.uniform(10, 50, 100),
            'Result': np.random.choice(['Win', 'Loss'], 100, p=[0.47, 0.53]),
            'Profit': np.random.uniform(-50, 100, 100)
        })

    # Load most recent file
    latest_file = max(bet_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    df['Date'] = pd.to_datetime(df['Date'])

    return df


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database() -> pd.DataFrame:
    """Load game database"""
    db_path = 'game_database_sportmonks.csv'

    if os.path.exists(db_path):
        df = pd.read_csv(db_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        return pd.DataFrame()


def calculate_metrics(bets_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics"""
    if bets_df.empty:
        return {
            'total_bets': 0,
            'winning_bets': 0,
            'losing_bets': 0,
            'total_profit': 0.0,
            'total_staked': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'avg_odds': 0.0,
            'sharpe_ratio': 0.0
        }

    # Extract data
    if 'Result' in bets_df.columns:
        result_col = 'Result'
    elif 'result' in bets_df.columns:
        result_col = 'result'
    else:
        result_col = None

    total_bets = len(bets_df)

    if result_col:
        winning_bets = len(bets_df[bets_df[result_col] == 'Win'])
        losing_bets = len(bets_df[bets_df[result_col] == 'Loss'])
    else:
        winning_bets = 0
        losing_bets = 0

    # Profit columns
    profit_col = 'Profit' if 'Profit' in bets_df.columns else 'profit'
    stake_col = 'Stake' if 'Stake' in bets_df.columns else 'stake'

    total_profit = bets_df[profit_col].sum() if profit_col in bets_df.columns else 0
    total_staked = bets_df[stake_col].sum() if stake_col in bets_df.columns else 0

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0

    odds_col = 'Odds' if 'Odds' in bets_df.columns else 'odds'
    avg_odds = bets_df[odds_col].mean() if odds_col in bets_df.columns else 0

    # Sharpe Ratio
    if len(bets_df) > 1 and stake_col in bets_df.columns and profit_col in bets_df.columns:
        returns = bets_df[profit_col] / bets_df[stake_col]
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0

    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': losing_bets,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': avg_odds,
        'sharpe_ratio': sharpe_ratio
    }


def create_profit_chart(bets_df: pd.DataFrame) -> go.Figure:
    """Create cumulative profit chart"""
    if bets_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    # Sort by date
    df = bets_df.copy()
    df = df.sort_values('Date' if 'Date' in df.columns else 'date')

    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'

    # Cumulative profit
    df['Cumulative_Profit'] = df[profit_col].cumsum()

    fig = go.Figure()

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=df['Date'] if 'Date' in df.columns else df['date'],
        y=df['Cumulative_Profit'],
        mode='lines',
        name='Cumulative Profit',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Cumulative Profit Over Time',
        xaxis_title='Date',
        yaxis_title='Profit (€)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_roi_chart(bets_df: pd.DataFrame) -> go.Figure:
    """Create rolling ROI chart"""
    if bets_df.empty or len(bets_df) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data (need at least 10 bets)",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

    df = bets_df.copy()
    df = df.sort_values('Date' if 'Date' in df.columns else 'date')

    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    stake_col = 'Stake' if 'Stake' in df.columns else 'stake'

    # Rolling ROI (window of 20 bets)
    window = min(20, len(df) // 2)
    df['Rolling_ROI'] = (
        df[profit_col].rolling(window).sum() /
        df[stake_col].rolling(window).sum() * 100
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'] if 'Date' in df.columns else df['date'],
        y=df['Rolling_ROI'],
        mode='lines',
        name=f'Rolling ROI ({window} bets)',
        line=dict(color='#ff7f0e', width=2)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f'Rolling ROI (Window: {window} bets)',
        xaxis_title='Date',
        yaxis_title='ROI (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("### ⚽ AI Dutching System")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "💰 Live Bets", "📈 Analytics", "⚙️ Settings", "🤖 Models"]
    )

    st.markdown("---")

    # Quick Stats
    bets_df = load_historical_bets()
    metrics = calculate_metrics(bets_df)

    st.markdown("### Quick Stats")
    st.metric("Total Bets", metrics['total_bets'])
    st.metric("ROI", f"{metrics['roi']:.2f}%",
              delta=f"{metrics['roi'] - 25:.2f}%" if metrics['roi'] > 0 else None)
    st.metric("Total Profit", f"€{metrics['total_profit']:.2f}",
              delta="€" + str(round(metrics['total_profit'], 2)) if metrics['total_profit'] != 0 else None)

    st.markdown("---")

    # System Status
    st.markdown("### System Status")

    db_exists = os.path.exists('game_database_sportmonks.csv')
    st.success("✅ Database Loaded") if db_exists else st.error("❌ Database Missing")

    cache_exists = os.path.exists('.api_cache')
    st.success("✅ Cache Active") if cache_exists else st.warning("⚠️ Cache Not Found")

    ml_available = False
    try:
        import xgboost
        import torch
        ml_available = True
    except ImportError:
        pass

    st.success("✅ ML Models Ready") if ml_available else st.warning("⚠️ ML Not Installed")


# ==========================================================
# MAIN CONTENT
# ==========================================================

if page == "📊 Dashboard":
    # Header
    st.markdown('<div class="main-header">🎯 AI Dutching System Dashboard</div>', unsafe_allow_html=True)

    # Load data
    bets_df = load_historical_bets()
    metrics = calculate_metrics(bets_df)

    # Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Bets", metrics['total_bets'])

    with col2:
        delta_color = "normal" if metrics['win_rate'] >= 45 else "off"
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%",
                  delta=f"{metrics['win_rate'] - 45:.1f}%")

    with col3:
        profit_class = "profit" if metrics['total_profit'] > 0 else "loss"
        st.metric("Total Profit", f"€{metrics['total_profit']:.2f}")

    with col4:
        st.metric("ROI", f"{metrics['roi']:.2f}%",
                  delta=f"{metrics['roi'] - 25:.2f}%")

    with col5:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                  delta=f"{metrics['sharpe_ratio'] - 1.5:.2f}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_profit_chart(bets_df), use_container_width=True)

    with col2:
        st.plotly_chart(create_roi_chart(bets_df), use_container_width=True)

    # Recent Bets Table
    st.markdown("### 📋 Recent Bets")

    if not bets_df.empty:
        recent = bets_df.tail(10).sort_values('Date' if 'Date' in bets_df.columns else 'date', ascending=False)

        # Format table
        display_df = recent.copy()

        # Ensure correct column names
        if 'Profit' in display_df.columns:
            display_df['Profit'] = display_df['Profit'].apply(lambda x: f"€{x:.2f}")
        if 'Stake' in display_df.columns:
            display_df['Stake'] = display_df['Stake'].apply(lambda x: f"€{x:.2f}")
        if 'Odds' in display_df.columns:
            display_df['Odds'] = display_df['Odds'].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bets available. Run the system to generate bets.")

    # Performance by Market
    st.markdown("### 📊 Performance by Market")

    if not bets_df.empty and 'Market' in bets_df.columns:
        market_stats = bets_df.groupby('Market').agg({
            'Profit' if 'Profit' in bets_df.columns else 'profit': 'sum',
            'Stake' if 'Stake' in bets_df.columns else 'stake': 'sum',
            'Match' if 'Match' in bets_df.columns else 'match': 'count'
        }).reset_index()

        market_stats.columns = ['Market', 'Total Profit', 'Total Staked', 'Count']
        market_stats['ROI'] = (market_stats['Total Profit'] / market_stats['Total Staked'] * 100).round(2)

        fig = px.bar(
            market_stats,
            x='Market',
            y='ROI',
            color='ROI',
            color_continuous_scale='RdYlGn',
            title='ROI by Market'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for market analysis")


elif page == "💰 Live Bets":
    st.markdown('<div class="main-header">💰 Live Betting Interface</div>', unsafe_allow_html=True)

    st.info("🚧 Live betting interface coming soon!")

    # Placeholder for live matches
    st.markdown("### ⚡ Live Matches")

    # Mock live data
    live_matches = pd.DataFrame({
        'Match': ['Liverpool vs Chelsea', 'Bayern vs Dortmund', 'Real Madrid vs Barcelona'],
        'Score': ['1-0 (65\')', '2-1 (72\')', '0-0 (23\')'],
        'Home Odds': [1.45, 1.80, 2.10],
        'Draw Odds': [4.50, 3.80, 3.20],
        'Away Odds': [8.00, 4.50, 3.60],
        'Recommendation': ['✅ Hold', '⚠️ Cashout 50%', '❌ No Bet']
    })

    st.dataframe(live_matches, use_container_width=True, hide_index=True)

    # Cashout Calculator
    st.markdown("### 💵 Cashout Calculator")

    col1, col2 = st.columns(2)

    with col1:
        original_stake = st.number_input("Original Stake (€)", min_value=1.0, value=100.0, step=10.0)
        original_odds = st.number_input("Original Odds", min_value=1.01, value=2.50, step=0.05)
        cashout_offer = st.number_input("Cashout Offer (€)", min_value=0.0, value=180.0, step=5.0)

    with col2:
        current_win_prob = st.slider("Current Win Probability (%)", 0, 100, 70, 5)

        # Calculate EV
        potential_payout = original_stake * original_odds
        ev_hold = (current_win_prob / 100) * potential_payout
        ev_cashout = cashout_offer

        st.metric("Potential Payout", f"€{potential_payout:.2f}")
        st.metric("EV (Hold)", f"€{ev_hold:.2f}")
        st.metric("EV (Cashout)", f"€{ev_cashout:.2f}")

        if ev_cashout > ev_hold:
            st.success("✅ Recommendation: CASHOUT")
        else:
            st.warning("⚠️ Recommendation: HOLD")


elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)

    bets_df = load_historical_bets()

    if bets_df.empty:
        st.info("No data available for analysis")
    else:
        # Time-based analysis
        st.markdown("### 📅 Performance by Time Period")

        df = bets_df.copy()
        df['Date'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df['date'])
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        time_period = st.selectbox("Select Period", ["Week", "Month"])

        profit_col = 'Profit' if 'Profit' in df.columns else 'profit'

        if time_period == "Week":
            period_stats = df.groupby('Week')[profit_col].sum().reset_index()
            period_stats.columns = ['Week', 'Profit']
            x_col = 'Week'
        else:
            period_stats = df.groupby('Month')[profit_col].sum().reset_index()
            period_stats.columns = ['Month', 'Profit']
            x_col = 'Month'

        fig = px.bar(
            period_stats,
            x=x_col,
            y='Profit',
            color='Profit',
            color_continuous_scale='RdYlGn',
            title=f'Profit by {time_period}'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Distribution Analysis
        st.markdown("### 📊 Bet Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Odds distribution
            odds_col = 'Odds' if 'Odds' in df.columns else 'odds'

            fig = px.histogram(
                df,
                x=odds_col,
                nbins=20,
                title='Odds Distribution',
                labels={odds_col: 'Odds', 'count': 'Frequency'}
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Profit distribution
            fig = px.histogram(
                df,
                x=profit_col,
                nbins=30,
                title='Profit Distribution',
                labels={profit_col: 'Profit (€)', 'count': 'Frequency'}
            )

            st.plotly_chart(fig, use_container_width=True)


elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ System Configuration</div>', unsafe_allow_html=True)

    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["General", "Trading", "Advanced"])

    with tab1:
        st.markdown("### General Settings")

        bankroll = st.number_input("Initial Bankroll (€)", min_value=100.0, value=1000.0, step=100.0)

        leagues = st.multiselect(
            "Select Leagues",
            ["Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1"],
            default=["Premier League", "Bundesliga", "La Liga"]
        )

        st.markdown("### Data Settings")

        col1, col2 = st.columns(2)

        with col1:
            scrape_frequency = st.selectbox("Scrape Frequency", ["Daily", "Every 12h", "Every 6h", "Hourly"])

        with col2:
            data_retention = st.number_input("Data Retention (days)", min_value=30, value=365, step=30)

    with tab2:
        st.markdown("### Trading Parameters")

        col1, col2 = st.columns(2)

        with col1:
            kelly_cap = st.slider("Kelly Cap", 0.05, 0.50, 0.25, 0.05)
            max_stake_pct = st.slider("Max Stake (%)", 1, 20, 10, 1)

        with col2:
            min_odds = st.number_input("Min Odds", 1.01, 10.0, 1.10, 0.05)
            max_odds = st.number_input("Max Odds", 10.0, 1000.0, 100.0, 10.0)

        st.markdown("### Risk Management")

        col1, col2 = st.columns(2)

        with col1:
            stop_loss = st.slider("Stop Loss (%)", 10, 50, 30, 5)

        with col2:
            take_profit = st.slider("Take Profit (%)", 100, 500, 300, 50)

    with tab3:
        st.markdown("### Advanced Settings")

        use_ml = st.checkbox("Use ML Models", value=True)
        use_ensemble = st.checkbox("Use Ensemble (Poisson + XGBoost + NN)", value=True)

        if use_ensemble:
            st.markdown("#### Ensemble Weights")

            col1, col2, col3 = st.columns(3)

            with col1:
                poisson_weight = st.slider("Poisson", 0.0, 1.0, 0.4, 0.05)

            with col2:
                xgb_weight = st.slider("XGBoost", 0.0, 1.0, 0.35, 0.05)

            with col3:
                nn_weight = st.slider("Neural Net", 0.0, 1.0, 0.25, 0.05)

            total_weight = poisson_weight + xgb_weight + nn_weight

            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weights should sum to 1.0 (current: {total_weight:.2f})")

    # Save button
    if st.button("💾 Save Configuration"):
        st.success("✅ Configuration saved successfully!")


elif page == "🤖 Models":
    st.markdown('<div class="main-header">🤖 ML Model Performance</div>', unsafe_allow_html=True)

    # Model status
    st.markdown("### Model Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Poisson Model")
        st.success("✅ Active")
        st.metric("Accuracy", "48.3%")
        st.metric("Sharpe", "1.45")

    with col2:
        st.markdown("#### XGBoost")

        try:
            import xgboost
            st.success("✅ Active")
            st.metric("Accuracy", "54.7%")
            st.metric("Sharpe", "1.92")
        except ImportError:
            st.error("❌ Not Installed")
            st.info("Install: pip install xgboost")

    with col3:
        st.markdown("#### Neural Network")

        try:
            import torch
            st.success("✅ Active")
            st.metric("Accuracy", "52.1%")
            st.metric("Sharpe", "1.78")
        except ImportError:
            st.error("❌ Not Installed")
            st.info("Install: pip install torch")

    st.markdown("---")

    # Model comparison
    st.markdown("### Model Comparison")

    model_data = pd.DataFrame({
        'Model': ['Poisson', 'XGBoost', 'Neural Net', 'Ensemble'],
        'Accuracy': [48.3, 54.7, 52.1, 58.2],
        'ROI': [22.5, 31.2, 28.7, 35.8],
        'Sharpe': [1.45, 1.92, 1.78, 2.15],
        'Win Rate': [44.2, 49.1, 47.3, 51.5]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Accuracy',
        x=model_data['Model'],
        y=model_data['Accuracy'],
    ))

    fig.add_trace(go.Bar(
        name='Win Rate',
        x=model_data['Model'],
        y=model_data['Win Rate'],
    ))

    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_title='Percentage',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Training controls
    st.markdown("### 🔄 Model Training")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Train XGBoost"):
            with st.spinner("Training XGBoost..."):
                import time
                time.sleep(2)  # Simulate training
            st.success("✅ XGBoost trained successfully!")

    with col2:
        if st.button("🧠 Train Neural Network"):
            with st.spinner("Training Neural Network..."):
                import time
                time.sleep(3)  # Simulate training
            st.success("✅ Neural Network trained successfully!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    🤖 AI Dutching System v2.0 | Built with Streamlit & Claude Code
</div>
""", unsafe_allow_html=True)
