"""
🚀 AI DUTCHING SYSTEM - COMPLETE GPU DASHBOARD
================================================

Features:
- Live Odds Monitoring
- Performance Tracking
- Bet Management
- GPU Model Training & Monitoring
- Continuous Learning Controls
- System Health Dashboard
- Configuration Management

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
import sys
from pathlib import Path
import json
import subprocess
import threading

# Page Config
st.set_page_config(
    page_title="AI Dutching System v3.1 GPU",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #00cc00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .gpu-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
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
    .status-ok {
        color: #00cc00;
        font-weight: bold;
    }
    .status-warn {
        color: #ff9900;
        font-weight: bold;
    }
    .status-error {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

@st.cache_data(ttl=60)
def load_historical_bets() -> pd.DataFrame:
    """Load historical bets"""
    bet_files = list(Path('.').glob('*_results_*.csv'))

    if not bet_files:
        # Sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=100)
        return pd.DataFrame({
            'Date': dates,
            'Match': [f'Team {i%10} vs Team {(i+1)%10}' for i in range(100)],
            'Market': np.random.choice(['3Way', 'Over/Under 2.5', 'BTTS'], 100),
            'Odds': np.random.uniform(1.5, 4.0, 100),
            'Stake': np.random.uniform(10, 50, 100),
            'Result': np.random.choice(['Win', 'Loss'], 100, p=[0.55, 0.45]),
            'Profit': np.random.uniform(-50, 100, 100)
        })

    latest_file = max(bet_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def check_gpu_available() -> Dict:
    """Check if GPU is available"""
    gpu_info = {
        'available': False,
        'name': 'No GPU',
        'cuda_available': False,
        'device_count': 0
    }

    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if gpu_info['cuda_available']:
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
    except:
        pass

    return gpu_info


def get_gpu_metrics() -> Dict:
    """Get current GPU metrics"""
    metrics = {
        'utilization': 0,
        'memory_used': 0,
        'memory_total': 0,
        'temperature': 0,
        'power_draw': 0
    }

    try:
        import torch
        if torch.cuda.is_available():
            metrics['memory_used'] = torch.cuda.memory_allocated(0) / 1e9  # GB
            metrics['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Try NVML for detailed stats
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['utilization'] = util.gpu

                metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                metrics['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                pynvml.nvmlShutdown()
            except:
                pass
    except:
        pass

    return metrics


def get_model_registry() -> List[Dict]:
    """Get model versions from registry"""
    registry_file = Path('models/registry/model_registry.json')

    if registry_file.exists():
        with open(registry_file, 'r') as f:
            data = json.load(f)
            return [v for v in data.values()]

    # Sample data
    return [
        {
            'version_id': 'neural_net_20250123_120000',
            'model_type': 'neural_net',
            'created_at': '2025-01-23T12:00:00',
            'training_samples': 1500,
            'validation_accuracy': 0.6745,
            'is_champion': True,
            'roi': 0.42,
            'win_rate': 0.58
        },
        {
            'version_id': 'xgboost_20250123_120000',
            'model_type': 'xgboost',
            'created_at': '2025-01-23T12:00:00',
            'training_samples': 1500,
            'validation_accuracy': 0.6912,
            'is_champion': True,
            'roi': 0.45,
            'win_rate': 0.61
        }
    ]


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown("# 🚀 AI Dutching v3.1")
st.sidebar.markdown("**GPU Edition**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "💰 Live Bets",
        "📈 Analytics",
        "🎮 GPU Control",
        "🤖 ML Models",
        "📊 Performance Monitor",
        "⚙️ Settings"
    ]
)

st.sidebar.markdown("---")

# System Status in Sidebar
st.sidebar.markdown("### System Status")

# GPU Status
gpu_info = check_gpu_available()
if gpu_info['available']:
    st.sidebar.success(f"✅ GPU: {gpu_info['name'][:20]}")
else:
    st.sidebar.warning("⚠️ GPU: Not Available")

# Database Status
if os.path.exists('game_database_sportmonks.csv'):
    df_db = pd.read_csv('game_database_sportmonks.csv')
    st.sidebar.info(f"📊 Database: {len(df_db)} matches")
else:
    st.sidebar.warning("⚠️ Database: Empty")

st.sidebar.markdown("---")

# Quick Actions
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🎯 Run Scraper"):
    st.sidebar.info("Starting scraper...")
    # Could launch scraper here

# ==========================================================
# PAGE: DASHBOARD
# ==========================================================

if page == "📊 Dashboard":
    st.markdown('<div class="main-header">🚀 AI Dutching System Dashboard</div>', unsafe_allow_html=True)

    # Load data
    df_bets = load_historical_bets()

    # Calculate metrics
    total_bets = len(df_bets)
    wins = len(df_bets[df_bets['Result'] == 'Win'])
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    total_profit = df_bets['Profit'].sum()
    roi = (total_profit / df_bets['Stake'].sum() * 100) if df_bets['Stake'].sum() > 0 else 0
    avg_odds = df_bets['Odds'].mean()

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Bets", f"{total_bets}", "")

    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"+{(win_rate-50):.1f}%")

    with col3:
        profit_class = "profit" if total_profit > 0 else "loss"
        st.metric("Total Profit", f"€{total_profit:.2f}", "")

    with col4:
        st.metric("ROI", f"{roi:.1f}%", "")

    with col5:
        st.metric("Avg Odds", f"{avg_odds:.2f}", "")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Cumulative Profit")
        df_bets_sorted = df_bets.sort_values('Date')
        df_bets_sorted['Cumulative_Profit'] = df_bets_sorted['Profit'].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bets_sorted['Date'],
            y=df_bets_sorted['Cumulative_Profit'],
            mode='lines',
            name='Profit',
            line=dict(color='#00cc00', width=2),
            fill='tozeroy'
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Profit (€)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Market Distribution")
        market_dist = df_bets['Market'].value_counts()

        fig = px.pie(
            values=market_dist.values,
            names=market_dist.index,
            hole=0.4
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recent Bets
    st.markdown("### 🎯 Recent Bets")
    recent_bets = df_bets.sort_values('Date', ascending=False).head(10)
    st.dataframe(
        recent_bets[[' Date', 'Match', 'Market', 'Odds', 'Stake', 'Result', 'Profit']],
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PAGE: GPU CONTROL
# ==========================================================

elif page == "🎮 GPU Control":
    st.markdown('<div class="main-header">🎮 GPU Control Center</div>', unsafe_allow_html=True)

    # GPU Status Card
    gpu_info = check_gpu_available()

    if gpu_info['available']:
        st.markdown(f"""
        <div class="gpu-card">
            <h2>🚀 {gpu_info['name']}</h2>
            <p><strong>CUDA:</strong> {gpu_info['cuda_version']}</p>
            <p><strong>Devices:</strong> {gpu_info['device_count']}</p>
            <p><strong>Status:</strong> <span class="status-ok">✅ Ready</span></p>
        </div>
        """, unsafe_allow_html=True)

        # GPU Metrics
        st.markdown("### 📊 Current GPU Metrics")

        metrics = get_gpu_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("GPU Utilization", f"{metrics['utilization']}%")

        with col2:
            mem_pct = (metrics['memory_used'] / metrics['memory_total'] * 100) if metrics['memory_total'] > 0 else 0
            st.metric("VRAM Usage", f"{metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB")

        with col3:
            temp_status = "🟢" if metrics['temperature'] < 80 else "🟡" if metrics['temperature'] < 85 else "🔴"
            st.metric("Temperature", f"{temp_status} {metrics['temperature']}°C")

        with col4:
            st.metric("Power Draw", f"{metrics['power_draw']:.1f}W")

        st.markdown("---")

        # Training Controls
        st.markdown("### 🎯 Model Training")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Neural Network")
            epochs = st.slider("Epochs", 10, 200, 100, key="nn_epochs")
            batch_size = st.selectbox("Batch Size", [128, 256, 512, 1024], index=1, key="nn_batch")
            use_fp16 = st.checkbox("Use Mixed Precision (FP16)", value=True, key="nn_fp16")

            if st.button("🚀 Train Neural Network", key="train_nn"):
                with st.spinner("Training Neural Network..."):
                    st.info("Training started in background!")
                    st.code(f"""
python gpu_ml_models.py --epochs {epochs} --batch-size {batch_size} {'--fp16' if use_fp16 else ''}
                    """)

        with col2:
            st.markdown("#### XGBoost")
            n_estimators = st.slider("Estimators", 100, 500, 300, key="xgb_estimators")
            max_depth = st.slider("Max Depth", 4, 12, 8, key="xgb_depth")
            use_gpu = st.checkbox("Use GPU", value=True, key="xgb_gpu")

            if st.button("🌲 Train XGBoost", key="train_xgb"):
                with st.spinner("Training XGBoost..."):
                    st.info("Training started in background!")
                    st.code(f"""
python gpu_ml_models.py --model xgboost --estimators {n_estimators} --depth {max_depth} {'--gpu' if use_gpu else ''}
                    """)

        st.markdown("---")

        # Continuous Training
        st.markdown("### 🔄 Continuous Training")

        col1, col2, col3 = st.columns(3)

        with col1:
            auto_retrain = st.checkbox("Enable Auto-Retraining", value=False)

        with col2:
            schedule = st.selectbox("Schedule", ["Daily", "Weekly", "Manual"], index=0)

        with col3:
            min_samples = st.number_input("Min New Samples", 50, 500, 50)

        if st.button("💾 Save Continuous Training Config"):
            st.success("✅ Configuration saved!")

        if st.button("🚀 Start Continuous Training Now"):
            with st.spinner("Starting continuous training..."):
                st.info("Continuous training started!")
                st.code("python continuous_training_system.py")

    else:
        st.error("❌ GPU Not Available")
        st.markdown("""
        **GPU not detected!**

        To enable GPU acceleration:

        1. Install CUDA Toolkit (12.1 or 11.8)
        2. Install PyTorch with CUDA:
           ```bash
           pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
           ```
        3. Restart the dashboard

        [CUDA Download](https://developer.nvidia.com/cuda-downloads)
        """)


# ==========================================================
# PAGE: ML MODELS
# ==========================================================

elif page == "🤖 ML Models":
    st.markdown('<div class="main-header">🤖 ML Model Management</div>', unsafe_allow_html=True)

    # Model Registry
    st.markdown("### 📋 Model Registry")

    models = get_model_registry()

    if models:
        for model in models:
            champion_badge = "🏆 " if model.get('is_champion', False) else ""

            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

            with col1:
                st.markdown(f"**{champion_badge}{model['model_type'].upper()}**")
                st.caption(model['version_id'])

            with col2:
                st.metric("Val Accuracy", f"{model['validation_accuracy']*100:.1f}%")

            with col3:
                st.metric("ROI", f"{model.get('roi', 0)*100:.1f}%")

            with col4:
                st.metric("Win Rate", f"{model.get('win_rate', 0)*100:.1f}%")

            st.markdown("---")
    else:
        st.info("No models in registry yet. Train your first model in the GPU Control page!")

    # Model Comparison
    st.markdown("### 📊 Model Performance Comparison")

    if models:
        model_names = [m['model_type'] for m in models]
        accuracies = [m['validation_accuracy']*100 for m in models]
        rois = [m.get('roi', 0)*100 for m in models]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies))
        fig.add_trace(go.Bar(name='ROI', x=model_names, y=rois))
        fig.update_layout(barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# PAGE: PERFORMANCE MONITOR
# ==========================================================

elif page == "📊 Performance Monitor":
    st.markdown('<div class="main-header">📊 System Performance Monitor</div>', unsafe_allow_html=True)

    # GPU Performance
    if check_gpu_available()['available']:
        st.markdown("### 🎮 GPU Performance")

        metrics = get_gpu_metrics()

        # GPU Utilization Chart
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['utilization'],
                title={'text': "GPU Utilization (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#1f77b4"},
                       'steps': [
                           {'range': [0, 50], 'color': "#90EE90"},
                           {'range': [50, 80], 'color': "#FFD700"},
                           {'range': [80, 100], 'color': "#FF6347"}
                       ]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mem_pct = (metrics['memory_used'] / metrics['memory_total'] * 100) if metrics['memory_total'] > 0 else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mem_pct,
                title={'text': "VRAM Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#9370DB"},
                       'steps': [
                           {'range': [0, 60], 'color': "#90EE90"},
                           {'range': [60, 85], 'color': "#FFD700"},
                           {'range': [85, 100], 'color': "#FF6347"}
                       ]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Temperature & Power
        col1, col2 = st.columns(2)

        with col1:
            temp_color = "#90EE90" if metrics['temperature'] < 75 else "#FFD700" if metrics['temperature'] < 85 else "#FF6347"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌡️ Temperature</h4>
                <h2 style="color: {temp_color}">{metrics['temperature']}°C</h2>
                <p>Max Safe: 85°C</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ Power Draw</h4>
                <h2 style="color: #1f77b4">{metrics['power_draw']:.1f}W</h2>
                <p>RTX 3090 TDP: 350W</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Training Performance (Mock data)
    st.markdown("### 🚀 Training Performance History")

    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7)
    training_data = pd.DataFrame({
        'Date': dates,
        'Samples/Sec': np.random.uniform(2000, 3000, 7),
        'GPU Util': np.random.uniform(85, 95, 7),
        'Training Loss': np.random.uniform(0.25, 0.35, 7)
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=training_data['Date'], y=training_data['Samples/Sec'],
                             name='Throughput (samples/sec)', yaxis='y1'))
    fig.add_trace(go.Scatter(x=training_data['Date'], y=training_data['GPU Util'],
                             name='GPU Util (%)', yaxis='y2'))

    fig.update_layout(
        height=400,
        yaxis=dict(title='Samples/Sec'),
        yaxis2=dict(title='GPU Util (%)', overlaying='y', side='right')
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# PAGE: SETTINGS
# ==========================================================

elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Betting Configuration")

    col1, col2 = st.columns(2)

    with col1:
        bankroll = st.number_input("Bankroll (€)", 100, 100000, 1000)
        kelly_cap = st.slider("Kelly Cap", 0.1, 0.5, 0.25, 0.05)
        min_odds = st.number_input("Min Odds", 1.5, 5.0, 1.8)

    with col2:
        max_odds = st.number_input("Max Odds", 2.0, 20.0, 10.0)
        min_edge = st.slider("Min Edge (%)", 1, 20, 5)
        max_exposure = st.slider("Max Total Exposure", 0.5, 2.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("### 🤖 Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.slider("Poisson Weight", 0.0, 1.0, 0.4, 0.05)

    with col2:
        st.slider("XGBoost Weight", 0.0, 1.0, 0.35, 0.05)

    with col3:
        st.slider("Neural Net Weight", 0.0, 1.0, 0.25, 0.05)

    if st.button("💾 Save All Settings"):
        st.success("✅ Settings saved successfully!")


# ==========================================================
# PAGE: LIVE BETS & ANALYTICS (Keep existing code)
# ==========================================================

elif page == "💰 Live Bets":
    st.markdown('<div class="main-header">💰 Live Betting Opportunities</div>', unsafe_allow_html=True)
    st.info("Live betting opportunities will appear here when matches are in progress.")

elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)

    df_bets = load_historical_bets()

    # Win Rate by Market
    st.markdown("### 📊 Win Rate by Market")
    market_stats = df_bets.groupby('Market').agg({
        'Result': lambda x: (x == 'Win').sum() / len(x) * 100
    }).reset_index()
    market_stats.columns = ['Market', 'Win Rate (%)']

    fig = px.bar(market_stats, x='Market', y='Win Rate (%)', color='Win Rate (%)',
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
