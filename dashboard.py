"""
⚽ AI DUTCHING SYSTEM - COMPLETE GPU DASHBOARD
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
import pynvml
import time # Hinzugefügt für Live-Updates

# Page Config
st.set_page_config(
    page_title="AI Dutching System v3.1 GPU",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Dark-Mode optimiert mit "Fussball-Grün")
st.markdown("""
<style>
    /* ----- HINTERGRUNDBILD ----- */
    [data-testid="stAppViewContainer"] {
        /* Dunkler Farbverlauf (Overlay) */
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                          
                          /* URL zum Hintergrundbild (Stadion) */
                          url("https://images.unsplash.com/photo-1579952363873-27f3bade9f55?auto=format&fit=crop&w=1920&q=80");
        
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* ----- Transparenz für Hauptinhalt ----- */
    /* Macht den Haupt-Content-Block leicht durchsichtig, damit das Bild durchscheint */
    [data-testid="block-container"] {
        background-color: rgba(14, 17, 23, 0.85); /* 85% Deckkraft von stApp-Hintergrund */
        border-radius: 0.5rem;
        padding: 2rem;
    }

    /* ----- Transparenz für Sidebar ----- */
    [data-testid="stSidebar"] {
        background-color: rgba(26, 26, 31, 0.85); /* 85% Deckkraft von Sidebar-Hintergrund */
    }
            
    /* ----- Basis-Layout ----- */
    /* Haupt-Hintergrund (Streamlit Dark) */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }

    /* Sidebar-Hintergrund */
    [data-testid="stSidebar"] {
        background-color: #1a1a1f; /* Etwas helleres Dunkelgrau */
    }

    /* ----- Überschriften ----- */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF; /* Weiß im Dark-Mode */
        text-align: center;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50; /* Fussball-Grün Akzent */
        margin-top: 1rem;
    }

    /* ----- Karten-Styling ----- */
    .metric-card {
        background-color: #1a1a1f; /* Dunkler Kartenhintergrund */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50; /* Grüner Rand */
    }

    .gpu-card {
        /* Modernerer Gradient (Grün/Blau) */
        background: linear-gradient(135deg, #4CAF50 0%, #1f77b4 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    /* ----- Text-Akzente ----- */
    .profit {
        color: #00cc00; /* Helles Grün für Profit */
        font-weight: bold;
    }
    .loss {
        color: #ff4b4b; /* Helles Rot für Verlust */
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
        color: #ff4b4b;
        font-weight: bold;
    }
    
    /* ----- Sliders & Knöpfe (Grün-Akzent) ----- */
    
    /* Slider-Leiste */
    div[data-testid="stSlider"] > div[data-baseweb="slider"] > div:nth-child(2) > div {
        background: #4CAF50 !important;
    }
    /* Slider-Punkt */
    div[data-testid="stSlider"] > div[data-baseweb="slider"] > div:nth-child(3) {
        background-color: #4CAF50 !important;
    }
    
    /* Checkbox */
    div[data-testid="stCheckbox"] span {
        border-color: #4CAF50 !important;
    }
    div[data-testid="stCheckbox"] input:checked + div[data-baseweb="checkbox"] > span {
         background-color: #4CAF50 !important;
         border-color: #4CAF50 !important;
    }

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

# (Wir belassen die Beispieldaten in load_historical_bets, falls die CSVs nicht da sind)
@st.cache_data(ttl=60)
def load_historical_bets() -> pd.DataFrame:
    """Load historical bets"""
    bet_files = list(Path('.').glob('sportmonks_results_*.csv'))

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

    try:
        latest_file = max(bet_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Konnte Wett-Ergebnisse nicht laden: {e}")
        return pd.DataFrame() # Leeres DF zurückgeben


@st.cache_resource
def get_gpu_monitor():
    """Initialisiert den GPU-Monitor (pynvml) einmal."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        print("PYNVML initialisiert.")
        return handle
    except Exception as e:
        print(f"PYNVML Init fehlgeschlagen: {e}")
        return None

# GPU-Handle global initialisieren
GPU_HANDLE = get_gpu_monitor()

@st.cache_data(ttl=2) # Cache für 2 Sekunden
def check_gpu_available() -> Dict:
    """Check if GPU is available"""
    gpu_info = {
        'available': False,
        'name': 'No GPU',
        'cuda_available': False,
        'device_count': 0,
        'cuda_version': 'N/A'
    }

    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if gpu_info['cuda_available']:
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
    except Exception as e:
        print(f"Fehler beim Prüfen von torch: {e}")

    return gpu_info

@st.cache_data(ttl=2) # Cache für 2 Sekunden
def get_gpu_metrics(_handle, gpu_info) -> Dict: # KORREKTUR 1: Argument heißt _handle
    """Get current GPU metrics"""
    metrics = {
        'utilization': 0,
        'memory_used': 0,
        'memory_total': gpu_info.get('total_vram', 0),
        'temperature': 0,
        'power_draw': 0
    }

    try:
        import torch
        if torch.cuda.is_available():
            # PyTorch-Metriken (immer verfügbar, aber weniger detailliert)
            metrics['memory_used'] = torch.cuda.memory_allocated(0) / 1e9  # GB
            props = torch.cuda.get_device_properties(0)
            metrics['memory_total'] = props.total_memory / 1e9

            # NVML Stats (detaillierter)
            if _handle: # KORREKTUR 2: Prüfe _handle
                try:
                    # KORREKTUR 3: Verwende _handle
                    util = pynvml.nvmlDeviceGetUtilizationRates(_handle) 
                    metrics['utilization'] = util.gpu

                    # KORREKTUR 4: Verwende _handle
                    metrics['temperature'] = pynvml.nvmlDeviceGetTemperature(
                        _handle, pynvml.NVML_TEMPERATURE_GPU 
                    )

                    # KORREKTUR 5: Verwende _handle
                    metrics['power_draw'] = pynvml.nvmlDeviceGetPowerUsage(_handle) / 1000.0
                    
                    # Wir rufen nvmlShutdown() hier nicht auf, da der Handle 
                    # von @st.cache_resource verwaltet wird
                except Exception as e:
                    # pynvml kann fehlschlagen, wenn der Prozess beendet wird
                    print(f"NVML-Fehler beim Lesen: {e}")
                    pass 
    except Exception as e:
        print(f"Fehler beim Abrufen der GPU-Metriken: {e}")

    return metrics

@st.cache_data(ttl=30)
def get_model_registry() -> List[Dict]:
    """Get model versions from registry"""
    registry_file = Path('models/registry/model_registry.json')

    if registry_file.exists():
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
                return [v for v in data.values()]
        except json.JSONDecodeError:
            st.error("Fehler: model_registry.json ist korrumpiert.")
            return []
    return [] # Leere Liste, wenn keine Registry da ist


@st.cache_data(ttl=5)
def read_log_file(log_path: str) -> str:
    """Liest den Inhalt einer Log-Datei."""
    log_file = Path(log_path)
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                # Lese die letzten 100 Zeilen
                lines = f.readlines()
                return "".join(lines[-100:])
        except Exception as e:
            return f"Fehler beim Lesen der Log-Datei: {e}"
    return f"Log-Datei nicht gefunden: {log_path}\nTraining läuft möglicherweise noch nicht."


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown("# ⚽ AI Dutching v3.1")
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
db_path = 'game_database_sportmonks.csv'
if 'db_matches_count' not in st.session_state:
    st.session_state.db_matches_count = 0

if os.path.exists(db_path):
    try:
        df_db = pd.read_csv(db_path)
        st.session_state.db_matches_count = len(df_db)
        if st.session_state.db_matches_count > 0:
            st.sidebar.info(f"📊 Database: {st.session_state.db_matches_count} matches")
        else:
            st.sidebar.warning("⚠️ Database: Empty (0 rows)")
    except pd.errors.EmptyDataError:
        st.session_state.db_matches_count = 0
        st.sidebar.warning("⚠️ Database: Empty (file is empty)")
    except Exception as e:
        st.session_state.db_matches_count = -1
        st.sidebar.error("⚠️ DB Load Error")
else:
    st.session_state.db_matches_count = 0
    st.sidebar.warning("⚠️ Database: Not found")
    
st.sidebar.markdown("---")

# Quick Actions
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# === NEUER KNOPF: SCRAPER STARTEN ===
if st.sidebar.button("🎯 Run Scraper (Get Data)"):
    with st.sidebar:
        with st.spinner("Starting sportmonks_xg_scraper.py..."):
            command = [sys.executable, "sportmonks_xg_scraper.py"]
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/scraper_output.log", "w") as log_file:
                    subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
                st.success("✅ Scraper started in background!")
                st.info("Log in 'logs/scraper_output.log'")
            except Exception as e:
                st.error(f"Failed to start scraper: {e}")

# === NEUER KNOPF: DUTCHING SYSTEM STARTEN ===
if st.sidebar.button("⚡ Run Dutching System"):
    with st.sidebar:
        with st.spinner("Starting sportmonks_dutching_system.py..."):
            command = [sys.executable, "sportmonks_dutching_system.py"]
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/dutching_system.log", "w") as log_file:
                    subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
                st.success("✅ Dutching System started in background!")
                st.info("Log in 'logs/dutching_system.log'")
            except Exception as e:
                st.error(f"Failed to start dutching system: {e}")

# ==========================================================
# PAGE: DASHBOARD
# ==========================================================

if page == "📊 Dashboard":
    st.markdown('<div class="main-header">⚽ AI Dutching System Dashboard</div>', unsafe_allow_html=True)

    # Load data
    df_bets = load_historical_bets()

    # Calculate metrics
    total_bets = len(df_bets)
    wins = len(df_bets[df_bets['Result'] == 'Win'])
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    total_profit = df_bets['Profit'].sum()
    roi = (total_profit / df_bets['Stake'].sum() * 100) if df_bets['Stake'].sum() > 0 else 0
    avg_odds = df_bets['Odds'].mean() if total_bets > 0 else 0.0

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Bets", f"{total_bets}", "")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Total Profit", f"€{total_profit:.2f}", "")
    col4.metric("ROI", f"{roi:.1f}%", "")
    col5.metric("Avg Odds", f"{avg_odds:.2f}", "")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Cumulative Profit")
        if not df_bets.empty and 'Date' in df_bets.columns and 'Profit' in df_bets.columns:
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine Profit-Daten zum Anzeigen.")

    with col2:
        st.markdown("### 📊 Market Distribution")
        if not df_bets.empty and 'Market' in df_bets.columns:
            market_dist = df_bets['Market'].value_counts()
            fig = px.pie(values=market_dist.values, names=market_dist.index, hole=0.4)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Keine Markt-Daten zum Anzeigen.")

    st.markdown("---")
    st.markdown("### 🎯 Recent Bets")
    if not df_bets.empty:
        recent_bets = df_bets.sort_values('Date', ascending=False).head(10)
        # KORREKTUR: 'Date'-Spalte (ohne Leerzeichen)
        st.dataframe(
            recent_bets[['Date', 'Match', 'Market', 'Odds', 'Stake', 'Result', 'Profit']],
            width='stretch',
            hide_index=True
        )
    else:
        st.info("Noch keine Wetten vorhanden.")


# ==========================================================
# PAGE: GPU CONTROL (ÜBERARBEITET)
# ==========================================================

elif page == "🎮 GPU Control":
    st.markdown('<div class="main-header">🎮 GPU Control Center</div>', unsafe_allow_html=True)
    gpu_info = check_gpu_available()

    if gpu_info['available']:
        # GPU Status Card
        st.markdown(f"""
        <div class="gpu-card">
            <h2>🚀 {gpu_info['name']}</h2>
            <p><strong>CUDA:</strong> {gpu_info['cuda_version']}</p>
            <p><strong>Devices:</strong> {gpu_info['device_count']}</p>
            <p><strong>Status:</strong> <span class="status-ok">✅ Ready</span></p>
        </div>
        """, unsafe_allow_html=True)

        # --- NEU: Live-Update-Sektion ---
        st.markdown("### 📊 Current GPU Metrics")
        live_update = st.checkbox("🔄 Live-Update (alle 2 Sek.)")
        
        # Platzhalter für die Metriken
        gpu_metrics_placeholder = st.empty()

        def display_gpu_metrics():
            """Funktion zum Anzeigen der GPU-Metriken in Spalten."""
            metrics = get_gpu_metrics(_handle=GPU_HANDLE, gpu_info=gpu_info)
            
            col1, col2, col3, col4 = gpu_metrics_placeholder.columns(4)
            with col1:
                st.metric("GPU Utilization", f"{metrics['utilization']}%")
            with col2:
                mem_pct = (metrics['memory_used'] / metrics['memory_total'] * 100) if metrics['memory_total'] > 0 else 0
                st.metric("VRAM Usage", f"{metrics['memory_used']:.1f}GB / {metrics['memory_total']:.1f}GB", f"{mem_pct:.0f}%")
            with col3:
                temp_status = "🟢" if metrics['temperature'] < 80 else "🟡" if metrics['temperature'] < 85 else "🔴"
                st.metric("Temperature", f"{temp_status} {metrics['temperature']}°C")
            with col4:
                st.metric("Power Draw", f"{metrics['power_draw']:.1f}W")

        # Führe die Live-Update-Schleife aus, wenn angehakt
        if live_update:
            while True:
                display_gpu_metrics()
                time.sleep(2) # Warte 2 Sekunden
        else:
            # Zeige nur einmal an
            display_gpu_metrics()
        # --- ENDE Live-Update ---

        st.markdown("---")
        st.markdown("### 🎯 Model Training")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Neural Network")
            epochs = st.slider("Epochs", 10, 200, 100, key="nn_epochs")
            batch_size = st.selectbox("Batch Size", [128, 256, 512, 1024], index=1, key="nn_batch")
            use_fp16 = st.checkbox("Use Mixed Precision (FP16)", value=True, key="nn_fp16")

            if st.button("🚀 Train Neural Network (Test)", key="train_nn_test"):
                with st.spinner("Starting Neural Network TEST (on mock data)..."):
                    command = [sys.executable, "gpu_ml_models.py"]
                    try:
                        os.makedirs("logs", exist_ok=True)
                        with open("logs/dashboard_nn_test.log", "w") as log_file:
                            subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
                        st.success("✅ NN-Test process started in background!")
                        st.info("Monitor output in 'logs/dashboard_nn_test.log'")
                    except Exception as e:
                        st.error(f"Failed to start process: {e}")
        with col2:
            st.markdown("#### XGBoost")
            n_estimators = st.slider("Estimators", 100, 500, 300, key="xgb_estimators")
            max_depth = st.slider("Max Depth", 4, 12, 8, key="xgb_depth")
            use_gpu = st.checkbox("Use GPU", value=True, key="xgb_gpu")

            if st.button("🌲 Train XGBoost (Test)", key="train_xgb_test"):
                with st.spinner("Starting XGBoost TEST (on mock data)..."):
                    command = [sys.executable, "gpu_ml_models.py"]
                    try:
                        os.makedirs("logs", exist_ok=True)
                        with open("logs/dashboard_xgb_test.log", "w") as log_file:
                            subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
                        st.success("✅ XGB-Test process started in background!")
                        st.info("Monitor output in 'logs/dashboard_xgb_test.log'")
                    except Exception as e:
                        st.error(f"Failed to start process: {e}")

        st.markdown("---")
        st.markdown("### 🔄 Continuous Training (Real Data)")
        
        if st.button("🚀 Start REAL Training (on DB matches)", key="train_continuous_real"):
            spinner_msg = f"Starting REAL training on {st.session_state.db_matches_count} matches..."
            if st.session_state.db_matches_count == 0:
                 spinner_msg = "Starting REAL training... (DB wird geladen)"

            with st.spinner(spinner_msg):
                command = [sys.executable, "continuous_training_system.py"]
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/training_dashboard_output.log", "w") as log_file:
                        subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True)
                    st.success("✅ REAL training started in background!")
                    st.info("Monitor output in 'logs/training_dashboard_output.log'")
                except Exception as e:
                    st.error(f"Failed to start training process: {e}")

        # --- NEU: Live Log Konsole ---
        st.markdown("### 📜 Live Training Log")
        
        log_file_path = "logs/training_dashboard_output.log"
        
        # NEU:
        if st.button("🔄 Refresh Log"):
            log_content = read_log_file(log_file_path)
            
            # KORREKTUR: Verwende st.code statt st.markdown für Logs
            # Das ist sicher und behebt den InvalidCharacterError
            with st.container(height=400): # Simuliert die Höhe der alten Log-Box
                 st.code(log_content, language='bash')
        
        with st.expander("Show last 100 lines of log (on page load)"):
            log_content = read_log_file(log_file_path)
            # Verwende st.code für eine einfachere Darstellung
            st.code(log_content, language='bash', line_numbers=True)
        # --- ENDE Live Log Konsole ---

    else:
        st.error("❌ GPU Not Available")
        st.markdown("Bitte stelle sicher, dass die NVIDIA-Treiber und PyTorch mit CUDA-Support korrekt in deiner WSL-Umgebung installiert sind.")


# ==========================================================
# ANDERE SEITEN (Bleiben größtenteils gleich)
# ==========================================================

elif page == "🤖 ML Models":
    st.markdown('<div class="main-header">🤖 ML Model Management</div>', unsafe_allow_html=True)
    st.markdown("### 📋 Model Registry")
    
    models = get_model_registry()

    if models:
        df_models = pd.DataFrame(models)
        df_models['created_at'] = pd.to_datetime(df_models['created_at'])
        df_models = df_models.sort_values('created_at', ascending=False)

        champion_nn = df_models[(df_models['model_type'] == 'neural_net') & (df_models['is_champion'])].iloc[0] if not df_models[(df_models['model_type'] == 'neural_net') & (df_models['is_champion'])].empty else None
        champion_xgb = df_models[(df_models['model_type'] == 'xgboost') & (df_models['is_champion'])].iloc[0] if not df_models[(df_models['model_type'] == 'xgboost') & (df_models['is_champion'])].empty else None

        st.markdown("#### 🏆 Champion Models")
        col1, col2 = st.columns(2)
        with col1:
            if champion_nn is not None:
                st.subheader("Neural Network")
                st.metric("Val Accuracy", f"{champion_nn['validation_accuracy']*100:.2f}%")
                st.caption(f"ID: {champion_nn['version_id']}")
            else:
                st.subheader("Neural Network")
                st.warning("Kein Champion-Modell trainiert.")
        
        with col2:
            if champion_xgb is not None:
                st.subheader("XGBoost")
                st.metric("Val Accuracy", f"{champion_xgb['validation_accuracy']*100:.2f}%")
                st.caption(f"ID: {champion_xgb['version_id']}")
            else:
                st.subheader("XGBoost")
                st.warning("Kein Champion-Modell trainiert.")

        st.markdown("---")
        st.markdown("#### 📜 Version History")
        st.dataframe(df_models[['created_at', 'model_type', 'validation_accuracy', 'training_samples', 'is_champion', 'version_id']], width='stretch')
    else:
        st.info("No models in registry yet. Train your first model in the GPU Control page!")


elif page == "📊 Performance Monitor":
    st.markdown('<div class="main-header">📊 System Performance Monitor</div>', unsafe_allow_html=True)
    st.info("Die Live-GPU-Metriken findest du jetzt auf der Seite 'GPU Control'.")
    st.markdown("### 🚀 Training Performance History")
    
    # Lade echte Trainingslogs statt Mock-Daten
    log_files = list(Path('logs').glob('training_dashboard_output*.log'))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        st.info(f"Zeige Daten aus: {latest_log.name}")
        
        lines = read_log_file(str(latest_log)).splitlines()
        
        history = []
        for line in lines:
            if line.startswith("Epoch"):
                try:
                    parts = line.split('|')
                    epoch = int(parts[0].split('/')[0].replace('Epoch', '').strip())
                    val_loss = float(parts[2].split(':')[1].strip())
                    val_acc = float(parts[3].split(':')[1].strip())
                    history.append({'epoch': epoch, 'val_loss': val_loss, 'val_accuracy': val_acc})
                except:
                    continue # Zeile ignorieren
        
        if history:
            df_history = pd.DataFrame(history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_history['epoch'], y=df_history['val_loss'], name='Validation Loss', yaxis='y1'))
            fig.add_trace(go.Scatter(x=df_history['epoch'], y=df_history['val_accuracy'], name='Validation Accuracy', yaxis='y2'))
            fig.update_layout(
                height=400,
                title="Letzter Trainingslauf (Neural Network)",
                xaxis_title="Epoch",
                yaxis=dict(title='Loss'),
                yaxis2=dict(title='Accuracy', overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Konnte keine Trainings-Statistiken aus dem Log parsen.")
    else:
        st.warning("Keine Trainings-Logs im '/logs'-Ordner gefunden.")


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
    st.markdown("### 🤖 Ensemble Model Weights")
    st.info("Steuert, wie die Vorhersagen von Poisson, NN und XGBoost gewichtet werden.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider("Poisson Weight", 0.0, 1.0, 0.34, 0.01, key="weight_poisson")
    with col2:
        st.slider("XGBoost Weight", 0.0, 1.0, 0.33, 0.01, key="weight_xgb")
    with col3:
        st.slider("Neural Net Weight", 0.0, 1.0, 0.33, 0.01, key="weight_nn")

    if st.button("💾 Save All Settings"):
        st.success("✅ Settings saved successfully! (Funktion noch nicht implementiert)")


elif page == "💰 Live Bets":
    st.markdown('<div class="main-header">💰 Live Betting Opportunities</div>', unsafe_allow_html=True)
    st.info("Klicke 'Run Dutching System' in der Sidebar, um Wetten zu finden. Ergebnisse erscheinen hier, sobald der Lauf beendet ist.")
    
    dutching_log = "logs/dutching_system.log"
    if st.button("🔄 Lade Wett-Ergebnisse"):
        st.rerun()

    if os.path.exists(dutching_log):
        with st.expander("Zeige Live-Log des Dutching Systems"):
             log_content = read_log_file(dutching_log)
             st.code(log_content, language='bash')
    
    # Lade die *Ergebnis*-CSV (nicht die Log-Datei)
    bet_files = list(Path('.').glob('sportmonks_results_*.csv'))
    if bet_files:
        latest_file = max(bet_files, key=lambda x: x.stat().st_mtime)
        st.success(f"Ergebnisse vom letzten Lauf ({latest_file.name}):")
        df_results = pd.read_csv(latest_file)
        st.dataframe(df_results, width=None)
    else:
        st.warning("Noch keine Ergebnis-Datei (sportmonks_results_...) gefunden.")


elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    df_bets = load_historical_bets()

    if not df_bets.empty and 'Market' in df_bets.columns:
        st.markdown("### 📊 Win Rate by Market")
        market_stats = df_bets.groupby('Market').agg({
            'Result': lambda x: (x == 'Win').sum() / len(x) * 100
        }).reset_index()
        market_stats.columns = ['Market', 'Win Rate (%)']
        fig = px.bar(market_stats, x='Market', y='Win Rate (%)', color='Win Rate (%)',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Keine Daten für Analytics vorhanden.")
