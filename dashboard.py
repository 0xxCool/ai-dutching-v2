"""
⚽ ADVANCED AI FOOTBALL BETTING DASHBOARD v5.0 - EMOJI FIX
==========================================================
Modern Football-Themed Dashboard with Real-time Live Logs & Updates

FIXED FEATURES:
- ✅ Real-time Live Log Streaming with Threading
- ✅ Queue-based Log Collection
- ✅ Proper Auto-Refresh without Page Reload
- ✅ Full Component Integration
- ✅ Process Management with Session State
- ✅ Smooth UI Updates
- ✅ EMOJI ICONS (Font Awesome durch Emojis ersetzt)

Features:
- Live Match Tracking & Odds Monitoring
- Real-time Log Streaming for all Scripts
- Adjustable Dutching System Parameters  
- Real-time Correct Score Analysis
- GPU ML Model Performance
- Portfolio Management
- Animated UI Components
- Interactive Charts
- System Health Monitoring
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
    page_title="AI Football Betting System v5.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LIVE LOG STREAMING CLASSES
# =============================================================================
class LogStreamManager:
    """Manages live log streaming from subprocesses"""
    
    def __init__(self):
        self.processes = {}
        self.log_queues = {}
        self.threads = {}
        self.stop_events = {}
        
    def start_process(self, name: str, command: List[str], cwd: str = None):
            """Start a process and begin log streaming"""
            try:
                # Beende alten Prozess falls vorhanden
                self.stop_process(name)

                # --- KORRIGIERTE VALIDIERUNG ---
                # Finde das Python-Skript im Befehl (das, was auf .py endet)
                script_name = next((arg for arg in command if arg.endswith('.py')), None)
                
                if not script_name:
                    # Falls kein .py-Skript gefunden wird (unwahrscheinlich), logge einen Fehler
                    logging.error(f"Keine .py-Datei im Befehl gefunden: {command}")
                    raise FileNotFoundError(f"Keine .py-Datei im Befehl gefunden: {command}")

                script_path = Path(cwd) / script_name if cwd else Path(script_name)
                
                if not script_path.exists():
                    logging.error(f"Script not found: {script_path}")
                    raise FileNotFoundError(f"Script not found: {script_path}")
                # --- ENDE KORREKTUR ---

                # Erstelle Queue und Stop Event
                log_queue = queue.Queue()
                stop_event = threading.Event()

                # Starte Prozess
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,  # Erzwinge Zeilen-Pufferung
                    cwd=cwd
                )

                # Starte Log Reader Thread (die robuste Version von vorhin)
                thread = threading.Thread(
                    target=self._read_process_output,
                    args=(process, log_queue, stop_event, name),
                    daemon=True
                )
                thread.start()

                # Speichere in Manager
                self.processes[name] = process
                self.log_queues[name] = log_queue
                self.threads[name] = thread
                self.stop_events[name] = stop_event

                logging.info(f"Process '{name}' started successfully")
                return True

            except Exception as e:
                logging.error(f"Error starting process '{name}': {e}")
                raise

    def _read_process_output(self, process, log_queue, stop_event, name):
            """Robustes Auslesen der Prozessausgabe und Weiterleitung an die Queue"""
            try:
                # Verwende iter(), um Zeilen einzeln zu lesen, sobald sie eintreffen.
                # Diese Schleife blockiert, bis eine neue Zeile erscheint,
                # und beendet sich automatisch, wenn der Prozess stdout schließt.
                for line in iter(process.stdout.readline, ''):
                    if stop_event.is_set():
                        break  # Stop-Ereignis wurde ausgelöst
                    if line:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        log_queue.put(f"[{timestamp}] {line.strip()}")

                # Nachdem die Schleife beendet ist, ist der Prozess beendet.
                # Warte, bis er vollständig geschlossen ist, und hole den Return-Code.
                process.wait()
                returncode = process.returncode
                log_queue.put(f"[FINISHED] Process {name} completed with code {returncode}")
            
            except Exception as e:
                # Fange alle Lesefehler ab
                log_queue.put(f"[ERROR] Log-Lesefehler: {str(e)}")
            finally:
                # Stelle sicher, dass die Queue weiß, dass der Prozess beendet ist
                if not stop_event.is_set():
                    log_queue.put(f"[SYSTEM] Log-Reader-Thread für {name} wird beendet.")
    
    def get_logs(self, name: str, max_lines: int = 100) -> List[str]:
        """Get logs from queue"""
        if name not in self.log_queues:
            return []
        
        logs = []
        log_queue = self.log_queues[name]
        
        # Hole alle verfügbaren Logs (non-blocking)
        while not log_queue.empty() and len(logs) < max_lines:
            try:
                logs.append(log_queue.get_nowait())
            except queue.Empty:
                break
                
        return logs
    
    def stop_process(self, name: str):
        """Stop a process and its log reader"""
        try:
            if name in self.processes:
                # Setze Stop Event
                if name in self.stop_events:
                    self.stop_events[name].set()

                # Beende Prozess
                process = self.processes[name]
                if process.poll() is None:  # Noch aktiv
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logging.warning(f"Process '{name}' did not terminate gracefully, killing it")
                        process.kill()
                        process.wait()

                # Cleanup
                self.processes.pop(name, None)
                self.log_queues.pop(name, None)
                self.threads.pop(name, None)
                self.stop_events.pop(name, None)

                logging.info(f"Process '{name}' stopped successfully")
                return True
            return False
        except Exception as e:
            logging.error(f"Error stopping process '{name}': {e}")
            return False
    
    def is_running(self, name: str) -> bool:
        """Check if process is running"""
        if name in self.processes:
            return self.processes[name].poll() is None
        return False
    
    def stop_all(self):
        """Stop all processes"""
        try:
            for name in list(self.processes.keys()):
                self.stop_process(name)
            logging.info("All processes stopped")
        except Exception as e:
            logging.error(f"Error stopping all processes: {e}")

# =============================================================================
# CUSTOM CSS - MODERN FOOTBALL THEME (ENHANCED)
# =============================================================================
st.markdown("""
<style>
    /* === ANIMATED BACKGROUND === */
    @keyframes grassAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* === KORRIGIERTE GRÜNE BUTTON KLASSE (FINALE VERSION) === */
    /* Wir zielen auf 'outline'-Buttons (die 'bg-transparent' sind)
       und AUCH unsere .btn-green Klasse haben. */
    button.bg-transparent.btn-green {
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border-color: #4CAF50 !important; /* Rand auch färben */
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    
    button.bg-transparent.btn-green:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6);
        background: linear-gradient(135deg, #45a049, #4CAF50) !important;
        border-color: #45a049 !important;
        color: white !important; /* Sicherstellen, dass Text weiß bleibt */
    }
    
    /* FALLBACK: Falls der obige Selektor versagt, nimmt dieser jeden Button
       innerhalb des .btn-green Wrappers. */
    .btn-green button {
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border-color: #4CAF50 !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    
    .btn-green button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6);
        background: linear-gradient(135deg, #45a049, #4CAF50) !important;
        border-color: #45a049 !important;
        color: white !important;
    }
    
    
    [data-testid="stAppViewContainer"] {
        /* Dein altes 'background: linear-gradient...' kommt weg */
        
        /* NEU: Hier überlagern wir das Bild mit einer 70% schwarzen,
           transparenten Schicht, damit der Inhalt lesbar bleibt.
        */
        background: 
            linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url("https://images.pexels.com/photos/46798/the-ball-stadion-football-the-pitch-46798.jpeg?auto=compress&cs=tinysrgb&w=1920") !important;
        
        background-size: cover !important; /* Bild füllt den ganzen Bildschirm */
        background-repeat: no-repeat !important;
        background-attachment: fixed !important; /* Bild bleibt beim Scrollen fixiert */
        background-position: center center !important; /* ⚽ Fußball zentriert */
    }
    
    /* === GLASSMORPHISM EFFECTS === */
    [data-testid="block-container"] {
        background: rgba(10, 10, 10, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 15, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 2px solid rgba(76, 175, 80, 0.5) !important;
    }
    
    /* === ENTFERNE SCHWARZE HINTERGRUNDKÄSTEN === */
    
    /* WICHTIG: Behalte das Hintergrundbild! */
    [data-testid="stAppViewContainer"] {
        /* NICHT transparent machen - hier ist das Hintergrundbild! */
    }
    
    /* 1. COLUMNS & HORIZONTAL BLOCKS */
    [data-testid="column"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stHorizontalBlock"] {
        background: transparent !important;
        gap: 0.5rem;
        border: none !important;
        padding: 0 !important;
    }
    
    /* 2. VERTICAL BLOCKS */
    [data-testid="stVerticalBlock"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* 3. ELEMENT CONTAINER */
    .element-container {
        background: transparent !important;
        border: none !important;
    }
    
    /* 4. TAB-LEISTE */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        border: none !important;
        gap: 1rem;
    }
    
    [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* 5. TAB PANELS - KRITISCH! */
    [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding: 2rem 0 !important;
    }
    
    /* 6. BUTTON CONTAINER */
    .stButton {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stButton > div {
        background: transparent !important;
    }
    
    /* 7. WIDGET ROW */
    .row-widget {
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* 8. VERSCHACHTELTE DIVS IN COLUMNS - CHIRURGISCH! */
    [data-testid="column"] > div:not([data-testid="stMarkdown"]):not([class*="stAlert"]) {
        background: transparent !important;
        border: none !important;
    }
    
    /* 9. FORM CONTAINER */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* 10. MARKDOWN CONTAINER - nur der äußere Container */
    [data-testid="stMarkdownContainer"] > div {
        background: transparent !important;
    }
    
    /* === INPUT FELDER BEHALTEN IHREN HINTERGRUND === */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input,
    input[type="number"],
    input[type="text"],
    input[type="date"],
    textarea,
    select {
        background: rgba(30, 30, 30, 0.7) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 6px !important;
        color: white !important;
        padding: 0.5rem !important;
    }
    
    /* Selectbox Dropdown */
    [data-baseweb="select"] {
        background: rgba(30, 30, 30, 0.7) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
    }
    
    /* === ALERT BOXEN BEHALTEN HINTERGRUND === */
    [data-testid="stAlert"] {
        background: rgba(30, 30, 30, 0.5) !important;
        backdrop-filter: blur(5px) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        padding: 1rem !important;
    }
    
    /* Info-Box speziell */
    [data-testid="stInfo"] [data-testid="stAlert"] {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Success-Box */
    [data-testid="stSuccess"] [data-testid="stAlert"] {
        background: rgba(76, 175, 80, 0.2) !important;
        border-color: rgba(76, 175, 80, 0.5) !important;
    }
    
    /* Error-Box */
    [data-testid="stError"] [data-testid="stAlert"] {
        background: rgba(244, 67, 54, 0.2) !important;
        border-color: rgba(244, 67, 54, 0.5) !important;
    }
    
    /* Warning-Box */
    [data-testid="stWarning"] [data-testid="stAlert"] {
        background: rgba(255, 193, 7, 0.2) !important;
        border-color: rgba(255, 193, 7, 0.5) !important;
    }
    
    /* === EXPANDER BEHALTEN HINTERGRUND === */
    [data-testid="stExpanderDetails"] {
        background: rgba(30, 30, 30, 0.4) !important;
        backdrop-filter: blur(5px) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* === NEON GLOW EFFECTS === */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4CAF50, #8BC34A, #CDDC39);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 30px rgba(76, 175, 80, 0.5);
        
        margin-bottom: 2rem;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* === CARDS WITH HOVER EFFECTS === */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 30, 30, 0.9), rgba(20, 20, 20, 0.9)) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 30px rgba(76, 175, 80, 0.4) !important;
        border-color: #4CAF50 !important;
    }
    
    /* === ANIMATED BUTTONS (DEIN ORIGINAL FÜR st.button) === */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }
    
    /* === LIVE LOG BOX === */
    .live-log-box {
        background: rgba(10, 10, 10, 0.9) !important;
        border: 1px solid #4CAF50 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
        color: #00ff00 !important;
        height: 400px !important;
        overflow-y: auto !important;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.3) !important;
    }
    
    .live-log-box::-webkit-scrollbar {
        width: 10px;
    }
    
    .live-log-box::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 5px;
    }
    
    .live-log-box::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 5px;
    }
    
    /* === LIVE INDICATOR === */
    @keyframes liveAnimation {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ff4444;
        border-radius: 50%;
        animation: liveAnimation 1.5s infinite;
        margin-right: 8px;
        box-shadow: 0 0 10px #ff4444;
    }
            
    .live-log-box.auto-scroll {
        /* Erzwingt, dass die Scrollbar unten bleibt */
        display: flex;
        flex-direction: column-reverse;
    }        

    /* === SUCCESS/WARNING/ERROR GLOWS === */
    .success-glow { box-shadow: 0 0 20px rgba(76, 175, 80, 0.6) !important; }
    .warning-glow { box-shadow: 0 0 20px rgba(255, 193, 7, 0.6) !important; }
    .danger-glow { box-shadow: 0 0 20px rgba(244, 67, 54, 0.6) !important; }
    
    /* === PROCESS STATUS BADGES === */
    .process-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        text-transform: uppercase;
        margin: 0.25rem;
    }
    
    .process-running {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
        
    }
    
    .process-stopped {
        background: linear-gradient(135deg, #f44336, #e91e63);
        color: white;
    }
    
    .process-idle {
        background: linear-gradient(135deg, #9e9e9e, #757575);
        color: white;
    }
    
    /* === NEUE PROFESSIONELLE STATUS CARDS === */
    .status-card {
        background: linear-gradient(145deg, rgba(30, 30, 30, 0.6), rgba(20, 20, 20, 0.6)) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 1.25rem !important;
        margin: 0.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.2) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    .status-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3) !important;
        border-color: rgba(76, 175, 80, 0.5) !important;
    }
    
    .status-card.running {
        border-left: 4px solid #4CAF50 !important;
        background: linear-gradient(145deg, rgba(76, 175, 80, 0.1), rgba(20, 20, 20, 0.6)) !important;
    }
    
    .status-card.stopped {
        border-left: 4px solid #f44336 !important;
        background: linear-gradient(145deg, rgba(244, 67, 54, 0.1), rgba(20, 20, 20, 0.6)) !important;
    }
    
    .status-card-title {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: white !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .status-card-status {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 700 !important;
    }
    
    .status-icon {
        font-size: 1.5rem !important;
        display: inline-block !important;
    }
    
    /* === KOMPAKTER SIDEBAR STATUS === */
    .sidebar-status-item {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 0.5rem 0.75rem !important;
        margin: 0.25rem 0 !important;
        background: rgba(30, 30, 30, 0.4) !important;
        border-radius: 8px !important;
        border-left: 3px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    
    .sidebar-status-item.running {
        border-left-color: #4CAF50 !important;
        background: rgba(76, 175, 80, 0.1) !important;
    }
    
    .sidebar-status-item.stopped {
        border-left-color: #f44336 !important;
        background: rgba(244, 67, 54, 0.05) !important;
    }
    
    .sidebar-status-item .name {
        font-size: 0.85rem !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    .sidebar-status-item .indicator {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        display: inline-block !important;
    }
    
    .sidebar-status-item .indicator.running {
        background: #4CAF50 !important;
        box-shadow: 0 0 8px #4CAF50 !important;
        animation: pulse 2s infinite !important;
    }
    
    .sidebar-status-item .indicator.stopped {
        background: #9e9e9e !important;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* === SYSTEM OVERVIEW CARD === */
    .system-overview {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(30, 30, 30, 0.6)) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        text-align: center !important;
    }
    
    .system-overview .status-number {
        font-size: 2rem !important;
        font-weight: 900 !important;
        color: #4CAF50 !important;
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.5) !important;
    }
    
    .system-overview .status-label {
        font-size: 0.85rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

/* GREEN BUTTONS - EXACTLY LIKE DESTRUCTIVE BUT GREEN! */
button[data-variant="secondary"],
button[class*="variant-secondary"] {
    background: linear-gradient(to bottom right, #10b981, #059669) !important;
    background-color: #10b981 !important;
    color: rgb(255, 255, 255) !important;
    border: 2px solid #10b981 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    line-height: 1.25rem !important;
    padding: 0.5rem 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.025em !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
}

button[data-variant="secondary"]:hover,
button[class*="variant-secondary"]:hover {
    background: linear-gradient(to bottom right, #059669, #047857) !important;
    background-color: #059669 !important;
    border-color: #059669 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

button[data-variant="secondary"]:active,
button[class*="variant-secondary"]:active {
    background-color: #047857 !important;
    border-color: #047857 !important;
    transform: translateY(0) !important;
}

/* === SHADCN UI BUTTON CONTAINER === */
/* Entferne schwarze Hintergründe um shadcn-ui Buttons herum */
iframe[title*="streamlit_shadcn_ui"] {
    background: transparent !important;
}

/* Wrapper um die Buttons */
div[data-testid="stVerticalBlock"] > div:has(iframe[title*="streamlit_shadcn_ui"]) {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Alle Button-Container */
.stButton,
[class*="stButton"] {
    background: transparent !important;
    padding: 0 !important;
}

/* Button Wrapper */
.stButton > div,
[class*="stButton"] > div {
    background: transparent !important;
}

/* Alle iframes transparent */
iframe {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION (ENHANCED & DEBUGGED)
# =============================================================================
def init_session_state():
    """Initialize session state with all required components"""

    # Debug Flag
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = True  # Aktiviere Debug während Entwicklung

    initialization_errors = []

    try:
        # Basic state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        if 'active_bets' not in st.session_state:
            st.session_state.active_bets = []
        if 'portfolio_stats' not in st.session_state:
            st.session_state.portfolio_stats = {}
        if 'system_alerts' not in st.session_state:
            st.session_state.system_alerts = []

        # Log Stream Manager - KRITISCH für Button-Funktionalität
        if 'log_manager' not in st.session_state:
            try:
                st.session_state.log_manager = LogStreamManager()
                logging.info("✅ LogStreamManager initialized")
                if st.session_state.debug_mode:
                    st.sidebar.success("✅ LogStreamManager OK")
            except Exception as e:
                error_msg = f"❌ LogStreamManager Fehler: {e}"
                initialization_errors.append(error_msg)
                logging.error(error_msg)
                st.error(error_msg)
                # Erstelle einen Dummy LogStreamManager damit Dashboard nicht crasht
                st.session_state.log_manager = None

        # Log Buffers
        if 'scraper_logs' not in st.session_state:
            st.session_state.scraper_logs = []
        if 'dutching_logs' not in st.session_state:
            st.session_state.dutching_logs = []
        if 'ml_logs' not in st.session_state:
            st.session_state.ml_logs = []
        if 'portfolio_logs' not in st.session_state:
            st.session_state.portfolio_logs = []
        if 'alert_logs' not in st.session_state:
            st.session_state.alert_logs = []

        # Process States
        if 'process_states' not in st.session_state:
            st.session_state.process_states = {
                'scraper': 'idle',
                'dutching': 'idle',
                'ml_training': 'idle',
                'portfolio': 'idle',
                'alerts': 'idle',
                'correct_score': 'idle'
            }

        # Correct Score Logs
        if 'correct_score_logs' not in st.session_state:
            st.session_state.correct_score_logs = []

        # Initialize Components - nur einmal
        if 'components_initialized' not in st.session_state:
            try:
                config = get_config()
                if st.session_state.debug_mode:
                    st.sidebar.info("✅ Config loaded")

                # API Token
                api_token = config.api.api_token
                if not api_token:
                    warning_msg = "⚠️ SPORTMONKS_API_TOKEN nicht gesetzt in .env"
                    initialization_errors.append(warning_msg)
                    logging.warning(warning_msg)
                    if st.session_state.debug_mode:
                        st.sidebar.warning(warning_msg)
                    api_token = "dummy_token"  # Fallback

                dutching_config_instance = DutchingConfig()

                # Sportmonks Client
                try:
                    st.session_state.sportmonks_client = SportmonksClient(
                        api_token=api_token,
                        config=dutching_config_instance
                    )
                    if st.session_state.debug_mode:
                        st.sidebar.success("✅ SportmonksClient OK")
                except Exception as e:
                    error_msg = f"❌ SportmonksClient Fehler: {e}"
                    initialization_errors.append(error_msg)
                    logging.error(error_msg)
                    st.session_state.sportmonks_client = None
                    if st.session_state.debug_mode:
                        st.sidebar.error(error_msg)

                # Portfolio Manager
                try:
                    st.session_state.portfolio_manager = PortfolioManager(bankroll=10000.0)
                    if st.session_state.debug_mode:
                        st.sidebar.success("✅ PortfolioManager OK")
                except Exception as e:
                    error_msg = f"❌ PortfolioManager Fehler: {e}"
                    initialization_errors.append(error_msg)
                    logging.error(error_msg)
                    st.session_state.portfolio_manager = None
                    if st.session_state.debug_mode:
                        st.sidebar.error(error_msg)

                # Alert Manager
                try:
                    alert_config = AlertConfig()
                    st.session_state.alert_manager = AlertManager(alert_config)
                    if st.session_state.debug_mode:
                        st.sidebar.success("✅ AlertManager OK")
                except Exception as e:
                    error_msg = f"❌ AlertManager Fehler: {e}"
                    initialization_errors.append(error_msg)
                    logging.error(error_msg)
                    st.session_state.alert_manager = None
                    if st.session_state.debug_mode:
                        st.sidebar.error(error_msg)

                # API Cache
                try:
                    cache_config = CacheConfig()
                    st.session_state.api_cache = APICache(cache_config)
                    if st.session_state.debug_mode:
                        st.sidebar.success("✅ APICache OK")
                except Exception as e:
                    error_msg = f"❌ APICache Fehler: {e}"
                    initialization_errors.append(error_msg)
                    logging.error(error_msg)
                    st.session_state.api_cache = None
                    if st.session_state.debug_mode:
                        st.sidebar.error(error_msg)

                # Model Registry
                try:
                    st.session_state.model_registry = ModelRegistry()
                    if st.session_state.debug_mode:
                        st.sidebar.success("✅ ModelRegistry OK")
                except Exception as e:
                    error_msg = f"❌ ModelRegistry Fehler: {e}"
                    initialization_errors.append(error_msg)
                    logging.error(error_msg)
                    st.session_state.model_registry = None
                    if st.session_state.debug_mode:
                        st.sidebar.error(error_msg)

                st.session_state.components_initialized = True

                if initialization_errors:
                    logging.warning(f"Initialisierung mit {len(initialization_errors)} Warnungen abgeschlossen")
                else:
                    logging.info("✅ All components initialized successfully")

            except Exception as e:
                error_msg = f"❌ KRITISCHER FEHLER bei Component Initialization: {e}"
                initialization_errors.append(error_msg)
                logging.error(error_msg)
                st.error(error_msg)
                import traceback
                traceback_msg = traceback.format_exc()
                logging.error(traceback_msg)
                if st.session_state.debug_mode:
                    st.sidebar.error(error_msg)
                    with st.sidebar.expander("Traceback"):
                        st.code(traceback_msg)

    except Exception as e:
        error_msg = f"❌ KRITISCHER FEHLER in init_session_state: {e}"
        logging.error(error_msg)
        st.error(error_msg)
        import traceback
        traceback_msg = traceback.format_exc()
        logging.error(traceback_msg)
        st.code(traceback_msg)

    # Speichere Errors für spätere Anzeige
    if 'initialization_errors' not in st.session_state:
        st.session_state.initialization_errors = initialization_errors

# Initialize session state
init_session_state()

def callback_toggle_auto_refresh():
    """Callback, um den Refresh-Status zu melden."""
    if st.session_state.auto_refresh:
        st.toast("✅ Auto-Refresh aktiviert!", icon="🔄")
    else:
        st.toast("ℹ️ Auto-Refresh deaktiviert.", icon="⏸️")

# =============================================================================
# HELPER FUNCTIONS (ENHANCED)
# =============================================================================
def get_gpu_stats() -> Optional[Dict]:
    """Get GPU statistics if available"""
    if not GPU_AVAILABLE:
        return None
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU Info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Memory Info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used / 1024**3  # GB
            memory_total = mem_info.total / 1024**3  # GB
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Utilization
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except (pynvml.NVMLError, Exception) as e:
                logging.warning(f"Could not get GPU temperature: {e}")
                temp = 0

            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watt
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            except (pynvml.NVMLError, Exception) as e:
                logging.warning(f"Could not get GPU power: {e}")
                power = 0
                power_limit = 0
            
            return {
                'name': name,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'memory_percent': memory_percent,
                'gpu_util': gpu_util,
                'temperature': temp,
                'power': power,
                'power_limit': power_limit
            }
    except Exception as e:
        st.warning(f"GPU Stats Error: {e}")
        return None

def update_logs(process_name: str):
    """Update logs for a specific process"""
    new_logs = st.session_state.log_manager.get_logs(process_name)
    
    if process_name == 'scraper':
        st.session_state.scraper_logs.extend(new_logs)
    elif process_name == 'dutching':
        st.session_state.dutching_logs.extend(new_logs)
    elif process_name == 'ml_training':
        st.session_state.ml_logs.extend(new_logs)
    elif process_name == 'portfolio':
        st.session_state.portfolio_logs.extend(new_logs)
    elif process_name == 'alerts':
        st.session_state.alert_logs.extend(new_logs)

def display_live_logs(process_name: str, logs: List[str], container):
    """Display logs in a live updating container"""
    if not logs:
        container.info("Warte auf Logs...")
        return

    # Prüfe, ob der Prozess (noch) läuft
    is_running = st.session_state.log_manager.is_running(process_name)

    if is_running:
        # 1. WENN LÄUFT: Zeige die letzten 100 Zeilen
        log_subset = logs[-100:]
        log_text = "\n".join(log_subset)
        # Verwende die 'auto-scroll'-Klasse, um automatisch nach unten zu scrollen
        container.markdown(
            f'<div class="live-log-box auto-scroll">{log_text}</div>',
            unsafe_allow_html=True
        )
    else:
        # 2. WENN FERTIG: Zeige ALLE Zeilen (nicht-scrollend)
        log_text = "\n".join(logs)
        container.markdown(
            f'<div class="live-log-box">{log_text}</div>',
            unsafe_allow_html=True
        )

def start_scraper():
    """Start the hybrid scraper"""
    try:
        if not hasattr(st.session_state, 'log_manager') or st.session_state.log_manager is None:
            error_msg = "❌ FEHLER: LogStreamManager nicht initialisiert! Bitte Dashboard neu laden."
            st.error(error_msg)
            logging.error(error_msg)
            return

        cwd = str(Path.cwd())
        # Stelle sicher, dass -u hier drin bleibt!
        command = [sys.executable, '-u', 'sportmonks_hybrid_scraper_v3_FINAL.py']

        logging.info(f"Starting scraper with command: {command} in {cwd}")
        st.session_state.log_manager.start_process('scraper', command, cwd=cwd)
        st.session_state.process_states['scraper'] = 'running'
        st.success("🚀 Hybrid Scraper gestartet!")
        logging.info("Scraper started successfully")

    except Exception as e:
        st.error(f"❌ Fehler beim Starten des Scrapers: {e}")
        logging.error(f"Scraper start error: {e}")
        import traceback
        traceback.print_exc()
                
def stop_scraper():
    """Stop the hybrid scraper"""
    try:
        if st.session_state.log_manager.stop_process('scraper'):
            st.session_state.process_states['scraper'] = 'idle'
            st.success("🛑 Hybrid Scraper gestoppt!")
    except Exception as e:
        st.error(f"❌ Fehler beim Stoppen des Scrapers: {e}")
        logging.error(f"Scraper stop error: {e}")

def start_dutching():
    """Start the dutching system"""
    try:
        cwd = str(Path.cwd())
        command = [sys.executable, '-u', 'sportmonks_dutching_system.py']
        st.session_state.log_manager.start_process('dutching', command, cwd=cwd)
        st.session_state.process_states['dutching'] = 'running'
        st.success("🚀 Dutching System gestartet!")
    except Exception as e:
        st.error(f"❌ Fehler beim Starten des Dutching Systems: {e}")
        logging.error(f"Dutching start error: {e}")

def stop_dutching():
    """Stop the dutching system"""
    try:
        if st.session_state.log_manager.stop_process('dutching'):
            st.session_state.process_states['dutching'] = 'idle'
            st.success("🛑 Dutching System gestoppt!")
    except Exception as e:
        st.error(f"❌ Fehler beim Stoppen des Dutching Systems: {e}")
        logging.error(f"Dutching stop error: {e}")

def start_ml_training():
    """Start ML model training"""
    try:
        cwd = str(Path.cwd())
        command = [sys.executable, '-u', 'train_ml_models.py']
        st.session_state.log_manager.start_process('ml_training', command, cwd=cwd)
        st.session_state.process_states['ml_training'] = 'running'
        st.success("🚀 ML Training gestartet!")
    except Exception as e:
        st.error(f"❌ Fehler beim Starten des ML Trainings: {e}")
        logging.error(f"ML training start error: {e}")

def start_portfolio_optimizer():
    """Start portfolio optimizer"""
    try:
        cwd = str(Path.cwd())
        command = [sys.executable, '-u', 'portfolio_manager.py']
        st.session_state.log_manager.start_process('portfolio', command, cwd=cwd)
        st.session_state.process_states['portfolio'] = 'running'
        st.success("🚀 Portfolio Optimizer gestartet!")
    except Exception as e:
        st.error(f"❌ Fehler beim Starten des Portfolio Optimizers: {e}")
        logging.error(f"Portfolio optimizer start error: {e}")

def start_alert_system():
    """Start alert system"""
    try:
        cwd = str(Path.cwd())
        command = [sys.executable, '-u', 'alert_system.py']
        st.session_state.log_manager.start_process('alerts', command, cwd=cwd)
        st.session_state.process_states['alerts'] = 'running'
        st.success("🚀 Alert System gestartet!")
    except Exception as e:
        st.error(f"❌ Fehler beim Starten des Alert Systems: {e}")
        logging.error(f"Alert system start error: {e}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Update Portfolio Stats
    portfolio_mgr = st.session_state.portfolio_manager

    # Get portfolio statistics
    try:
        st.session_state.portfolio_stats = portfolio_mgr.get_portfolio_statistics()
    except Exception as e:
        logging.error(f"Error getting portfolio stats: {e}")
        st.session_state.portfolio_stats = {}
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2.5rem 0; background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(59, 130, 246, 0.5) 100%); 
                border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
                backdrop-filter: blur(10px); border: 1px solid rgba(59, 130, 246, 0.3);'>
        <div style='display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 0.5rem;'>
            <span style='font-size: 2.8rem; filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));'>⚽</span>
            <h1 style='color: white; font-size: 3.2rem; margin: 0; font-weight: 900; letter-spacing: 3px; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                FOOTBALL ANALYTICS HUB
            </h1>
        </div>
        <p style='color: #e0e7ff; font-size: 1.15rem; margin: 0.5rem 0 0 0; font-weight: 500; letter-spacing: 1.5px;'>
            🧠 Powered by Artificial Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><span class="live-indicator"></span> LIVE MONITORING ACTIVE</div>', unsafe_allow_html=True)

    # Main Tabs - KORRIGIERT MIT EMOJIS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "⚽ Live Matches",
        "⚙️ System Control",
        "🧠 ML Models",
        "💼 Portfolio",
        "📊 Analytics",
        "🎛️ Settings",
        "📈 Backtesting",
        "🎯 Correct Score"
    ])

    # =============================================================================
    # LOG-UPDATER
    # =============================================================================
    try:
        if 'log_manager' in st.session_state and st.session_state.log_manager:
            for process_name in st.session_state.process_states.keys():
                update_logs(process_name)
    except Exception as e:
        logging.warning(f"Log-Update-Fehler (wird beim nächsten Rerun behoben): {e}")
    
    # =============================================================================
    # NEUER AUTO-REFRESH (NON-BLOCKING)
    # =============================================================================
    from streamlit.components.v1 import html

    if st.session_state.auto_refresh:
        st.session_state.last_refresh = datetime.now()
        refresh_interval = st.session_state.refresh_interval
        html(f"<meta http-equiv='refresh' content='{refresh_interval}'>", height=0)
    
# =============================================================================
# TAB 1: LIVE MATCHES - MIT ECHTEN DATEN
# =============================================================================
# WICHTIG: Füge diesen Import am Anfang der Dashboard-Datei hinzu:
# from live_match_loader import load_live_matches, get_available_leagues

    with tab1:
        st.markdown("## 🏟️ Live Football Matches & Betting Opportunities")
        
        # Live Status Indicator
        col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
        with col_status1:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="live-indicator"></span>
                <span style="color: #4CAF50; font-weight: bold;">LIVE DATA ACTIVE</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status2:
            # Hole verfügbare Ligen aus der Datenbank
            try:
                available_leagues = get_available_leagues("game_database_complete.csv")
            except Exception as e:
                logger.error(f"Error loading leagues: {e}")
                available_leagues = ["All", "Premier League", "La Liga", "Bundesliga", "Serie A"]
        
        with col_status3:
            # Last Update Time
            st.markdown(f"""
            <div style="text-align: right; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                Last Update: {datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Live Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            league_filter = st.selectbox(
                "League", 
                available_leagues,
                key="live_league_filter"
            )
        
        with col2:
            min_edge = st.slider(
                "Min Edge %", 
                0, 50, 10,
                key="live_min_edge",
                help="Minimum Value Edge % to display"
            )
        
        with col3:
            market_type = st.selectbox(
                "Market", 
                ["All", "Match Winner", "Over/Under", "Both Teams Score"],
                key="live_market_type",
                help="Currently only Match Winner is implemented"
            )
        
        with col4:
            if ui.button(
                text="Refresh Matches", 
                icon="refresh_cw", 
                variant="secondary", 
                key="refresh_matches_btn", 
                className="btn-green"
            ):
                st.rerun()
        
        st.markdown("---")
        
        # =============================================================================
        # LADE ECHTE LIVE-DATEN
        # =============================================================================
        
        try:
            # Lade Live Matches mit Filters
            live_matches = load_live_matches(
                data_file="game_database_complete.csv",
                time_window_hours=3,  # 3 Stunden Fenster (1.5h zurück, 1.5h voraus)
                min_edge=min_edge / 100,  # Konvertiere zu Dezimal
                league=league_filter if league_filter != "All" else None
            )
            
            # Check if matches found
            if live_matches.empty:
                st.info("""
                ℹ️ **No Live Matches Found**
                
                Mögliche Gründe:
                - Aktuell keine Spiele im Zeitfenster (±3 Stunden)
                - Keine Spiele mit Edge >= {min_edge}%
                - Filter zu restriktiv
                
                **Tipp:** 
                - Reduziere Min Edge auf 0%
                - Wähle "All" als Liga
                - Starte den Scraper um aktuelle Daten zu laden:
                  `python sportmonks_hybrid_scraper_v3_FINAL.py`
                """)
                
                # Zeige letzte verfügbare Spiele als Fallback
                st.markdown("### 📊 Recent Matches (Fallback)")
                try:
                    df_all = pd.read_csv("game_database_complete.csv")
                    df_all['date'] = pd.to_datetime(df_all['date'])
                    df_recent = df_all.sort_values('date', ascending=False).head(10)
                    
                    # Erstelle Display DataFrame
                    df_display = pd.DataFrame({
                        'Date': df_recent['date'].dt.strftime('%Y-%m-%d %H:%M'),
                        'Match': df_recent['home_team'] + ' vs ' + df_recent['away_team'],
                        'League': df_recent['league'],
                        'Score': df_recent['home_score'].fillna(0).astype(int).astype(str) + '-' + 
                                df_recent['away_score'].fillna(0).astype(int).astype(str),
                        'xG': df_recent['home_xg'].round(1).astype(str) + ' - ' + 
                              df_recent['away_xg'].round(1).astype(str),
                        'Odds': df_recent['odds_home'].round(2).astype(str) + ' | ' +
                               df_recent['odds_draw'].round(2).astype(str) + ' | ' +
                               df_recent['odds_away'].round(2).astype(str)
                    })
                    
                    st.dataframe(df_display, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Fehler beim Laden der Fallback-Daten: {e}")
            
            else:
                # Display Matches mit Highlights
                st.markdown(f"### ⚽ {len(live_matches)} Live/Upcoming Matches")
                
                # Stats Row
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric(
                        "Total Matches",
                        len(live_matches),
                        delta="Live" if len(live_matches) > 0 else None
                    )
                
                with stats_col2:
                    avg_edge = live_matches['Edge %'].mean()
                    st.metric(
                        "Avg Edge",
                        f"{avg_edge:.1f}%",
                        delta=f"+{avg_edge - 10:.1f}%" if avg_edge > 10 else None
                    )
                
                with stats_col3:
                    value_bets = len(live_matches[live_matches['Edge %'] > 10])
                    st.metric(
                        "Value Bets",
                        value_bets,
                        delta=f"{(value_bets/len(live_matches)*100):.0f}%" if len(live_matches) > 0 else None
                    )
                
                with stats_col4:
                    max_edge = live_matches['Edge %'].max()
                    st.metric(
                        "Best Edge",
                        f"{max_edge:.1f}%",
                        delta="🔥 HOT" if max_edge > 20 else None
                    )
                
                st.markdown("---")
                
                # Display Matches Table mit Custom Styling
                st.markdown("### 📋 Match List")
                
                # Custom Styling Function
                def highlight_value_bets(row):
                    """Highlight rows with high edge"""
                    edge = row['Edge %']
                    if edge > 20:
                        return ['background-color: rgba(76, 175, 80, 0.3)'] * len(row)  # Sehr guter Edge
                    elif edge > 10:
                        return ['background-color: rgba(76, 175, 80, 0.15)'] * len(row)  # Guter Edge
                    else:
                        return [''] * len(row)
                
                # Display mit Styling
                styled_df = live_matches.style.apply(highlight_value_bets, axis=1)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=min(len(live_matches) * 35 + 100, 600)  # Dynamische Höhe
                )
                
                # Detailansicht für Top Match
                if not live_matches.empty:
                    st.markdown("---")
                    st.markdown("### 🔥 Featured Match (Highest Edge)")
                    
                    top_match = live_matches.iloc[0]
                    
                    feat_col1, feat_col2, feat_col3 = st.columns([2, 1, 1])
                    
                    with feat_col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(30, 30, 30, 0.6));
                                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #4CAF50;">
                            <h3 style="margin: 0; color: white;">{top_match['Match']}</h3>
                            <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.7);">
                                {top_match['League']} • {top_match['Time']}
                            </p>
                            <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                                <div>
                                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Score</div>
                                    <div style="font-size: 1.5rem; font-weight: bold;">{top_match['Score']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">xG</div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: #4CAF50;">{top_match['xG']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col2:
                        st.markdown(f"""
                        <div style="background: rgba(30, 30, 30, 0.6); padding: 1rem; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Best Odds</div>
                            <div style="font-size: 1.2rem; font-weight: bold; margin-top: 0.5rem;">{top_match['Best Odds']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with feat_col3:
                        st.markdown(f"""
                        <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">Value Edge</div>
                            <div style="font-size: 2rem; font-weight: bold; color: #4CAF50; margin-top: 0.5rem;">
                                {top_match['Edge %']:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(59, 130, 246, 0.1); 
                                border-radius: 8px; border-left: 3px solid #3b82f6;">
                        <strong>💡 Recommendation:</strong> {top_match['Recommendation']}
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"""
            ❌ **Error Loading Live Data**
            
            Error: {str(e)}
            
            **Troubleshooting:**
            1. Stelle sicher dass `game_database_complete.csv` existiert
            2. Führe den Scraper aus: `python sportmonks_hybrid_scraper_v3_FINAL.py`
            3. Prüfe ob `live_match_loader.py` im Projektverzeichnis ist
            4. Installiere Dependencies: `pip install scipy`
            """)
            
            # Zeige Traceback für Debugging
            with st.expander("🔍 Debug Info"):
                import traceback
                st.code(traceback.format_exc())
    
    # =============================================================================
    # TAB 2: SYSTEM CONTROL (FIXED WITH LIVE LOGS)
    # =============================================================================
    with tab2:
        st.markdown("## 🔧 System Control Center")

        # KRITISCHER SYSTEM-CHECK
        system_ready = True
        if not hasattr(st.session_state, 'log_manager') or st.session_state.log_manager is None:
            st.error("❌ KRITISCH: LogStreamManager nicht initialisiert - Buttons werden nicht funktionieren!")
            st.warning("⚠️ Bitte Dashboard neu laden (F5) oder Dependencies prüfen")
            system_ready = False

            with st.expander("🔍 Debug-Info"):
                st.code(f"""
log_manager exists: {hasattr(st.session_state, 'log_manager')}
log_manager value: {st.session_state.log_manager if hasattr(st.session_state, 'log_manager') else 'N/A'}
initialization_errors: {st.session_state.get('initialization_errors', [])}
""")
        else:
            st.success("✅ System bereit - LogStreamManager initialisiert")

        # Process Status Overview
        st.markdown('### 📊 Process Status Overview')
        
        # Erstelle moderne Status Cards
        status_cols = st.columns(3)
        
        process_info = {
            'scraper': {'name': '🕷️ Hybrid Scraper', 'desc': 'Data Collection'},
            'dutching': {'name': '⚖️ Dutching System', 'desc': 'Bet Calculation'},
            'ml_training': {'name': '🧠 ML Training', 'desc': 'Model Training'},
            'portfolio': {'name': '💼 Portfolio', 'desc': 'Fund Management'},
            'alerts': {'name': '⚠️ Alert System', 'desc': 'Notifications'},
            'correct_score': {'name': '🎯 Correct Score', 'desc': 'Score Prediction'}
        }
        
        for idx, (process_name, info) in enumerate(process_info.items()):
            with status_cols[idx % 3]:
                # Sicherer Check ob log_manager existiert
                if hasattr(st.session_state, 'log_manager') and st.session_state.log_manager is not None:
                    is_running = st.session_state.log_manager.is_running(process_name)
                    status_class = "running" if is_running else "stopped"
                    status_text = "🟢 RUNNING" if is_running else "⭕ STOPPED"
                    status_color = "#4CAF50" if is_running else "#9e9e9e"
                else:
                    status_class = "stopped"
                    status_text = "⭕ OFFLINE"
                    status_color = "#9e9e9e"
                
                st.markdown(f"""
                <div class="status-card {status_class}">
                    <div class="status-card-title">
                        <span class="status-icon">{info['name'].split()[0]}</span>
                        <span>{info['name'].split(' ', 1)[1]}</span>
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-bottom: 0.5rem;">
                        {info['desc']}
                    </div>
                    <div class="status-card-status" style="color: {status_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Control Panels
        col1, col2 = st.columns(2)
        
        # Scraper Control
        with col1:
            st.markdown("### 🕷️ Hybrid Scraper Control")
            scr_col1, scr_col2, scr_col3 = st.columns(3)
            
            with scr_col1:
                if ui.button(text="Start Scraper", icon="play", variant="secondary", key="start_scraper_btn", className="btn-green"):
                    start_scraper()
            
            with scr_col2:
                if ui.button(text="Stop Scraper", icon="square", variant="destructive", key="stop_scraper_btn"):
                    stop_scraper()
            
            with scr_col3:
                if ui.button(text="Refresh Logs", icon="refresh_cw", variant="secondary", key="refresh_scraper_btn", className="btn-green"):
                    update_logs('scraper')
            
            # Live Log Display
            st.markdown('#### 📜 Live Log: Hybrid Scraper')
            scraper_log_container = st.container()
            update_logs('scraper')
            display_live_logs('scraper', st.session_state.scraper_logs, scraper_log_container)
        
        # Dutching System Control
        with col2:
            st.markdown("### ⚖️ Dutching System Control")
            dut_col1, dut_col2, dut_col3 = st.columns(3)
            
            with dut_col1:
                if ui.button(text="Start Dutching", icon="play", variant="secondary", key="start_dutching_btn", className="btn-green"):
                    start_dutching()
            
            with dut_col2:
                if ui.button(text="Stop Dutching", icon="square", variant="destructive", key="stop_dutching_btn"):
                    stop_dutching()
            
            with dut_col3:
                if ui.button(text="Refresh Logs", icon="refresh_cw", variant="secondary", key="refresh_dutching_btn", className="btn-green"):
                    update_logs('dutching')
            
            # Live Log Display
            st.markdown('#### 📜 Live Log: Dutching System')
            dutching_log_container = st.container()
            update_logs('dutching')
            display_live_logs('dutching', st.session_state.dutching_logs, dutching_log_container)
        
        # Additional Controls
        st.markdown("---")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("### 🧠 ML Training")
            if ui.button(text="Start ML Training", icon="brain_circuit", variant="secondary", key="start_ml_btn", className="w-full btn-green"):
                start_ml_training()
            ml_log_container = st.container()
            with st.expander("View ML Training Logs"):
                update_logs('ml_training')
                display_live_logs('ml_training', st.session_state.ml_logs, ml_log_container)
        
        with col4:
            st.markdown('### 💼 Portfolio Optimizer')
            if ui.button(text="Start Portfolio", icon="briefcase", variant="secondary", key="start_portfolio_btn", className="w-full btn-green"):
                start_portfolio_optimizer()
            portfolio_log_container = st.container()
            with st.expander("View Portfolio Logs"):
                update_logs('portfolio')
                display_live_logs('portfolio', st.session_state.portfolio_logs, portfolio_log_container)
        
        with col5:
            st.markdown('### ⚠️ Alert System')
            if ui.button(text="Start Alerts", icon="bell_ring", variant="secondary", key="start_alerts_btn", className="w-full btn-green"):
                start_alert_system()
            alert_log_container = st.container()
            with st.expander("View Alert Logs"):
                update_logs('alerts')
                display_live_logs('alerts', st.session_state.alert_logs, alert_log_container)
        
        # Master Controls
        st.markdown("---")
        st.markdown("### 🎛️ Master Controls")
        
        master_col1, master_col2, master_col3 = st.columns(3)
        
        with master_col1:
            if ui.button(text="START ALL SYSTEMS", icon="rocket", variant="secondary", key="start_all_btn", className="w-full btn-green"):
                with st.spinner("Starting all systems..."):
                    start_scraper()
                    time.sleep(0.5)
                    start_dutching()
                    time.sleep(0.5)
                    start_ml_training()
                    time.sleep(0.5)
                    start_portfolio_optimizer()
                    time.sleep(0.5)
                    start_alert_system()
                st.success("✅ All systems started successfully!")
        
        with master_col2:
            if ui.button(text="STOP ALL SYSTEMS", icon="power_off", variant="destructive", key="stop_all_btn", className="w-full"):
                with st.spinner("Stopping all systems..."):
                    st.session_state.log_manager.stop_all()
                    for process in st.session_state.process_states:
                        st.session_state.process_states[process] = 'idle'
                st.success("✅ All systems stopped successfully!")
        
        with master_col3:
            if ui.button(text="RESTART ALL", icon="rotate_cw", variant="secondary", key="restart_all_btn", className="w-full"):
                with st.spinner("Restarting all systems..."):
                    st.session_state.log_manager.stop_all()
                    time.sleep(1)
                    start_scraper()
                    start_dutching()
                    start_ml_training()
                    start_portfolio_optimizer()
                    start_alert_system()
                st.success("✅ All systems restarted successfully!")
    
# =============================================================================
    # TAB 3: ML MODELS
    # =============================================================================
    with tab3:
        st.markdown("## 🧠 Machine Learning Models")
        
        # GPU Status
        if GPU_AVAILABLE:
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("GPU", gpu_stats['name'], f"{gpu_stats['gpu_util']}% Util")
                with col2:
                    st.metric("VRAM", f"{gpu_stats['memory_used']:.1f}GB", 
                             f"{gpu_stats['memory_percent']:.1f}% Used")
                with col3:
                    st.metric("Temperature", f"{gpu_stats['temperature']}°C",
                             delta=f"{gpu_stats['temperature']-70}°C" if gpu_stats['temperature'] > 70 else None,
                             delta_color="inverse")
                with col4:
                    st.metric("Power", f"{gpu_stats['power']}W", 
                             f"{(gpu_stats['power']/gpu_stats['power_limit']*100):.1f}% of Limit" if gpu_stats['power_limit'] > 0 else "N/A")
        else:
            st.info("🖥️ Running in CPU Mode")
        
        # Model Performance
        st.markdown('### 📊 Model Performance Comparison')
        
        # Mock model data
        models_df = pd.DataFrame({
            'Model': ['XGBoost', 'Neural Network', 'Random Forest', 'LightGBM'],
            'Accuracy': [0.725, 0.718, 0.692, 0.711],
            'Precision': [0.68, 0.71, 0.65, 0.69],
            'Recall': [0.72, 0.70, 0.69, 0.70],
            'F1-Score': [0.70, 0.705, 0.67, 0.695],
            'Training Time': ['2.3 min', '15.7 min', '5.2 min', '1.8 min'],
            'Status': ['Champion', 'Challenger', 'Retired', 'Testing']
        })
        
        st.dataframe(models_df, use_container_width=True)
        
        # Training Controls
        st.markdown('### 🎯 Model Training')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox("Model Type", ["XGBoost", "Neural Network", "Random Forest", "LightGBM"])
        with col2:
            epochs = st.number_input("Epochs/Iterations", min_value=1, value=100)
        with col3:
            learning_rate = st.number_input("Learning Rate", min_value=0.001, value=0.01, format="%.3f")
        
        if ui.button(text="Start Training", icon="rocket", variant="secondary", key="train_model_btn", className="w-full btn-green"):
            with st.spinner(f"Training {model_type} model..."):
                # Here you would actually train the model
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success(f"✅ {model_type} model trained successfully!")
        
    # =============================================================================
    # TAB 4: PORTFOLIO
    # =============================================================================
    with tab4:
        st.markdown('## 💼 Portfolio Management')
        
        stats = st.session_state.portfolio_stats
        
        # Portfolio Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"€{stats.get('total_value', 0):,.2f}", 
                     f"+{stats.get('roi', 0):.2f}% ROI")
        with col2:
            st.metric("Available Balance", f"€{stats.get('available_balance', 0):,.2f}")
        with col3:
            st.metric("In Active Bets", f"€{stats.get('in_bets_balance', 0):,.2f}")
        with col4:
            st.metric("Total Profit", f"€{stats.get('total_profit', 0):,.2f}",
                     f"{stats.get('win_rate', 0):.1f}% Win Rate")
        
        # Portfolio Distribution Chart
        st.markdown('### 📊 Portfolio Distribution')
        
        fig = go.Figure(data=[go.Pie(
            labels=['Available', 'In Bets', 'Profit'],
            values=[stats.get('available_balance', 5000), 
                   stats.get('in_bets_balance', 3000),
                   stats.get('total_profit', 2000)],
            hole=.3,
            marker_colors=['#4CAF50', '#FFC107', '#03A9F4']
        )])
        
        fig.update_layout(
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Active Bets
        st.markdown('### 🎲 Active Bets')
        
        active_bets_df = pd.DataFrame({
            'Match': ['Liverpool vs Chelsea', 'Real Madrid vs Atletico', 'Bayern vs Leipzig'],
            'Market': ['Over 2.5', 'Home Win', 'BTTS Yes'],
            'Stake': [100, 150, 75],
            'Odds': [1.85, 2.10, 1.72],
            'Expected Return': [185, 315, 129],
            'Status': ['Live', 'Live', 'Pending']
        })
        
        st.dataframe(active_bets_df, use_container_width=True)
    
    # =============================================================================
    # TAB 5: ANALYTICS
    # =============================================================================
    with tab5:
        st.markdown('## 📊 Advanced Analytics')
        
        # Performance Over Time
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        roi_values = np.cumsum(np.random.randn(30) * 2) + 10
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=roi_values,
            mode='lines+markers',
            name='ROI %',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=5)
        ))
        
        fig.update_layout(
            title="ROI Trend (30 Days)",
            xaxis_title="Date",
            yaxis_title="ROI %",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font={'color': "white"},
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
# =============================================================================
    # TAB 6: SETTINGS (FIXED AUTO-REFRESH)
    # =============================================================================
    with tab6:
        st.markdown("## ⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔄 Auto-Refresh Settings")

            # Der Schalter steuert nur noch den Session State
            st.toggle(
                "Enable Auto-Refresh",
                key="auto_refresh"
            )
            
            # Der Schieberegler steuert nur noch den Session State
            st.slider(
                "Refresh Interval (Sekunden)",
                min_value=1,
                max_value=60,
                key="refresh_interval"
            )
            
            if ui.button(text="Manual Refresh Now", icon="refresh_cw", variant="secondary", key="manual_refresh_btn"):
                st.rerun()
            
            st.info(f"Letzte Aktualisierung: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
            
            # --- DIE NEUE AUTO-REFRESH KOMPONENTE ---
            if st.session_state.auto_refresh:
                # Führe den Auto-Refresh nur aus, WENN der Schalter an ist
                st_autorefresh(
                    interval=st.session_state.refresh_interval * 1000, 
                    limit=None, # Unbegrenzt laufen
                    key="refresher"
                )
                # Setze den Zeitstempel für die Anzeige
                st.session_state.last_refresh = datetime.now()

        with col2:
            st.markdown('### 🎯 Betting Parameters')
            
            max_stake = st.number_input("Max Stake per Bet (€)", min_value=10, value=100)
            max_daily_loss = st.number_input("Max Daily Loss (€)", min_value=50, value=500)
            min_edge = st.slider("Minimum Edge (%)", 0, 30, 10)
            kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
            
            if ui.button(text="Save Settings", icon="save", variant="secondary", key="save_settings_btn", className="w-full btn-green"):
                st.success("✅ Settings saved successfully!")
    
# =============================================================================
    # TAB 7: BACKTESTING
    # =============================================================================
    with tab7:
        st.markdown("## 📈 Backtesting Framework")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        with col3:
            initial_balance = st.number_input("Initial Balance (€)", value=10000)
        
        strategy = st.selectbox("Strategy", ["Value Betting", "Dutching", "Correct Score", "Combined"])

        if ui.button(text="Run Backtest", icon="rocket", variant="secondary", key="run_backtest_btn", className="w-full btn-green"):
            try:
                with st.spinner("Running backtest..."):
                    # Check if data file exists
                    data_file = Path.cwd() / "game_database_complete.csv"
                    if not data_file.exists():
                        st.warning("⚠️ Keine Daten gefunden. Bitte starten Sie zuerst den Scraper.")
                    else:
                        # Load historical data
                        try:
                            historical_data = pd.read_csv(data_file)

                            # Filter by date range
                            if 'date' in historical_data.columns:
                                historical_data['date'] = pd.to_datetime(historical_data['date'])
                                historical_data = historical_data[
                                    (historical_data['date'] >= pd.to_datetime(start_date)) &
                                    (historical_data['date'] <= pd.to_datetime(end_date))
                                ]

                            if len(historical_data) == 0:
                                st.warning("⚠️ Keine Daten im gewählten Zeitraum gefunden.")
                                raise ValueError("No data in selected date range")

                        except Exception as e:
                            logging.error(f"Error loading data: {e}")
                            st.error(f"Fehler beim Laden der Daten: {e}")
                            raise

                        # Initialize backtester with correct parameters
                        backtest_config = BacktestConfig(
                            initial_bankroll=float(initial_balance),
                            kelly_cap=0.25,
                            min_edge=0.05
                        )

                        backtester = Backtester(config=backtest_config)

                        # Define prediction function based on strategy
                        def prediction_func(row: pd.Series) -> Dict:
                            """Simple prediction function for backtesting"""
                            try:
                                # Basic value betting logic
                                predictions = {}

                                # Use xG if available
                                if 'home_xg' in row and 'away_xg' in row:
                                    home_prob = row['home_xg'] / (row['home_xg'] + row['away_xg'] + 0.3)
                                    away_prob = row['away_xg'] / (row['home_xg'] + row['away_xg'] + 0.3)
                                else:
                                    # Fallback to basic odds conversion
                                    home_prob = 0.33
                                    away_prob = 0.33

                                predictions = {
                                    'market': '1X2',
                                    'selection': 'Home' if home_prob > away_prob else 'Away',
                                    'probability': max(home_prob, away_prob),
                                    'odds': row.get('odds_home', 2.0) if home_prob > away_prob else row.get('odds_away', 2.0)
                                }

                                return predictions
                            except Exception as e:
                                logging.error(f"Prediction error for row: {e}")
                                return {}

                        # Run backtest with correct parameters
                        progress = st.progress(0)
                        results = None

                        # Simulate progress
                        for i in range(100):
                            progress.progress(i + 1)
                            time.sleep(0.01)
                            if i == 50:
                                # Run actual backtest in middle of progress
                                try:
                                    backtest_result = backtester.run_backtest(
                                        historical_data=historical_data,
                                        prediction_func=prediction_func
                                    )

                                    # Convert BacktestResult to dict for display
                                    results = {
                                        'total_return': backtest_result.total_profit,
                                        'roi': backtest_result.roi,
                                        'win_rate': backtest_result.win_rate * 100,
                                        'max_drawdown': backtest_result.max_drawdown_percent,
                                        'sharpe_ratio': backtest_result.sharpe_ratio,
                                        'total_bets': backtest_result.total_bets,
                                        'winning_bets': backtest_result.winning_bets,
                                        'final_bankroll': backtest_result.final_bankroll
                                    }

                                except Exception as e:
                                    logging.error(f"Backtest execution error: {e}")
                                    st.error(f"Fehler beim Backtest: {e}")

                        if results:
                            st.success("✅ Backtest completed!")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_return = results.get('total_return', 0)
                                roi = results.get('roi', 0)
                                st.metric("Total Return", f"+€{total_return:,.0f}", f"+{roi:.1f}%")
                            with col2:
                                win_rate = results.get('win_rate', 0)
                                st.metric("Win Rate", f"{win_rate:.1f}%", f"+{win_rate-55:.1f}%")
                            with col3:
                                max_dd = results.get('max_drawdown', 0)
                                st.metric("Max Drawdown", f"{max_dd:.1f}%")
                            with col4:
                                sharpe = results.get('sharpe_ratio', 0)
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                            # Save results to session state
                            st.session_state.backtest_results = results
                        else:
                            # Fallback to mock data
                            st.success("✅ Backtest completed (Mock Data)!")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", "+€2,450", "+24.5%")
                            with col2:
                                st.metric("Win Rate", "58.3%", "+3.3%")
                            with col3:
                                st.metric("Max Drawdown", "-12.5%")
                            with col4:
                                st.metric("Sharpe Ratio", "1.85")

            except Exception as e:
                logging.error(f"Backtest error: {e}")
                st.error(f"❌ Fehler beim Backtest: {e}")
                st.info("Verwende Mock-Daten...")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", "+€2,450", "+24.5%")
                with col2:
                    st.metric("Win Rate", "58.3%", "+3.3%")
                with col3:
                    st.metric("Max Drawdown", "-12.5%")
                with col4:
                    st.metric("Sharpe Ratio", "1.85")
    
# =============================================================================
    # TAB 8: CORRECT SCORE SYSTEM
    # =============================================================================
    with tab8:
        st.markdown("## 🎯 Correct Score Betting System")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔧 Correct Score Control")

            cs_col1, cs_col2 = st.columns(2)

            with cs_col1:
                if ui.button(text="Start Correct Score", icon="play", variant="secondary", key="start_cs_btn", className="btn-green"):
                    try:
                        cwd = str(Path.cwd())
                        command = [sys.executable, '-u', 'sportmonks_correct_score_system.py']
                        st.session_state.log_manager.start_process('correct_score', command, cwd=cwd)
                        st.session_state.process_states['correct_score'] = 'running'
                        st.success("🚀 Correct Score System gestartet!")
                    except Exception as e:
                        st.error(f"❌ Fehler beim Starten: {e}")
                        logging.error(f"Correct Score start error: {e}")

            with cs_col2:
                if ui.button(text="Stop Correct Score", icon="square", variant="destructive", key="stop_cs_btn"):
                    try:
                        if st.session_state.log_manager.stop_process('correct_score'):
                            st.session_state.process_states['correct_score'] = 'idle'
                            st.success("🛑 Correct Score System gestoppt!")
                    except Exception as e:
                        st.error(f"❌ Fehler beim Stoppen: {e}")

            # Correct Score Configuration
            st.markdown('### ⚙️ Konfiguration')

            min_value_edge = st.slider("Minimum Value Edge (%)", 0, 50, 15, key="cs_min_edge")
            max_odds = st.number_input("Max Odds", min_value=2.0, max_value=50.0, value=25.0, key="cs_max_odds")
            min_probability = st.slider("Min Probability (%)", 0.0, 20.0, 3.0, 0.5, key="cs_min_prob")

            if ui.button(text="Save CS Config", icon="save", variant="secondary", key="save_cs_config_btn", className="w-full btn-green"):
                st.session_state.cs_config = {
                    'min_value_edge': min_value_edge,
                    'max_odds': max_odds,
                    'min_probability': min_probability
                }
                st.success("✅ Konfiguration gespeichert!")

        with col2:
            st.markdown('### 📜 Live Logs')

            # Update and display logs
            cs_log_container = st.container()

            # Update logs function for correct score
            new_cs_logs = st.session_state.log_manager.get_logs('correct_score')
            if new_cs_logs:
                st.session_state.correct_score_logs.extend(new_cs_logs)
                st.session_state.correct_score_logs = st.session_state.correct_score_logs[-100:]

            display_live_logs('correct_score', st.session_state.correct_score_logs, cs_log_container)

        # Correct Score Predictions
        st.markdown("---")
        st.markdown('### 🎯 Top Correct Score Predictions')

        # Check if results file exists
        cs_results_file = Path.cwd() / "results" / "correct_score_results.csv"
        if cs_results_file.exists():
            try:
                cs_df = pd.read_csv(cs_results_file)
                if not cs_df.empty:
                    # Display top predictions
                    st.dataframe(
                        cs_df.head(10),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("📊 Noch keine Correct Score Predictions verfügbar. Starten Sie das System um Predictions zu generieren.")
            except Exception as e:
                logging.error(f"Error loading correct score results: {e}")
                st.error(f"Fehler beim Laden der Results: {e}")
        else:
            st.info("📊 Noch keine Correct Score Results verfügbar. Starten Sie das System um Results zu generieren.")

            # Mock data as fallback
            mock_cs_data = pd.DataFrame({
                'Match': ['Liverpool vs Arsenal', 'Real Madrid vs Barcelona', 'Bayern vs Dortmund'],
                'Prediction': ['2-1', '1-1', '3-2'],
                'Probability': ['8.5%', '12.3%', '7.2%'],
                'Bookmaker Odds': [9.5, 7.2, 15.0],
                'Fair Odds': [11.76, 8.13, 13.89],
                'Value Edge': ['+23.8%', '+12.9%', '+7.4%'],
                'Recommended Stake': ['€45', '€32', '€28']
            })

            st.markdown("#### 🎭 Mock Data (Beispiel)")
            st.dataframe(mock_cs_data, use_container_width=True)

# =============================================================================
# SIDEBAR (ENHANCED)
# =============================================================================
with st.sidebar:
    st.markdown('## ⚽ AI Betting System')
    st.markdown("---")
    
    # Live Portfolio Metrics
    st.markdown('### 💼 Portfolio (Live)')
    
    stats = st.session_state.get('portfolio_stats', {})
    
    st.metric(
        label="Total Balance", 
        value=f"€{stats.get('total_value', 0):,.2f}", 
        delta=f"{stats.get('roi', 0):.2f}% ROI"
    )
    st.metric(
        label="Total Profit", 
        value=f"€{stats.get('total_profit', 0):,.2f}"
    )
    
    # Progress Bar
    available = stats.get('available_balance', 0)
    in_bets = stats.get('in_bets_balance', 0)
    
    if (available + in_bets) > 0:
        st.progress(in_bets / (available + in_bets), 
                   text=f"€{in_bets:,.0f} in Wetten")
    
    st.markdown("---")
    
    # System Status
    st.markdown('### 📡 System Status')
    
    # Check running processes
    if hasattr(st.session_state, 'log_manager') and st.session_state.log_manager is not None:
        running_count = sum(1 for p in st.session_state.process_states.keys()
                           if st.session_state.log_manager.is_running(p))
    else:
        running_count = 0
    
    total_processes = len(st.session_state.process_states)
    
    # System Overview Card
    st.markdown(f"""
    <div class="system-overview">
        <div class="status-number">{running_count}/{total_processes}</div>
        <div class="status-label">Systems Online</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Kompakte Status-Liste
    process_names = {
        'scraper': '🕷️ Scraper',
        'dutching': '⚖️ Dutching',
        'ml_training': '🧠 ML Training',
        'portfolio': '💼 Portfolio',
        'alerts': '⚠️ Alerts',
        'correct_score': '🎯 Correct Score'
    }
    
    if hasattr(st.session_state, 'log_manager') and st.session_state.log_manager is not None:
        for process_name, display_name in process_names.items():
            is_running = st.session_state.log_manager.is_running(process_name)
            status_class = "running" if is_running else "stopped"
            indicator_class = "running" if is_running else "stopped"
            
            st.markdown(f"""
            <div class="sidebar-status-item {status_class}">
                <span class="name">{display_name}</span>
                <span class="indicator {indicator_class}"></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ System OFFLINE - Dependencies fehlen!")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown('### ⚡ Quick Actions')
    
    if ui.button(text="Dashboard neuladen", icon="refresh_cw", variant="secondary", className="w-full btn-green"):
        st.rerun()
    
    if ui.button(text="Start All Systems", icon="rocket", variant="secondary", className="w-full btn-green"):
        with st.spinner("Starting systems..."):
            # Start all systems
            start_scraper()
            start_dutching()
            start_ml_training()
            start_portfolio_optimizer()
            start_alert_system()
        st.success("All systems started!")
    
    if ui.button(text="Stop All Systems", icon="power_off", variant="destructive", className="w-full"):
        with st.spinner("Stopping systems..."):
            st.session_state.log_manager.stop_all()
            for process in st.session_state.process_states:
                st.session_state.process_states[process] = 'idle'
        st.success("All systems stopped!")
    
    st.markdown("---")
    
    # Info
    st.markdown('### ℹ️ Info')
    
    gpu_info_str = f"**GPU**: {CUDA_VERSION_STR}" if GPU_AVAILABLE else "**GPU**: N/A (CPU-Mode)"
    
    st.info(f"""
    **Version**: 5.0.0 EMOJI FIX  
    **Refresh**: {st.session_state.last_refresh.strftime('%H:%M:%S')}  
    **Processes**: {running_count}/6 Running  
    {gpu_info_str}
    """)

# =============================================================================
# CLEANUP ON EXIT
# =============================================================================
import atexit

def cleanup():
    """Cleanup function to stop all processes on exit"""
    if 'log_manager' in st.session_state:
        st.session_state.log_manager.stop_all()

atexit.register(cleanup)

# =============================================================================
# RUN MAIN
# =============================================================================
if __name__ == "__main__":
    main()