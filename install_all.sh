#!/bin/bash
# ============================================================
# MASTER INSTALLATION SCRIPT
# ============================================================
# Dieses Skript installiert zuerst die GPU-spezifischen Pakete
# und dann den Rest aus der requirements.txt.
#
# Ausführen mit:
# chmod +x install_all.sh
# ./install_all.sh
# ============================================================

# Stoppt das Skript sofort, wenn ein Befehl fehlschlägt
set -e

echo "➡️ Schritt 1: Installiere PyTorch für CUDA 12.1..."
# Installiert PyTorch mit CUDA 12.1 Support 
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

echo "➡️ Schritt 2: Installiere XGBoost für GPU..."
# Installiert XGBoost mit GPU Support 
pip install xgboost==2.1.2 --no-cache-dir 

echo "➡️ Schritt 3: Installiere LightGBM für GPU..."
# Installiert LightGBM und kompiliert es mit GPU-Support 
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON 

echo "➡️ Schritt 4: Installiere alle verbleibenden Pakete..."
# Installiert alle anderen Pakete aus der bereinigten Datei
pip install -r requirements.txt

echo "✅✅✅ Installation abgeschlossen! ✅✅✅"
echo "Führe jetzt deine Verifizierung durch:"
echo "python verify_installation.py"
