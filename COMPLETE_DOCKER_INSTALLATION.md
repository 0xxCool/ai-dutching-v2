# 🚀 VOLLSTÄNDIGE DOCKER INSTALLATION FÜR AI-DUTCHING ML SYSTEM

## 📋 Systemanforderungen

### Hardware
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **CPU:** Min. 8 Cores empfohlen
- **RAM:** Min. 16GB, 32GB empfohlen
- **Storage:** Min. 50GB freier Speicherplatz

### Software
- **Docker Desktop** mit WSL2 Backend
- **NVIDIA Docker Toolkit** (für GPU Support)
- **CUDA 12.1+** kompatible GPU Treiber

---

## 🐳 Docker Container Setup

### Docker Container herunterladen:

```bash
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

### Option 1: PyTorch Base Image (EMPFOHLEN)

```bash
# Starte Container mit GPU Support
docker run -it --rm \
  --name ai-dutching-ml \
  --gpus all \
  --shm-size=8g \
  -v /mnt/c/ai-dutching-v1:/workspace \
  -w /workspace \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel \
  bash
```

### Option 2: RAPIDS AI Image (Mit vorinstalliertem XGBoost GPU)

```bash
docker run -it --rm \
  --name ai-dutching-rapids \
  --gpus all \
  --shm-size=8g \
  -v /mnt/c/ai-dutching-v1:/workspace \
  -w /workspace \
  rapidsai/rapidsai:24.12-cuda12.5-py3.11 \
  bash
```

---

## 📦 KOMPLETTE PAKET-INSTALLATION

### Schritt 1: System Updates & Build Tools

```bash
# System Updates
apt-get update && apt-get upgrade -y

# Build Tools & Dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    nvidia-utils-545 \
    libgomp1 \
    libnccl2 \
    libnccl-dev
```

### Schritt 2: Python Package Manager Updates

```bash
# Pip aktualisieren
pip install --upgrade pip setuptools wheel

# Conda aktualisieren (falls vorhanden)
conda update -n base conda -y
```

### Schritt 3: Core ML Libraries

```bash
# ========================================
# PYTORCH & DEEP LEARNING
# ========================================
# PyTorch mit CUDA (falls nicht im Base Image)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ========================================
# XGBOOST MIT GPU SUPPORT (WICHTIG!)
# ========================================
# Entferne alte XGBoost Versionen
pip uninstall xgboost -y
conda remove xgboost py-xgboost libxgboost _py-xgboost-mutex -y 2>/dev/null || true

# Installiere XGBoost mit GPU Support
pip install xgboost==2.1.2 --no-cache-dir

# Alternative falls obiges fehlschlägt:
# conda install -c conda-forge py-xgboost-gpu -y

# ========================================
# LIGHTGBM MIT GPU SUPPORT
# ========================================
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

### Schritt 4: Data Science & ML Libraries

```bash
# ========================================
# DATA SCIENCE ESSENTIALS
# ========================================
pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4

# ========================================
# FEATURE ENGINEERING & PREPROCESSING
# ========================================
pip install \
    imbalanced-learn==0.12.4 \
    category_encoders==2.6.4 \
    feature-engine==1.8.1

# ========================================
# MODEL OPTIMIZATION & TUNING
# ========================================
pip install \
    optuna==4.1.0 \
    hyperopt==0.2.7 \
    ray[tune]==2.40.0 \
    scikit-optimize==0.10.2

# ========================================
# VISUALIZATION & MONITORING
# ========================================
pip install \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    plotly==5.24.1 \
    tensorboard==2.18.0 \
    wandb==0.18.7

# ========================================
# PROGRESS & LOGGING
# ========================================
pip install \
    tqdm==4.67.1 \
    rich==13.9.4 \
    colorama==0.4.6 \
    loguru==0.7.2
```

### Schritt 5: Database & API Libraries

```bash
# ========================================
# DATABASE CONNECTIVITY
# ========================================
pip install \
    psycopg2-binary==2.9.10 \
    sqlalchemy==2.0.36 \
    pymongo==4.10.1 \
    redis==5.2.0

# ========================================
# WEB & API TOOLS
# ========================================
pip install \
    requests==2.32.3 \
    aiohttp==3.11.10 \
    beautifulsoup4==4.12.3 \
    selenium==4.27.1 \
    scrapy==2.12.0

# ========================================
# DATA FORMATS & SERIALIZATION
# ========================================
pip install \
    pyyaml==6.0.2 \
    toml==0.10.2 \
    python-dotenv==1.0.1 \
    jsonschema==4.23.0
```

### Schritt 6: ML Tracking & Deployment

```bash
# ========================================
# EXPERIMENT TRACKING
# ========================================
pip install \
    mlflow==2.18.0 \
    sacred==0.8.5 \
    neptune-client==1.12.0

# ========================================
# MODEL SERVING & DEPLOYMENT
# ========================================
pip install \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    streamlit==1.41.0 \
    gradio==5.7.1

# ========================================
# TESTING & VALIDATION
# ========================================
pip install \
    pytest==8.3.4 \
    pytest-cov==6.0.0 \
    hypothesis==6.122.1
```

### Schritt 7: Jupyter & Development Tools

```bash
# ========================================
# JUPYTER ECOSYSTEM
# ========================================
pip install \
    jupyter==1.1.1 \
    jupyterlab==4.3.3 \
    ipykernel==6.29.5 \
    ipywidgets==8.1.5 \
    nbconvert==7.16.4

# Jupyter Kernel registrieren
python -m ipykernel install --user --name ai-dutching --display-name "AI Dutching"

# ========================================
# CODE QUALITY & FORMATTING
# ========================================
pip install \
    black==24.10.0 \
    flake8==7.1.1 \
    pylint==3.3.2 \
    mypy==1.13.0 \
    isort==5.13.2
```

### Schritt 8: GPU Monitoring & Optimization

```bash
# ========================================
# GPU UTILITIES
# ========================================
pip install \
    nvidia-ml-py==12.560.30 \
    gpustat==1.1.1 \
    py3nvml==0.2.7 \
    pynvml==11.5.3

# ========================================
# PERFORMANCE PROFILING
# ========================================
pip install \
    memory_profiler==0.61.0 \
    line_profiler==4.2.0 \
    py-spy==0.4.0
```

---

## 🔧 VERIFIKATION DER INSTALLATION

### Erstelle Test-Skript

```bash
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Verifiziere die vollständige Installation
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def check_package(name, import_name=None):
    """Prüfe ob Paket installiert ist"""
    import_name = import_name or name
    try:
        __import__(import_name)
        return True, "✅"
    except ImportError:
        return False, "❌"

def main():
    print("=" * 60)
    print("🔍 INSTALLATION VERIFICATION")
    print("=" * 60)
    
    # Core Packages
    packages = [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("Optuna", "optuna"),
        ("MLflow", "mlflow"),
        ("FastAPI", "fastapi"),
        ("Requests", "requests"),
        ("TQDM", "tqdm"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("PostgreSQL", "psycopg2"),
        ("Redis", "redis"),
        ("BeautifulSoup", "bs4"),
        ("Selenium", "selenium"),
        ("PyYAML", "yaml"),
        ("python-dotenv", "dotenv"),
        ("Jupyter", "jupyter"),
    ]
    
    print("\n📦 CORE PACKAGES:")
    print("-" * 40)
    all_ok = True
    for name, import_name in packages:
        ok, icon = check_package(import_name)
        print(f"{icon} {name:20} {'Installed' if ok else 'Missing'}")
        if not ok:
            all_ok = False
    
    # GPU Check
    print("\n🚀 GPU SUPPORT:")
    print("-" * 40)
    
    # PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ PyTorch CUDA:     Available")
            print(f"   GPU:              {gpu_name}")
            print(f"   VRAM:             {gpu_memory:.1f} GB")
            print(f"   CUDA Version:     {torch.version.cuda}")
        else:
            print("❌ PyTorch CUDA:     Not Available")
    except Exception as e:
        print(f"❌ PyTorch CUDA:     Error - {e}")
    
    # XGBoost GPU
    try:
        import xgboost as xgb
        import numpy as np
        
        # Test GPU Training
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'multi:softmax',
            'num_class': 3
        }
        
        bst = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print(f"✅ XGBoost GPU:      Available")
    except Exception as e:
        print(f"❌ XGBoost GPU:      Not Available")
        print(f"   Error: {str(e)[:50]}...")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok and cuda_available:
        print("✅ INSTALLATION COMPLETE - All Systems Go!")
    else:
        print("⚠️  Some packages missing or GPU not available")
        print("   Run the installation commands for missing packages")
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

# Führe Verifikation aus
python verify_installation.py
```

---

## 🎯 PROJEKT-SPEZIFISCHE SETUP

### Erstelle Projektstruktur

```bash
# Arbeitsverzeichnis
cd /workspace

# Erstelle Verzeichnisstruktur
mkdir -p models/registry
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p notebooks
mkdir -p outputs

# Setze Berechtigungen
chmod -R 755 .
```

### Environment Variables (.env)

```bash
cat > .env << 'EOF'
# API Keys
SPORTMONKS_API_KEY=your_api_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_dutching
DB_USER=postgres
DB_PASSWORD=your_password

# Model Settings
MODEL_DIR=/workspace/models
DATA_DIR=/workspace/data
LOG_DIR=/workspace/logs

# GPU Settings
CUDA_VISIBLE_DEVICES=0
XGB_USE_GPU=1
TF_FORCE_GPU_ALLOW_GROWTH=true

# Training
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=100
EARLY_STOPPING_PATIENCE=15
EOF
```

---

## 🚨 TROUBLESHOOTING

### Problem: XGBoost GPU nicht verfügbar

```bash
# Lösung 1: Reinstall mit pip
pip uninstall xgboost -y
pip install xgboost --no-cache-dir

# Lösung 2: Build from Source
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
make -j$(nproc)
cd ../python-package
pip install -e .
```

### Problem: CUDA Out of Memory

```bash
# Lösung: Reduziere Batch Size oder verwende Gradient Accumulation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Oder in Python:
import torch
torch.cuda.empty_cache()
```

### Problem: Container startet nicht mit GPU

```bash
# Prüfe Docker GPU Support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Installiere NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 📝 FINALE CHECKLISTE

- [ ] Docker Container gestartet mit GPU Support
- [ ] Alle Python Packages installiert
- [ ] XGBoost GPU Support verifiziert
- [ ] PyTorch CUDA verfügbar
- [ ] Projektstruktur erstellt
- [ ] Environment Variables gesetzt
- [ ] Test-Skript erfolgreich durchgelaufen

---

## 🚀 TRAINING STARTEN

Nach erfolgreicher Installation:

```bash
# Teste GPU Performance
python test_xgboost_gpu.py

# Starte Training
python train_ml_models.py

# Starte Dutching System
python sportmonks_dutching_system.py
```

---

**✅ Installation abgeschlossen! Viel Erfolg mit dem Training!**