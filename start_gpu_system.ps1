# ============================================================
# AI DUTCHING GPU SYSTEM - WINDOWS POWERSHELL LAUNCHER
# ============================================================
# Optimiert für Windows Server + RTX 3090
#
# Features:
# - GPU Detection & Verification
# - CUDA Check
# - Dependency Check
# - System Health Monitoring
# - Auto-Start Options
# ============================================================

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  AI DUTCHING GPU SYSTEM - WINDOWS SERVER" -ForegroundColor Cyan
Write-Host "  RTX 3090 Optimized" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan

# ============================================================
# SCHRITT 1: SYSTEM CHECK
# ============================================================
Write-Host "[1/5] System Check..." -ForegroundColor Yellow

# Python Version Check
Write-Host "  - Checking Python..." -NoNewline
try {
    $pythonVersion = python --version 2>&1
    Write-Host " OK" -ForegroundColor Green
    Write-Host "    $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host " FEHLER" -ForegroundColor Red
    Write-Host "    Python nicht gefunden!" -ForegroundColor Red
    Write-Host "    Bitte installiere Python 3.10 oder 3.11" -ForegroundColor Yellow
    exit 1
}

# GPU Check
Write-Host "  - Checking GPU..." -NoNewline
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
        Write-Host "    $gpuInfo" -ForegroundColor Gray
    } else {
        Write-Host " WARNUNG" -ForegroundColor Yellow
        Write-Host "    nvidia-smi nicht verfügbar - GPU eventuell nicht erkannt" -ForegroundColor Yellow
    }
} catch {
    Write-Host " WARNUNG" -ForegroundColor Yellow
    Write-Host "    Nvidia Treiber eventuell nicht installiert" -ForegroundColor Yellow
}

# ============================================================
# SCHRITT 2: CUDA CHECK
# ============================================================
Write-Host "`n[2/5] CUDA Check..." -ForegroundColor Yellow

Write-Host "  - Checking PyTorch CUDA..." -NoNewline
try {
    $cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1

    if ($cudaCheck -match "CUDA: True") {
        Write-Host " OK" -ForegroundColor Green
        $cudaCheck -split "`n" | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    } else {
        Write-Host " FEHLER" -ForegroundColor Red
        Write-Host "    CUDA nicht verfügbar!" -ForegroundColor Red
        Write-Host "    System läuft im CPU-Modus (sehr langsam!)" -ForegroundColor Yellow
        Write-Host "    Installiere PyTorch mit CUDA:" -ForegroundColor Yellow
        Write-Host "    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Cyan
    }
} catch {
    Write-Host " FEHLER" -ForegroundColor Red
    Write-Host "    PyTorch nicht installiert!" -ForegroundColor Red
}

# ============================================================
# SCHRITT 3: DEPENDENCY CHECK
# ============================================================
Write-Host "`n[3/5] Dependency Check..." -ForegroundColor Yellow

$requiredPackages = @(
    "pandas",
    "numpy",
    "torch",
    "xgboost",
    "streamlit",
    "plotly"
)

$missingPackages = @()

foreach ($package in $requiredPackages) {
    Write-Host "  - Checking $package..." -NoNewline
    try {
        $check = python -c "import $package" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
        } else {
            Write-Host " FEHLT" -ForegroundColor Red
            $missingPackages += $package
        }
    } catch {
        Write-Host " FEHLT" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "`n  WARNUNG: Fehlende Pakete gefunden!" -ForegroundColor Yellow
    Write-Host "  Fehlende Pakete: $($missingPackages -join ', ')" -ForegroundColor Yellow
    Write-Host "  Installiere mit: pip install -r requirements.txt" -ForegroundColor Cyan

    $install = Read-Host "`n  Sollen die Pakete jetzt installiert werden? (j/n)"
    if ($install -eq "j" -or $install -eq "J") {
        Write-Host "`n  Installiere Dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
}

# ============================================================
# SCHRITT 4: MENU
# ============================================================
Write-Host "`n[4/5] System Ready!" -ForegroundColor Green
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  HAUPTMENU" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  1) Start Dashboard (Web UI)" -ForegroundColor White
Write-Host "  2) Start GPU Training (Neural Network + XGBoost)" -ForegroundColor White
Write-Host "  3) Start Continuous Training System" -ForegroundColor White
Write-Host "  4) Start Deep RL Cashout Optimizer Training" -ForegroundColor White
Write-Host "  5) Run GPU Performance Test" -ForegroundColor White
Write-Host "  6) Run Scraper (Fetch xG Data)" -ForegroundColor White
Write-Host "  7) Run Dutching System (Live Betting)" -ForegroundColor White
Write-Host "  8) GPU Monitor (Performance Tracking)" -ForegroundColor White
Write-Host "  9) System Verification" -ForegroundColor White
Write-Host "  0) Exit" -ForegroundColor White
Write-Host "============================================================`n" -ForegroundColor Cyan

$choice = Read-Host "Wähle eine Option (0-9)"

# ============================================================
# SCHRITT 5: EXECUTE
# ============================================================
Write-Host "`n[5/5] Executing..." -ForegroundColor Yellow

switch ($choice) {
    "1" {
        Write-Host "`nStarting Dashboard..." -ForegroundColor Green
        Write-Host "  Öffne Browser: http://localhost:8501" -ForegroundColor Cyan
        streamlit run dashboard.py
    }
    "2" {
        Write-Host "`nStarting GPU Training..." -ForegroundColor Green
        python gpu_ml_models.py
    }
    "3" {
        Write-Host "`nStarting Continuous Training System..." -ForegroundColor Green
        python continuous_training_system.py
    }
    "4" {
        Write-Host "`nStarting Deep RL Training..." -ForegroundColor Green
        python gpu_deep_rl_cashout.py
    }
    "5" {
        Write-Host "`nRunning GPU Performance Test..." -ForegroundColor Green
        Write-Host "  Testing Neural Network..." -ForegroundColor Yellow
        python -c "from gpu_ml_models import GPUNeuralNetworkPredictor, GPUConfig; import numpy as np; config = GPUConfig(); model = GPUNeuralNetworkPredictor(input_size=20, gpu_config=config); X = np.random.rand(1000, 20).astype(np.float32); y = np.random.randint(0, 3, 1000); model.train(X, y, epochs=10, batch_size=256, verbose=True)"

        Write-Host "`n  Testing XGBoost GPU..." -ForegroundColor Yellow
        python -c "from gpu_ml_models import GPUXGBoostPredictor; import numpy as np; model = GPUXGBoostPredictor(use_gpu=True); X = np.random.rand(1000, 20).astype(np.float32); y = np.random.randint(0, 3, 1000); model.train(X, y, verbose=True); print('GPU XGBoost Test OK!')"

        Write-Host "`n  GPU Performance Test abgeschlossen!" -ForegroundColor Green
    }
    "6" {
        Write-Host "`nStarting Scraper..." -ForegroundColor Green
        python sportmonks_xg_scraper.py
    }
    "7" {
        Write-Host "`nStarting Dutching System..." -ForegroundColor Green
        python sportmonks_dutching_system.py
    }
    "8" {
        Write-Host "`nStarting GPU Monitor..." -ForegroundColor Green
        python gpu_performance_monitor.py
    }
    "9" {
        Write-Host "`nRunning System Verification..." -ForegroundColor Green
        python verify_integration.py
    }
    "0" {
        Write-Host "`nExit. Auf Wiedersehen!" -ForegroundColor Green
        exit 0
    }
    default {
        Write-Host "`nUngültige Auswahl!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Programm beendet" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan
