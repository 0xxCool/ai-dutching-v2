#!/usr/bin/env python3
"""
XGBoost GPU Support Test und Konfiguration
==========================================
Dieses Skript testet und demonstriert die korrekte GPU-Nutzung mit XGBoost
"""

import numpy as np
import xgboost as xgb
import torch
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def check_gpu_environment():
    """Überprüfe GPU Umgebung"""
    print("=" * 60)
    print("🔍 SYSTEM CHECK")
    print("=" * 60)
    
    # XGBoost Version
    print(f"XGBoost Version: {xgb.__version__}")
    
    # PyTorch CUDA Check
    if torch.cuda.is_available():
        print(f"✅ CUDA verfügbar: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ Keine CUDA GPU gefunden")
    
    # Test XGBoost GPU Support
    try:
        # Kleiner Test-Datensatz
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        dtrain = xgb.DMatrix(X_test, label=y_test)
        
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        bst = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("✅ XGBoost GPU Support: AKTIV")
        return True
    except Exception as e:
        print(f"❌ XGBoost GPU Support: NICHT VERFÜGBAR")
        print(f"   Fehler: {str(e)[:100]}...")
        return False

def train_xgboost_comparison(X_train, X_test, y_train, y_test):
    """Vergleiche CPU vs GPU Training"""
    
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE VERGLEICH")
    print("=" * 60)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Gemeinsame Parameter
    base_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss'
    }
    
    num_rounds = 100
    
    # CPU Training
    print("\n🖥️  CPU Training...")
    cpu_params = {**base_params, 'tree_method': 'hist'}
    
    start_time = time.time()
    cpu_model = xgb.train(
        cpu_params, 
        dtrain, 
        num_boost_round=num_rounds,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )
    cpu_time = time.time() - start_time
    
    cpu_pred = cpu_model.predict(dtest)
    cpu_accuracy = np.mean(cpu_pred == y_test)
    
    print(f"   Zeit: {cpu_time:.2f}s")
    print(f"   Accuracy: {cpu_accuracy:.4f}")
    
    # GPU Training (falls verfügbar)
    try:
        print("\n🚀 GPU Training...")
        gpu_params = {**base_params, 'tree_method': 'gpu_hist', 'gpu_id': 0}
        
        start_time = time.time()
        gpu_model = xgb.train(
            gpu_params, 
            dtrain, 
            num_boost_round=num_rounds,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )
        gpu_time = time.time() - start_time
        
        gpu_pred = gpu_model.predict(dtest)
        gpu_accuracy = np.mean(gpu_pred == y_test)
        
        print(f"   Zeit: {gpu_time:.2f}s")
        print(f"   Accuracy: {gpu_accuracy:.4f}")
        print(f"\n⚡ Speedup: {cpu_time/gpu_time:.2f}x schneller auf GPU!")
        
    except Exception as e:
        print(f"   ❌ GPU Training fehlgeschlagen: {str(e)[:100]}...")

def get_correct_xgboost_params(use_gpu=True):
    """Gibt die korrekten XGBoost Parameter zurück"""
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    if use_gpu:
        # GPU spezifische Parameter
        params.update({
            'tree_method': 'gpu_hist',  # WICHTIG: gpu_hist für GPU
            'gpu_id': 0,                 # GPU ID (0 für erste GPU)
            'predictor': 'gpu_predictor', # Optional: GPU Predictor
            'max_bin': 256               # Kann bei GPU erhöht werden
        })
    else:
        # CPU Parameter
        params.update({
            'tree_method': 'hist',       # oder 'exact' für kleine Datensätze
            'predictor': 'cpu_predictor'
        })
    
    return params

def main():
    """Hauptfunktion"""
    
    # 1. Check GPU Environment
    gpu_available = check_gpu_environment()
    
    # 2. Erstelle Test-Datensatz
    print("\n" + "=" * 60)
    print("📊 ERSTELLE TEST-DATENSATZ")
    print("=" * 60)
    
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # 3. Performance Vergleich
    train_xgboost_comparison(X_train, X_test, y_train, y_test)
    
    # 4. Zeige korrekte Konfiguration
    print("\n" + "=" * 60)
    print("✅ KORREKTE KONFIGURATION FÜR IHR TRAINING")
    print("=" * 60)
    
    if gpu_available:
        print("\n🚀 Verwenden Sie diese Parameter für GPU Training:")
        gpu_params = get_correct_xgboost_params(use_gpu=True)
        for key, value in gpu_params.items():
            print(f"   {key}: {value}")
    else:
        print("\n⚠️  GPU nicht verfügbar. Installationshinweise:")
        print("   1. pip uninstall xgboost")
        print("   2. pip install xgboost --upgrade")
        print("   ODER")
        print("   3. conda install -c conda-forge py-xgboost-gpu")
        
    print("\n" + "=" * 60)
    print("💡 WICHTIGE HINWEISE")
    print("=" * 60)
    print("""
    1. Stellen Sie sicher, dass CUDA korrekt installiert ist
    2. XGBoost muss mit GPU Support kompiliert sein
    3. Verwenden Sie 'tree_method': 'gpu_hist' für GPU
    4. Bei kleinen Datensätzen (<1000 Samples) kann CPU schneller sein
    5. GPU ist besonders effizient bei:
       - Großen Datensätzen (>10k Samples)
       - Vielen Features (>50)
       - Deep Trees (max_depth > 10)
    """)

if __name__ == "__main__":
    main()