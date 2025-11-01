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
