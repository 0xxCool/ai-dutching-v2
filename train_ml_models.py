#!/usr/bin/env python3
"""
🚀 ML TRAINING PIPELINE - Neural Network & XGBoost
====================================================

Trainiert beide ML-Modelle mit Daten vom Hybrid-Scraper und speichert sie in der Model Registry.

Usage:
    python train_ml_models.py

Output:
    - models/neural_net_YYYYMMDD_HHMMSS.pth
    - models/xgboost_YYYYMMDD_HHMMSS.pkl
    - models/registry/model_registry.json

Hardware:
    - Optimiert für Nvidia RTX 3090 (24GB VRAM)
    - CPU-Fallback verfügbar
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import pickle
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Eigene Imports
from gpu_ml_models import (
    GPUNeuralNetworkPredictor,
    GPUXGBoostPredictor,
    GPUFeatureEngineer,
    GPUConfig
)
from continuous_training_system import ModelRegistry

# ==========================================================
# KONFIGURATION
# ==========================================================
class TrainingConfig:
    """Training Configuration"""

    # Daten
    DATABASE_FILE = "game_database_complete.csv"  # Vom Hybrid-Scraper!
    MIN_SAMPLES = 100  # Minimum Spiele für Training

    # Train/Validation/Test Split
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    # Feature Engineering
    FORM_WINDOW = 5  # Letzte N Spiele für Form

    # Neural Network
    NN_EPOCHS = 100
    NN_BATCH_SIZE = 64
    NN_LEARNING_RATE = 0.001
    NN_EARLY_STOPPING_PATIENCE = 15

    # XGBoost
    XGB_N_ESTIMATORS = 500
    XGB_MAX_DEPTH = 8
    XGB_LEARNING_RATE = 0.05
    XGB_EARLY_STOPPING_ROUNDS = 50

    # Output
    MODEL_DIR = Path("models")
    SAVE_MODELS = True
    VERBOSE = True


# ==========================================================
# DATA PREPARATION
# ==========================================================
class DataPreparator:
    """Bereitet Daten für ML-Training vor"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_config = GPUConfig()
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Lade Spieldaten vom Hybrid-Scraper"""
        print("\n📂 LADE DATEN...")
        print("=" * 70)

        if not Path(self.config.DATABASE_FILE).exists():
            raise FileNotFoundError(
                f"❌ Datenbank nicht gefunden: {self.config.DATABASE_FILE}\n"
                f"Bitte zuerst den Hybrid-Scraper ausführen:\n"
                f"  python sportmonks_hybrid_scraper_v3_FINAL.py"
            )

        df = pd.read_csv(self.config.DATABASE_FILE)
        df['date'] = pd.to_datetime(df['date'])

        # Sortiere nach Datum
        df = df.sort_values('date').reset_index(drop=True)

        print(f"✅ Geladen: {len(df)} Spiele")
        print(f"   Zeitraum: {df['date'].min()} bis {df['date'].max()}")
        print(f"   Ligen: {df['league'].nunique()}")

        # Prüfe erforderliche Spalten
        required_cols = ['date', 'home_team', 'away_team', 'home_score',
                         'away_score', 'home_xg', 'away_xg', 'league']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"❌ Fehlende Spalten: {missing}")

        # Entferne Zeilen mit fehlenden Werten
        df_clean = df.dropna(subset=required_cols)

        if len(df_clean) < len(df):
            print(f"   ⚠️ {len(df) - len(df_clean)} Zeilen mit fehlenden Werten entfernt")

        if len(df_clean) < self.config.MIN_SAMPLES:
            raise ValueError(
                f"❌ Zu wenig Daten: {len(df_clean)} < {self.config.MIN_SAMPLES}\n"
                f"Bitte mehr Daten scrapen!"
            )

        print(f"✅ Finale Daten: {len(df_clean)} Spiele\n")

        return df_clean

    def create_features_and_labels(self, df: pd.DataFrame) -> tuple:
        """Erstelle Features (X) und Labels (y)"""
        print("🔧 ERSTELLE FEATURES...")
        print("=" * 70)

        # Feature Engineer (GPU-beschleunigt)
        engineer = GPUFeatureEngineer(df, self.gpu_config.device)

        X_list = []
        y_list = []
        valid_indices = []

        # Iteriere durch alle Spiele (ab Spiel 6, damit wir Form berechnen können)
        for idx in tqdm(range(self.config.FORM_WINDOW, len(df)), desc="Feature Engineering"):
            row = df.iloc[idx]

            try:
                # Erstelle Features
                features = engineer.create_match_features(
                    row['home_team'],
                    row['away_team'],
                    row['date']
                )

                # Erstelle Label (1X2 Result)
                if row['home_score'] > row['away_score']:
                    label = 0  # Home Win
                elif row['home_score'] < row['away_score']:
                    label = 2  # Away Win
                else:
                    label = 1  # Draw

                X_list.append(features.cpu().numpy())
                y_list.append(label)
                valid_indices.append(idx)

            except Exception as e:
                if self.config.VERBOSE:
                    print(f"   ⚠️ Fehler bei Spiel {idx}: {e}")
                continue

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n✅ Features erstellt:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Klassen: {len(np.unique(y))}")

        # Klassenverteilung
        unique, counts = np.unique(y, return_counts=True)
        class_names = ['Home Win', 'Draw', 'Away Win']
        print(f"\n   Klassenverteilung:")
        for cls, count in zip(unique, counts):
            print(f"     {class_names[cls]}: {count} ({count/len(y)*100:.1f}%)")

        print()

        return X, y, valid_indices

    def split_data(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Split in Train/Validation/Test Sets"""
        print("✂️  SPLIT DATEN...")
        print("=" * 70)

        # Temporal Split (wichtig für Zeitreihendaten!)
        n_samples = len(X)
        train_end = int(n_samples * self.config.TRAIN_SIZE)
        val_end = int(n_samples * (self.config.TRAIN_SIZE + self.config.VAL_SIZE))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        print(f"✅ Split abgeschlossen:")
        print(f"   Training:   {len(X_train)} Samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"   Validation: {len(X_val)} Samples ({len(X_val)/n_samples*100:.1f}%)")
        print(f"   Test:       {len(X_test)} Samples ({len(X_test)/n_samples*100:.1f}%)")
        print()

        # Normalisiere Features
        print("🔄 Normalisiere Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        print("✅ Normalisierung abgeschlossen\n")

        return {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test
        }


# ==========================================================
# NEURAL NETWORK TRAINER
# ==========================================================
class NeuralNetworkTrainer:
    """Trainiert Neural Network"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_config = GPUConfig()
        self.model = GPUNeuralNetworkPredictor(
            input_size=20,  # 20 Features vom FeatureEngineer
            gpu_config=self.gpu_config
        )

    def train(self, data_dict: dict) -> dict:
        """Trainiere Neural Network"""
        print("\n🧠 TRAINIERE NEURAL NETWORK...")
        print("=" * 70)

        # Prepare Data
        X_train = torch.FloatTensor(data_dict['X_train']).to(self.gpu_config.device)
        y_train = torch.LongTensor(data_dict['y_train']).to(self.gpu_config.device)
        X_val = torch.FloatTensor(data_dict['X_val']).to(self.gpu_config.device)
        y_val = torch.LongTensor(data_dict['y_val']).to(self.gpu_config.device)

        # Train
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config.NN_EPOCHS,
            batch_size=self.config.NN_BATCH_SIZE,
            learning_rate=self.config.NN_LEARNING_RATE,
            early_stopping_patience=self.config.NN_EARLY_STOPPING_PATIENCE,
            verbose=self.config.VERBOSE
        )

        # Evaluate
        X_test = torch.FloatTensor(data_dict['X_test']).to(self.gpu_config.device)
        y_test = data_dict['y_test']

        y_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"\n✅ Neural Network Training abgeschlossen!")
        print(f"   Beste Validation Accuracy: {history['best_val_acc']:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")

        # Classification Report
        print(f"\n📊 Classification Report (Test Set):")
        print(classification_report(
            y_test, y_pred,
            target_names=['Home Win', 'Draw', 'Away Win'],
            digits=4
        ))

        return {
            'model': self.model,
            'val_acc': history['best_val_acc'],
            'test_acc': test_acc,
            'history': history
        }


# ==========================================================
# XGBOOST TRAINER
# ==========================================================
class XGBoostTrainer:
    """Trainiert XGBoost"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = GPUXGBoostPredictor(use_gpu=True)

    def train(self, data_dict: dict) -> dict:
        """Trainiere XGBoost"""
        print("\n🚀 TRAINIERE XGBOOST...")
        print("=" * 70)

        # Train
        self.model.train(
            data_dict['X_train'], data_dict['y_train'],
            data_dict['X_val'], data_dict['y_val'],
            n_estimators=self.config.XGB_N_ESTIMATORS,
            max_depth=self.config.XGB_MAX_DEPTH,
            learning_rate=self.config.XGB_LEARNING_RATE,
            early_stopping_rounds=self.config.XGB_EARLY_STOPPING_ROUNDS,
            verbose=self.config.VERBOSE
        )

        # Evaluate
        y_pred = self.model.predict(data_dict['X_test'])
        y_pred_proba = self.model.predict_proba(data_dict['X_test'])

        test_acc = accuracy_score(data_dict['y_test'], y_pred)

        # Validation Accuracy
        y_val_pred = self.model.predict(data_dict['X_val'])
        val_acc = accuracy_score(data_dict['y_val'], y_val_pred)

        print(f"\n✅ XGBoost Training abgeschlossen!")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")

        # Classification Report
        print(f"\n📊 Classification Report (Test Set):")
        print(classification_report(
            data_dict['y_test'], y_pred,
            target_names=['Home Win', 'Draw', 'Away Win'],
            digits=4
        ))

        return {
            'model': self.model,
            'val_acc': val_acc,
            'test_acc': test_acc
        }


# ==========================================================
# MODEL REGISTRY INTEGRATION
# ==========================================================
class ModelSaver:
    """Speichert Modelle in Registry"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

    def save_neural_network(self, result: dict, n_samples: int) -> str:
        """Speichere Neural Network"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"neural_net_{timestamp}"
        model_path = self.config.MODEL_DIR / f"{model_name}.pth"

        # Speichere Checkpoint
        result['model'].save_checkpoint(model_name)

        # Registriere in Registry
        version = self.registry.register_model(
            model_type='neural_net',
            model_path=str(model_path),
            training_samples=n_samples,
            validation_accuracy=result['val_acc']
        )

        # Setze als Champion wenn es das erste Modell ist oder besser als bisheriges
        current_champion = self.registry.get_champion('neural_net')
        if not current_champion or result['val_acc'] > current_champion.validation_accuracy:
            self.registry.set_champion('neural_net', version.version_id)
            print(f"   🏆 Neues Champion-Modell gesetzt!")

        print(f"   💾 Gespeichert: {model_path}")
        print(f"   📝 Registry ID: {version.version_id}")

        return str(model_path)

    def save_xgboost(self, result: dict, n_samples: int) -> str:
        """Speichere XGBoost"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_{timestamp}"
        model_path = self.config.MODEL_DIR / f"{model_name}.pkl"

        # Speichere Modell
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'].model, f)

        # Registriere in Registry
        version = self.registry.register_model(
            model_type='xgboost',
            model_path=str(model_path),
            training_samples=n_samples,
            validation_accuracy=result['val_acc']
        )

        # Setze als Champion wenn es das erste Modell ist oder besser als bisheriges
        current_champion = self.registry.get_champion('xgboost')
        if not current_champion or result['val_acc'] > current_champion.validation_accuracy:
            self.registry.set_champion('xgboost', version.version_id)
            print(f"   🏆 Neues Champion-Modell gesetzt!")

        print(f"   💾 Gespeichert: {model_path}")
        print(f"   📝 Registry ID: {version.version_id}")

        return str(model_path)


# ==========================================================
# MAIN TRAINING PIPELINE
# ==========================================================
def main():
    """Hauptfunktion"""

    print("\n" + "=" * 70)
    print("🚀 ML TRAINING PIPELINE - Neural Network & XGBoost")
    print("=" * 70)

    config = TrainingConfig()

    # 1. DATA PREPARATION
    preparator = DataPreparator(config)

    df = preparator.load_data()
    X, y, valid_indices = preparator.create_features_and_labels(df)
    data_dict = preparator.split_data(X, y)

    # 2. TRAIN NEURAL NETWORK
    nn_trainer = NeuralNetworkTrainer(config)
    nn_result = nn_trainer.train(data_dict)

    # 3. TRAIN XGBOOST
    xgb_trainer = XGBoostTrainer(config)
    xgb_result = xgb_trainer.train(data_dict)

    # 4. SAVE MODELS
    if config.SAVE_MODELS:
        print("\n💾 SPEICHERE MODELLE...")
        print("=" * 70)

        saver = ModelSaver(config)

        print("\n📦 Neural Network:")
        nn_path = saver.save_neural_network(nn_result, len(X))

        print("\n📦 XGBoost:")
        xgb_path = saver.save_xgboost(xgb_result, len(X))

    # 5. SUMMARY
    print("\n" + "=" * 70)
    print("✅ TRAINING ABGESCHLOSSEN!")
    print("=" * 70)

    print(f"\n📊 FINALE ERGEBNISSE:")
    print(f"\n   Neural Network:")
    print(f"     • Validation Accuracy: {nn_result['val_acc']:.4f}")
    print(f"     • Test Accuracy: {nn_result['test_acc']:.4f}")

    print(f"\n   XGBoost:")
    print(f"     • Validation Accuracy: {xgb_result['val_acc']:.4f}")
    print(f"     • Test Accuracy: {xgb_result['test_acc']:.4f}")

    print(f"\n📁 Modelle gespeichert in: {config.MODEL_DIR}/")
    print(f"📝 Registry: {config.MODEL_DIR}/registry/model_registry.json")

    print("\n🎯 NÄCHSTE SCHRITTE:")
    print("   1. Überprüfe die Modell-Performance")
    print("   2. Starte das Dutching-System:")
    print("      python sportmonks_dutching_system.py")
    print("   3. Bei Bedarf: Retrain mit mehr Daten")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training abgebrochen")
    except Exception as e:
        print(f"\n\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()
