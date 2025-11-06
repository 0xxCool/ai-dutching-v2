"""
🧠 AI DUTCHING SYSTEM - ML TRAINING PIPELINE V2
===============================================
Verbesserte Version mit Random Forest & LightGBM

Features:
- 4 Modelle: Neural Net, XGBoost, Random Forest, LightGBM
- GPU Support für alle kompatiblen Modelle
- Automatic Model Registry
- Early Stopping & Cross-Validation
- ✅ KOMPATIBEL MIT V4 FIXED SCRAPER!
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Daten
    DATABASE_PATH = "game_database_complete.csv"
    MODELS_DIR = "models"
    REGISTRY_FILE = "models/registry/model_registry.json"
    
    # Training
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Neural Network
    NN_HIDDEN_SIZES = [256, 128, 64]
    NN_DROPOUT = 0.3
    NN_BATCH_SIZE = 64
    NN_EPOCHS = 100
    NN_LR = 0.001
    NN_PATIENCE = 20
    
    # XGBoost
    XGB_N_ESTIMATORS = 200
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    XGB_EARLY_STOPPING = 50
    
    # Random Forest
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 10
    RF_N_JOBS = -1
    
    # LightGBM
    LGB_N_ESTIMATORS = 200
    LGB_MAX_DEPTH = 15
    LGB_LEARNING_RATE = 0.1
    LGB_EARLY_STOPPING = 50
    
    # GPU
    USE_GPU = True

# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout: float = 0.3):
        super(ImprovedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess_data(config: Config) -> Tuple:
    """Lädt und bereitet Daten vor - ✅ KOMPATIBEL MIT V4 FIXED!"""
    
    print("📂 LADE DATEN...")
    print("="*70)
    
    df = pd.read_csv(config.DATABASE_PATH)
    
    print(f"✅ Geladen: {len(df)} Spiele")
    
    # ✅ NEUE SPALTEN-NORMALISIERUNG FÜR V4 FIXED!
    # Konvertiere neue Spalten-Namen zu alten (für Kompatibilität)
    if 'home_team' in df.columns and 'HomeTeam' not in df.columns:
        print("   📝 Erkenne V4 FIXED Format - konvertiere Spalten...")
        df['HomeTeam'] = df['home_team']
        df['AwayTeam'] = df['away_team']
        df['FTHG'] = df['home_score']
        df['FTAG'] = df['away_score']
        
        # FTR aus Score berechnen
        def calculate_ftr(row):
            if row['home_score'] > row['away_score']:
                return 'H'
            elif row['home_score'] < row['away_score']:
                return 'A'
            else:
                return 'D'
        
        df['FTR'] = df.apply(calculate_ftr, axis=1)
        print("   ✅ Spalten konvertiert")
    
    # Datum konvertieren
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Zeitraum: {df['date'].min()} bis {df['date'].max()}")
    
    # Filter: Nur Spiele mit vollständigen Daten
    required_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
    df = df.dropna(subset=required_cols)
    
    print(f"✅ Finale Daten: {len(df)} Spiele\n")
    
    return df

def create_features(df: pd.DataFrame) -> Tuple:
    """Erstellt Features für ML-Modelle"""
    
    print("🔧 ERSTELLE FEATURES...")
    print("="*70)
    
    features_list = []
    labels_list = []
    
    # Sortiere nach Datum
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature Engineering mit rollendem Fenster
    for idx in tqdm(range(5, len(df)), desc="Feature Engineering"):
        match = df.iloc[idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Historische Daten (letzte 5 Spiele)
        past_data = df.iloc[:idx]
        
        # Home Team Stats
        home_matches = past_data[
            (past_data['HomeTeam'] == home_team) | 
            (past_data['AwayTeam'] == home_team)
        ].tail(5)
        
        home_goals_for = []
        home_goals_against = []
        home_wins = 0
        home_draws = 0
        home_losses = 0
        
        for _, m in home_matches.iterrows():
            if m['HomeTeam'] == home_team:
                home_goals_for.append(m['FTHG'])
                home_goals_against.append(m['FTAG'])
                if m['FTR'] == 'H':
                    home_wins += 1
                elif m['FTR'] == 'D':
                    home_draws += 1
                else:
                    home_losses += 1
            else:
                home_goals_for.append(m['FTAG'])
                home_goals_against.append(m['FTHG'])
                if m['FTR'] == 'A':
                    home_wins += 1
                elif m['FTR'] == 'D':
                    home_draws += 1
                else:
                    home_losses += 1
        
        # Away Team Stats
        away_matches = past_data[
            (past_data['HomeTeam'] == away_team) | 
            (past_data['AwayTeam'] == away_team)
        ].tail(5)
        
        away_goals_for = []
        away_goals_against = []
        away_wins = 0
        away_draws = 0
        away_losses = 0
        
        for _, m in away_matches.iterrows():
            if m['HomeTeam'] == away_team:
                away_goals_for.append(m['FTHG'])
                away_goals_against.append(m['FTAG'])
                if m['FTR'] == 'H':
                    away_wins += 1
                elif m['FTR'] == 'D':
                    away_draws += 1
                else:
                    away_losses += 1
            else:
                away_goals_for.append(m['FTAG'])
                away_goals_against.append(m['FTHG'])
                if m['FTR'] == 'A':
                    away_wins += 1
                elif m['FTR'] == 'D':
                    away_draws += 1
                else:
                    away_losses += 1
        
        # Skip if not enough data
        if len(home_goals_for) < 3 or len(away_goals_for) < 3:
            continue
        
        # Create feature vector
        features = [
            np.mean(home_goals_for),
            np.mean(home_goals_against),
            home_wins / max(len(home_matches), 1),
            home_draws / max(len(home_matches), 1),
            home_losses / max(len(home_matches), 1),
            np.mean(away_goals_for),
            np.mean(away_goals_against),
            away_wins / max(len(away_matches), 1),
            away_draws / max(len(away_matches), 1),
            away_losses / max(len(away_matches), 1),
            np.std(home_goals_for) if len(home_goals_for) > 1 else 0,
            np.std(away_goals_for) if len(away_goals_for) > 1 else 0,
            max(home_goals_for) if home_goals_for else 0,
            max(away_goals_for) if away_goals_for else 0,
            min(home_goals_against) if home_goals_against else 0,
            min(away_goals_against) if away_goals_against else 0,
            len(home_matches),
            len(away_matches),
            1 if len(home_matches) >= 5 else 0,
            1 if len(away_matches) >= 5 else 0
        ]
        
        # Label (0=Home, 1=Draw, 2=Away)
        if match['FTR'] == 'H':
            label = 0
        elif match['FTR'] == 'D':
            label = 1
        else:
            label = 2
        
        features_list.append(features)
        labels_list.append(label)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\n✅ Features erstellt:")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    return X, y

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_neural_network(X_train, y_train, X_val, y_val, config: Config):
    """Trainiert Neural Network"""
    
    print("\n🧠 TRAINIERE NEURAL NETWORK...")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
    print(f"Device: {device}")
    
    input_size = X_train.shape[1]
    output_size = 3
    
    model = ImprovedNeuralNetwork(
        input_size, config.NN_HIDDEN_SIZES, output_size, config.NN_DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.NN_LR)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NN_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.NN_EPOCHS} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.NN_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✅ Neural Network Training abgeschlossen!")
    
    return model, device

def train_xgboost(X_train, y_train, X_val, y_val, config: Config):
    """Trainiert XGBoost"""
    
    print("\n🌲 TRAINIERE XGBOOST...")
    print("="*70)
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': config.XGB_MAX_DEPTH,
        'learning_rate': config.XGB_LEARNING_RATE,
        'tree_method': 'gpu_hist' if config.USE_GPU else 'hist',
        'predictor': 'gpu_predictor' if config.USE_GPU else 'cpu_predictor',
        'eval_metric': 'mlogloss'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.XGB_N_ESTIMATORS,
        evals=evals,
        early_stopping_rounds=config.XGB_EARLY_STOPPING,
        verbose_eval=20
    )
    
    print(f"\n✅ XGBoost Training abgeschlossen!")
    
    return model

def train_random_forest(X_train, y_train, config: Config):
    """Trainiert Random Forest"""
    
    print("\n🌳 TRAINIERE RANDOM FOREST...")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        n_jobs=config.RF_N_JOBS,
        random_state=config.RANDOM_STATE,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print(f"\n✅ Random Forest Training abgeschlossen!")
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, config: Config):
    """Trainiert LightGBM"""
    
    print("\n💡 TRAINIERE LIGHTGBM...")
    print("="*70)
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': config.LGB_LEARNING_RATE,
        'feature_fraction': 0.9,
        'verbose': 1,
        'device': 'gpu' if config.USE_GPU else 'cpu'
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.LGB_N_ESTIMATORS,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(config.LGB_EARLY_STOPPING)]
    )
    
    print(f"\n✅ LightGBM Training abgeschlossen!")
    
    return model

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name: str, device=None):
    """Evaluiert ein Modell"""
    
    print(f"\n📊 EVALUIERE {model_name}...")
    print("="*70)
    
    if model_name == "Neural Network":
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_t)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    elif model_name == "XGBoost":
        # XGBoost braucht DMatrix!
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
    
    elif model_name == "LightGBM":
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
    
    else:
        # Random Forest
        predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}\n")
    
    print("Classification Report:")
    class_names = ['Home Win', 'Draw', 'Away Win']
    print(classification_report(y_test, predictions, target_names=class_names))
    
    return accuracy

# =============================================================================
# MODEL REGISTRY
# =============================================================================

def save_models(models: Dict, scaler, config: Config):
    """Speichert alle Modelle"""
    
    print("\n💾 SPEICHERE MODELLE...")
    print("="*70)
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.REGISTRY_FILE), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(config.REGISTRY_FILE):
        with open(config.REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': []}
    
    saved_models = {}
    
    for model_type, (model, accuracy, device) in models.items():
        
        model_id = f"{model_type.lower().replace(' ', '_')}_{timestamp}"
        model_path = os.path.join(config.MODELS_DIR, f"{model_id}.pkl")
        
        if model_type == "Neural Network":
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': model.network[0].in_features,
                    'hidden_sizes': config.NN_HIDDEN_SIZES,
                    'output_size': 3,
                    'dropout': config.NN_DROPOUT
                }
            }, model_path.replace('.pkl', '.pth'))
            
            saved_models[model_type] = model_path.replace('.pkl', '.pth')
        
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            saved_models[model_type] = model_path
        
        registry['models'].append({
            'id': model_id,
            'type': model_type,
            'path': saved_models[model_type],
            'accuracy': accuracy,
            'timestamp': timestamp,
            'active': True
        })
        
        print(f"✅ {model_type}: {model_id}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Path: {saved_models[model_type]}\n")
    
    scaler_path = os.path.join(config.MODELS_DIR, f"scaler_{timestamp}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    registry['scaler'] = scaler_path
    
    with open(config.REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Registry gespeichert: {config.REGISTRY_FILE}")

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Haupttraining-Pipeline"""
    
    print("="*70)
    print("🚀 ML TRAINING PIPELINE V2 - KOMPATIBEL MIT V4 FIXED")
    print("="*70)
    print()
    
    config = Config()
    
    df = load_and_preprocess_data(config)
    
    X, y = create_features(df)
    
    print("\n✂️  SPLIT DATEN...")
    print("="*70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE/(1-config.TEST_SIZE), 
        random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"✅ Split abgeschlossen:")
    print(f"   Training:   {len(X_train)} Samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} Samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} Samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\n🔄 Normalisiere Features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print("✅ Normalisierung abgeschlossen")
    
    models = {}
    
    nn_model, device = train_neural_network(X_train, y_train, X_val, y_val, config)
    nn_acc = evaluate_model(nn_model, X_test, y_test, "Neural Network", device)
    models["Neural Network"] = (nn_model, nn_acc, device)
    
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config)
    xgb_acc = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    models["XGBoost"] = (xgb_model, xgb_acc, None)
    
    rf_model = train_random_forest(X_train, y_train, config)
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    models["Random Forest"] = (rf_model, rf_acc, None)
    
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, config)
    lgb_acc = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    models["LightGBM"] = (lgb_model, lgb_acc, None)
    
    save_models(models, scaler, config)
    
    print("\n" + "="*70)
    print("✅ TRAINING ABGESCHLOSSEN!")
    print("="*70)
    print("\n📊 FINALE ERGEBNISSE:\n")
    
    for model_type, (_, accuracy, _) in models.items():
        print(f"   {model_type:20s} Test Acc: {accuracy:.4f}")
    
    print(f"\n📁 Modelle gespeichert in: {config.MODELS_DIR}/")
    print(f"📝 Registry: {config.REGISTRY_FILE}")
    
    print("\n🎯 NÄCHSTE SCHRITTE:")
    print("   1. Starte Dutching System:")
    print("      python sportmonks_dutching_system_v2.py")
    print()
    print("   2. Starte Correct Score System:")
    print("      python sportmonks_correct_score_system_v2.py")
    print()
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
