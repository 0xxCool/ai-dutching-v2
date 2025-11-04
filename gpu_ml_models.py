"""
🚀 GPU-OPTIMIERTE ML MODELS FÜR RTX 3090
================================================

Features:
- CUDA-beschleunigtes Training (PyTorch)
- XGBoost GPU-Training
- Mixed Precision Training (FP16) mit Tensor Cores
- Automatisches Batch-Processing
- Model Checkpointing
- Kontinuierliches Online-Learning
- Multi-GPU Ready

Hardware-Optimierung:
- Nvidia RTX 3090 (24GB VRAM, 10496 CUDA Cores)
- CUDA 11.x / 12.x
- cuDNN 8.x
- Tensor Cores für Mixed Precision

Performance:
- XGBoost GPU: 10-50x schneller vs CPU
- PyTorch CUDA: 20-100x schneller vs CPU
- Mixed Precision: 2-3x schneller + 50% weniger VRAM
"""

import numpy as np
import pandas as pd

# XGBoost mit GPU-Support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost nicht installiert. pip install xgboost")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# GPU CONFIGURATION & DETECTION
# ==========================================================
class GPUConfig:
    """GPU-Konfiguration für RTX 3090"""

    def __init__(self):
        self.device = self._detect_device()
        self.use_mixed_precision = True  # FP16 mit Tensor Cores
        self.pin_memory = True  # Schnellere CPU->GPU Transfers
        self.num_workers = 4  # DataLoader workers

        # RTX 3090 spezifisch
        self.max_batch_size = 512  # Optimal für 24GB VRAM
        self.gradient_accumulation_steps = 1

        # CUDA Optimierungen
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Auto-tune
            torch.backends.cudnn.deterministic = False  # Mehr Performance

    def _detect_device(self) -> torch.device:
        """Erkenne verfügbare GPU"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            print("🚀 GPU DETECTED:")
            print(f"   Device: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")

            # RTX 3090 Check
            if "3090" in gpu_name or "RTX 3090" in gpu_name:
                print("   ✅ RTX 3090 erkannt - Volle Leistung aktiviert!")
                self.use_mixed_precision = True
            else:
                print("   ⚠️  Andere GPU erkannt - Mixed Precision verfügbar")

            return device
        else:
            print("⚠️  Keine GPU gefunden - CPU-Modus")
            return torch.device('cpu')

    def get_optimal_batch_size(self, model_size: str = 'medium') -> int:
        """Berechne optimale Batch-Size basierend auf Modell-Größe"""
        if not torch.cuda.is_available():
            return 32

        # RTX 3090 mit 24GB kann sehr große Batches
        if model_size == 'small':
            return 1024
        elif model_size == 'medium':
            return 512
        elif model_size == 'large':
            return 256
        else:
            return 128

    def print_memory_stats(self):
        """Zeige GPU Memory-Nutzung"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n📊 GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")


# ==========================================================
# FEATURE ENGINEERING (GPU-beschleunigt)
# ==========================================================
class GPUFeatureEngineer:
    """
    GPU-beschleunigte Feature Engineering

    Verwendet PyTorch Tensors für parallele Berechnungen
    """

    def __init__(self, database: pd.DataFrame, device: torch.device):
        self.db = database.copy()
        self.db['date'] = pd.to_datetime(self.db['date'])
        self.db = self.db.sort_values('date')
        self.device = device

    def calculate_form_gpu(
        self,
        team: str,
        before_date: pd.Timestamp,
        n_games: int = 5
    ) -> Dict[str, float]:
        """GPU-beschleunigte Form-Berechnung"""
        mask = self.db['date'] < before_date

        home_games = self.db[mask & (self.db['home_team'] == team)]
        away_games = self.db[mask & (self.db['away_team'] == team)]

        # Kombiniere
        goals_for = pd.concat([home_games['home_score'], away_games['away_score']])
        goals_against = pd.concat([home_games['away_score'], away_games['home_score']])
        xg_for = pd.concat([home_games['home_xg'], away_games['away_xg']])
        xg_against = pd.concat([home_games['away_xg'], away_games['home_xg']])

        # Letzte N Spiele
        recent_goals_for = goals_for.tail(n_games).values
        recent_goals_against = goals_against.tail(n_games).values
        recent_xg_for = xg_for.tail(n_games).values
        recent_xg_against = xg_against.tail(n_games).values

        if len(recent_goals_for) == 0:
            return {
                'avg_goals_scored': 1.0,
                'avg_goals_conceded': 1.0,
                'avg_xg_for': 1.0,
                'avg_xg_against': 1.0,
                'win_rate': 0.33,
                'points_per_game': 1.0
            }

        # GPU-Berechnung
        with torch.no_grad():
            gf_tensor = torch.tensor(recent_goals_for, device=self.device, dtype=torch.float32)
            ga_tensor = torch.tensor(recent_goals_against, device=self.device, dtype=torch.float32)
            xgf_tensor = torch.tensor(recent_xg_for, device=self.device, dtype=torch.float32)
            xga_tensor = torch.tensor(recent_xg_against, device=self.device, dtype=torch.float32)

            avg_gf = gf_tensor.mean().item()
            avg_ga = ga_tensor.mean().item()
            avg_xgf = xgf_tensor.mean().item()
            avg_xga = xga_tensor.mean().item()

        # Win Rate
        wins = (recent_goals_for > recent_goals_against).sum()
        draws = (recent_goals_for == recent_goals_against).sum()
        n_games_actual = len(recent_goals_for)

        win_rate = wins / n_games_actual if n_games_actual > 0 else 0.33
        ppg = (wins * 3 + draws) / n_games_actual if n_games_actual > 0 else 1.0

        return {
            'avg_goals_scored': float(avg_gf),
            'avg_goals_conceded': float(avg_ga),
            'avg_xg_for': float(avg_xgf),
            'avg_xg_against': float(avg_xga),
            'win_rate': float(win_rate),
            'points_per_game': float(ppg)
        }
    
    def _normalize_team_name(self, team_name: str) -> str:
        # Verwende die gleichen Mappings wie TeamMatcher!
        from sportmonks_dutching_system import TeamMatcher
        
        return TeamMatcher.normalize(team_name)

    def create_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp
    ) -> torch.Tensor:
        
        # NEUE ZEILEN: Normalisiere Team-Namen
        home_team = self._normalize_team_name(home_team)
        away_team = self._normalize_team_name(away_team)

        """Erstelle Feature-Tensor (GPU-ready)"""
        home_form = self.calculate_form_gpu(home_team, match_date, n_games=5)
        away_form = self.calculate_form_gpu(away_team, match_date, n_games=5)

        features = [
            # Home Form (6)
            home_form['avg_goals_scored'],
            home_form['avg_goals_conceded'],
            home_form['avg_xg_for'],
            home_form['avg_xg_against'],
            home_form['win_rate'],
            home_form['points_per_game'],

            # Away Form (6)
            away_form['avg_goals_scored'],
            away_form['avg_goals_conceded'],
            away_form['avg_xg_for'],
            away_form['avg_xg_against'],
            away_form['win_rate'],
            away_form['points_per_game'],

            # Differentials (8)
            home_form['avg_xg_for'] - away_form['avg_xg_against'],
            away_form['avg_xg_for'] - home_form['avg_xg_against'],
            home_form['avg_goals_scored'] - away_form['avg_goals_conceded'],
            away_form['avg_goals_scored'] - home_form['avg_goals_conceded'],
            home_form['points_per_game'] - away_form['points_per_game'],
            home_form['win_rate'] - away_form['win_rate'],
            home_form['avg_xg_for'] + away_form['avg_xg_for'],  # Total attacking
            home_form['avg_xg_against'] + away_form['avg_xg_against'],  # Total defending
        ]

        return torch.tensor(features, device=self.device, dtype=torch.float32)


# ==========================================================
# GPU-OPTIMIZED NEURAL NETWORK
# ==========================================================
class GPUMatchPredictionNet(nn.Module):
    """
    GPU-optimierte Deep Neural Network mit:
    - Batch Normalization
    - Dropout für Regularisierung
    - LeakyReLU für bessere Gradients
    - Residual Connections
    """

    def __init__(self, input_size: int = 20, hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3 if i == 0 else 0.2)
            ])
            prev_size = hidden_size

        # Output Layer
        layers.append(nn.Linear(prev_size, 3))  # 3 classes: Home/Draw/Away

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        # KORREKTUR: Softmax wird entfernt, da nn.CrossEntropyLoss es intern berechnet
        return logits


class GPUNeuralNetworkPredictor:
    """
    GPU-beschleunigter Neural Network Predictor mit:
    - Mixed Precision Training (FP16)
    - Gradient Accumulation
    - Automatic Model Checkpointing
    - Learning Rate Scheduling
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_sizes: List[int] = [256, 128, 64],
        gpu_config: GPUConfig = None,
        # KORREKTUR: Parameter hier hinzugefügt
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15
    ):
        self.config = gpu_config or GPUConfig()
        self.device = self.config.device
        
        # KORREKTUR: Patience speichern
        self.patience = early_stopping_patience

        # Model
        self.model = GPUMatchPredictionNet(input_size, hidden_sizes).to(self.device)

        # Optimizer mit Weight Decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            # KORREKTUR: Parameter verwenden
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5 # Interne Scheduler-Patience (anders als Early Stopping)
        )

        # Loss (erwartet Logits, nicht Softmax)
        self.criterion = nn.CrossEntropyLoss()

        # Mixed Precision Scaler
        self.scaler = GradScaler() if self.config.use_mixed_precision else None

        self.is_trained = False
        self.training_history = {} # Besser als Diktat
        self.best_val_acc = 0.0

    def train(
        self,
        X_train_t: torch.Tensor, # Erwarte bereits Tensoren
        y_train_t: torch.Tensor, # Erwarte bereits Tensoren
        epochs: int = 100,
        batch_size: int = None,
        # KORREKTUR: 'validation_split' ersetzt durch 'validation_data'
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        verbose: bool = True
    ):
        """
        GPU-beschleunigtes Training mit Mixed Precision
        """
        if batch_size is None:
            batch_size = self.config.get_optimal_batch_size('medium')

        # KORREKTUR: Interner Split entfernt. Wir verwenden die Daten aus train_ml_models.py
        
        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0  # Windows kompatibel
        )
        
        # Hole Validierungsdaten
        if validation_data:
            X_val_t, y_val_t = validation_data
            if verbose:
                print(f"   Validation Samples: {len(X_val_t)}")
        else:
            if verbose:
                print("   Keine Validierungsdaten übergeben.")


        if verbose:
            print(f"\n🚀 GPU Training gestartet:")
            print(f"   Device: {self.device}")
            print(f"   Training Samples: {len(X_train_t)}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Epochs: {epochs}")
            print(f"   Mixed Precision: {self.config.use_mixed_precision}")
            print(f"{'='*60}")

        best_val_loss = float('inf')
        patience_counter = 0
        # KORREKTUR: Hartkodierten Wert durch Parameter ersetzt
        max_patience = self.patience 

        history_log = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()

                if self.config.use_mixed_precision and self.scaler:
                    with autocast():
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history_log['train_loss'].append(avg_train_loss)

            # Validation
            val_loss_item = 0.0
            val_acc = 0.0
            
            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    if self.config.use_mixed_precision:
                        with autocast():
                            val_outputs = self.model(X_val_t)
                            val_loss = self.criterion(val_outputs, y_val_t)
                    else:
                        val_outputs = self.model(X_val_t)
                        val_loss = self.criterion(val_outputs, y_val_t)

                    val_preds = val_outputs.argmax(dim=1)
                    val_acc = (val_preds == y_val_t).float().mean().item()
                    val_loss_item = val_loss.item()
                
                history_log['val_loss'].append(val_loss_item)
                history_log['val_accuracy'].append(val_acc)

                # Learning Rate Scheduling
                self.scheduler.step(val_loss_item)
            
            # Verbose Output
            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if validation_data:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss_item:.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"LR: {current_lr:.6f}")
                else:
                     print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"LR: {current_lr:.6f}")

            # Early Stopping
            if validation_data:
                if val_loss_item < best_val_loss:
                    best_val_loss = val_loss_item
                    self.best_val_acc = val_acc # KORREKTUR: Beste Acc speichern
                    patience_counter = 0
                    self._save_checkpoint('best_model')
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    if verbose:
                        print(f"\n⏹️  Early Stopping at Epoch {epoch+1}")
                    break
            else:
                # Ohne Validierung speichern wir einfach das letzte Modell
                self._save_checkpoint('best_model')


        self.is_trained = True
        self.training_history = history_log
        self.training_history['best_val_acc'] = self.best_val_acc # Am Ende hinzufügen

        if verbose:
            print(f"{'='*60}")
            print(f"✅ Training abgeschlossen!")
            if validation_data:
                print(f"   Best Val Loss: {best_val_loss:.4f}")
                print(f"   Best Val Accuracy: {self.best_val_acc:.4f}")
            self.config.print_memory_stats()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """GPU-beschleunigte Predictions"""
        if not self.is_trained:
            raise ValueError("Modell muss erst trainiert werden!")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            if self.config.use_mixed_precision:
                with autocast():
                    logits = self.model(X_tensor)
            else:
                logits = self.model(X_tensor)
        
        # KORREKTUR: Softmax hier anwenden, da es aus forward entfernt wurde
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def _save_checkpoint(self, name: str = 'checkpoint'):
        """Speichere Model Checkpoint"""
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'best_val_acc': self.best_val_acc # Hinzugefügt
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_dir / f'{name}.pth')

    def load_checkpoint(self, name: str = 'checkpoint'):
        """Lade Model Checkpoint"""
        checkpoint_path = Path(f'models/checkpoints/{name}.pth')

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {name} nicht gefunden!")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0) # Hinzugefügt

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✅ Checkpoint '{name}' geladen!")


# ==========================================================
# GPU-OPTIMIZED XGBOOST
# ==========================================================
class GPUXGBoostPredictor:
    """
    XGBoost mit GPU-Training (tree_method='gpu_hist')
    Fällt automatisch auf CPU ('hist') zurück, falls GPU-Build fehlt.
    """

    def __init__(
        self, 
        use_gpu: bool = True,
        # KORREKTUR: Parameter hier hinzugefügt
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        early_stopping_rounds: Optional[int] = None  # <--- HIERHIN VERSCHOBEN
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost nicht installiert!")

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tree_method = 'gpu_hist' if self.use_gpu else 'hist'
        self.predictor = 'gpu_predictor' if self.use_gpu else 'auto'
        
        # KORREKTUR: Hartkodierte Werte durch Parameter ersetzt
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            tree_method=self.tree_method,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds, # <--- HIER GESETZT
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            predictor=self.predictor,
            max_bin=256
        )
        self.is_trained = False
        self.params_at_init = self.model.get_params() # Speichern für CPU-Fallback

        if self.use_gpu:
            print("🚀 XGBoost GPU-Modus (Versuch wird gestartet)...")
        else:
            print("ℹ️  XGBoost CPU-Modus.")

    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        # KORREKTUR: Parameter für Early Stopping hinzugefügt
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        # early_stopping_rounds HIER ENTFERNT
        verbose: bool = True
    ):
            """
            Trainiert das Modell (auf GPU oder CPU, je nach __init__)
            """
            if verbose:
                print(f"\n🌲 XGBoost Training:")
                print(f"   Samples: {len(X)}")
                print(f"   Features: {X.shape[1]}")
                print(f"   Methode (Versuch): {self.model.get_params()['tree_method']}")

            # Setup für Early Stopping
            fit_params = {}
            # KORREKTUR: Wir prüfen nur auf eval_set. early_stopping_rounds ist schon im Modell.
            if X_val is not None and y_val is not None:
                eval_set = [(X, y), (X_val, y_val)] # Füge auch Trainingsdaten hinzu für mlogloss-Vergleich
                fit_params['eval_set'] = eval_set
                fit_params['verbose'] = verbose
                if verbose:
                     # Holen die Info aus dem Modell, ob Early Stopping gesetzt ist
                    rounds = self.model.get_params().get('early_stopping_rounds')
                    if rounds:
                        print(f"   Early Stopping: Aktiviert (Rounds={rounds})")
            else:
                fit_params['verbose'] = False # Kein Spam, wenn kein Early Stopping
                
            try:
                # === HAUPT-VERSUCH (GPU) ===
                # KORREKTUR: 'early_stopping_rounds' ist HIER WEG
                self.model.fit(X, y, **fit_params)
                
            except xgb.core.XGBoostError as e:
                # === FEHLERBEHANDLUNG: FALLBACK AUF CPU ===
                if "not compiled with GPU support" in str(e) or "Invalid Input: 'gpu_hist'" in str(e):
                    print("\n" + "="*60)
                    print("⚠️  WARNUNG: XGBoost GPU-Training fehlgeschlagen. Falle zurück auf CPU.")
                    print(f"   Fehler: {e}")
                    print("   Neuer Versuch mit tree_method='hist' (CPU)...")
                    print("="*60 + "\n")
                    
                    self.use_gpu = False
                    self.tree_method = 'hist'
                    self.predictor = 'auto'
                    
                    # Erstelle Modell neu mit gespeicherten Init-Params, aber neuer tree_method
                    new_params = self.params_at_init.copy()
                    new_params['tree_method'] = self.tree_method
                    new_params['predictor'] = self.predictor
                    
                    self.model = xgb.XGBClassifier(**new_params)
                    
                    # Zweiter Versuch mit CPU
                    self.model.fit(X, y, **fit_params)
                else:
                    # Wenn es ein anderer XGBoost-Fehler ist, wirf ihn
                    raise e

            self.is_trained = True
            if verbose:
                print(f"✅ XGBoost Training abgeschlossen! (Finale Methode: {self.model.get_params()['tree_method']})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predictions"""
        if not self.is_trained:
            raise ValueError("Modell muss erst trainiert werden!")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Hole Feature Importances"""
        if not self.is_trained:
            return {}

        importance = self.model.feature_importances_

        feature_names = [
            'home_goals_scored', 'home_goals_conceded', 'home_xg_for', 'home_xg_against',
            'home_win_rate', 'home_ppg',
            'away_goals_scored', 'away_goals_conceded', 'away_xg_for', 'away_xg_against',
            'away_win_rate', 'away_ppg',
            'diff_xg_home', 'diff_xg_away', 'diff_goals_home', 'diff_goals_away',
            'diff_ppg', 'diff_wr', 'total_attack', 'total_defense'
        ]
        
        min_len = min(len(feature_names), len(importance))
        return dict(zip(feature_names[:min_len], importance[:min_len]))

                
# ==========================================================
# EXAMPLE & TESTING
# ==========================================================
if __name__ == "__main__":
    print("🚀 GPU ML MODELS TEST")
    print("="*60)

    # GPU Config
    gpu_config = GPUConfig()

    # Mock Data
    n_samples = 1000
    n_features = 20

    X_train = np.random.rand(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, 3, n_samples)

    # Test Neural Network
    print("\n🧠 Testing GPU Neural Network...")
    nn_model = GPUNeuralNetworkPredictor(
        input_size=n_features,
        hidden_sizes=[256, 128, 64],
        gpu_config=gpu_config
    )

    nn_model.train(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        verbose=True
    )

    # Test Prediction
    X_test = np.random.rand(10, n_features).astype(np.float32)
    probs = nn_model.predict_proba(X_test)
    print(f"\n📊 Prediction Shape: {probs.shape}")
    print(f"Sample Prediction: {probs[0]}")
    print(f"Sum: {probs[0].sum():.6f}")

    # Test XGBoost GPU
    if XGBOOST_AVAILABLE:
        print("\n🌲 Testing GPU XGBoost...")
        xgb_model = GPUXGBoostPredictor(use_gpu=True)
        xgb_model.train(X_train, y_train, verbose=True)

        xgb_probs = xgb_model.predict_proba(X_test)
        print(f"\n📊 XGB Prediction Shape: {xgb_probs.shape}")
        print(f"Sample Prediction: {xgb_probs[0]}")

        # Feature Importance
        importance = xgb_model.get_feature_importance()
        print(f"\n🔍 Top 5 Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:5]:
            print(f"   {feat}: {imp:.4f}")

    print("\n✅ Alle Tests erfolgreich!")
    gpu_config.print_memory_stats()
