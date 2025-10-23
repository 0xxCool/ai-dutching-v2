"""
🔄 KONTINUIERLICHES TRAINING SYSTEM
====================================

Automatisches Retraining der ML-Modelle mit neuen Daten

Features:
- Automatische Erkennung neuer Spieldaten
- Scheduled Retraining (täglich, wöchentlich)
- Model Versioning & Checkpointing
- A/B Testing neuer Modelle
- Performance-Tracking über Zeit
- Rollback bei schlechteren Modellen
- Online Learning für kontinuierliche Verbesserung

Workflow:
1. Neue Daten → Database
2. System erkennt neue Daten
3. Retrain auf updated dataset
4. A/B Test: Neues vs Altes Modell
5. Deploy besseres Modell
6. Track Performance
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import pickle
import time
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Eigene Imports
try:
    from gpu_ml_models import (
        GPUNeuralNetworkPredictor,
        GPUXGBoostPredictor,
        GPUFeatureEngineer,
        GPUConfig
    )
    from optimized_poisson_model import VectorizedPoissonModel
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU Models nicht verfügbar - Continuous Training deaktiviert")


# ==========================================================
# MODEL VERSION MANAGEMENT
# ==========================================================
@dataclass
class ModelVersion:
    """Model Version Metadata"""
    version_id: str
    model_type: str  # 'neural_net', 'xgboost', 'poisson'
    created_at: datetime
    training_samples: int
    validation_accuracy: float
    test_accuracy: float = 0.0

    # Performance Metrics
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    num_bets: int = 0

    # Status
    is_active: bool = False
    is_champion: bool = False  # Bestes Modell

    # Paths
    model_path: str = ""
    config_path: str = ""

    def to_dict(self) -> Dict:
        """Konvertiere zu Dict (für JSON)"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Erstelle aus Dict"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """
    Zentrale Registry für alle Modell-Versionen

    Speichert:
    - Alle trainierten Modelle
    - Performance-Metriken
    - A/B-Test Ergebnisse
    - Champion-Modell (bestes Modell)
    """

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "model_registry.json"
        self.versions: Dict[str, ModelVersion] = {}

        self._load_registry()

    def _load_registry(self):
        """Lade Registry aus File"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)

            self.versions = {
                vid: ModelVersion.from_dict(vdata)
                for vid, vdata in data.items()
            }

            print(f"✅ Registry geladen: {len(self.versions)} Versionen")
        else:
            print("📝 Neue Registry erstellt")

    def _save_registry(self):
        """Speichere Registry"""
        data = {
            vid: version.to_dict()
            for vid, version in self.versions.items()
        }

        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_type: str,
        model_path: str,
        training_samples: int,
        validation_accuracy: float
    ) -> ModelVersion:
        """Registriere neues Modell"""
        # Generate Version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_type}_{timestamp}"

        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now(),
            training_samples=training_samples,
            validation_accuracy=validation_accuracy,
            model_path=model_path,
            is_active=True
        )

        self.versions[version_id] = version
        self._save_registry()

        print(f"✅ Modell registriert: {version_id}")
        return version

    def get_champion(self, model_type: str) -> Optional[ModelVersion]:
        """Hole aktuelles Champion-Modell"""
        champions = [
            v for v in self.versions.values()
            if v.model_type == model_type and v.is_champion
        ]

        return champions[0] if champions else None

    def set_champion(self, version_id: str):
        """Setze neues Champion-Modell"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} nicht gefunden!")

        version = self.versions[version_id]

        # Entferne alte Champions
        for v in self.versions.values():
            if v.model_type == version.model_type:
                v.is_champion = False

        # Setze neuen Champion
        version.is_champion = True
        self._save_registry()

        print(f"🏆 Neuer Champion: {version_id}")

    def update_performance(
        self,
        version_id: str,
        roi: float,
        sharpe_ratio: float,
        win_rate: float,
        num_bets: int
    ):
        """Update Performance-Metriken"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} nicht gefunden!")

        version = self.versions[version_id]
        version.roi = roi
        version.sharpe_ratio = sharpe_ratio
        version.win_rate = win_rate
        version.num_bets = num_bets

        self._save_registry()

    def get_version_history(self, model_type: str) -> List[ModelVersion]:
        """Hole Version-Historie"""
        versions = [
            v for v in self.versions.values()
            if v.model_type == model_type
        ]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)


# ==========================================================
# CONTINUOUS TRAINING ENGINE
# ==========================================================
class ContinuousTrainingEngine:
    """
    Engine für kontinuierliches Training

    Workflow:
    1. Check for new data
    2. Retrain models
    3. Validate performance
    4. A/B test vs champion
    5. Deploy if better
    """

    def __init__(
        self,
        database_path: str = "game_database_sportmonks.csv",
        min_new_samples: int = 50,  # Min neue Samples für Retrain
        retrain_schedule: str = "daily"  # 'daily', 'weekly', 'manual'
    ):
        self.database_path = database_path
        self.min_new_samples = min_new_samples
        self.retrain_schedule = retrain_schedule

        # Registry
        self.registry = ModelRegistry()

        # GPU Config
        self.gpu_config = GPUConfig() if GPU_AVAILABLE else None

        # State
        self.last_training_date = self._load_last_training_date()
        self.db = None

    def _load_last_training_date(self) -> datetime:
        """Lade letztes Training-Datum"""
        state_file = Path("models/training_state.json")

        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)
            return datetime.fromisoformat(data['last_training_date'])
        else:
            return datetime.now() - timedelta(days=365)  # Vor 1 Jahr

    def _save_last_training_date(self):
        """Speichere letztes Training-Datum"""
        state_file = Path("models/training_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, 'w') as f:
            json.dump({
                'last_training_date': datetime.now().isoformat()
            }, f)

    def check_for_new_data(self) -> Tuple[bool, int]:
        """
        Prüfe ob neue Daten verfügbar sind

        Returns:
            (has_new_data, num_new_samples)
        """
        try:
            self.db = pd.read_csv(self.database_path)
            self.db['date'] = pd.to_datetime(self.db['date'])
        except FileNotFoundError:
            print(f"⚠️  Database nicht gefunden: {self.database_path}")
            return False, 0

        # Zähle neue Samples seit letztem Training
        new_samples = self.db[self.db['date'] > self.last_training_date]
        num_new = len(new_samples)

        has_new_data = num_new >= self.min_new_samples

        if has_new_data:
            print(f"✅ {num_new} neue Samples gefunden!")
        else:
            print(f"⏸️  Nur {num_new} neue Samples (min: {self.min_new_samples})")

        return has_new_data, num_new

    def should_retrain(self) -> bool:
        """Entscheide ob Retrain notwendig"""
        # Check Schedule
        if self.retrain_schedule == "daily":
            days_since_training = (datetime.now() - self.last_training_date).days
            if days_since_training < 1:
                print(f"⏸️  Letztes Training vor {days_since_training} Tagen")
                return False

        elif self.retrain_schedule == "weekly":
            days_since_training = (datetime.now() - self.last_training_date).days
            if days_since_training < 7:
                print(f"⏸️  Letztes Training vor {days_since_training} Tagen")
                return False

        # Check für neue Daten
        has_new_data, num_new = self.check_for_new_data()

        return has_new_data

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bereite Training-Daten vor"""
        if self.db is None:
            self.db = pd.read_csv(self.database_path)
            self.db['date'] = pd.to_datetime(self.db['date'])

        print(f"📊 Preparing Training Data...")
        print(f"   Total Samples: {len(self.db)}")

        # Feature Engineering
        engineer = GPUFeatureEngineer(self.db, self.gpu_config.device)

        X_list = []
        y_list = []

        for idx, row in self.db.iterrows():
            try:
                features = engineer.create_match_features(
                    row['home_team'],
                    row['away_team'],
                    row['date']
                )

                # Label: 0=Home, 1=Draw, 2=Away
                if row['home_score'] > row['away_score']:
                    label = 0
                elif row['home_score'] == row['away_score']:
                    label = 1
                else:
                    label = 2

                X_list.append(features.cpu().numpy())
                y_list.append(label)

            except Exception as e:
                continue

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        print(f"   Features: {X.shape}")
        print(f"   Labels: {y.shape}")

        return X, y

    def train_neural_network(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100
    ) -> Tuple[GPUNeuralNetworkPredictor, float]:
        """Trainiere neues Neural Network"""
        print("\n🧠 Training Neural Network...")

        model = GPUNeuralNetworkPredictor(
            input_size=X.shape[1],
            hidden_sizes=[256, 128, 64],
            gpu_config=self.gpu_config
        )

        model.train(
            X, y,
            epochs=epochs,
            batch_size=512,  # Optimal für RTX 3090
            validation_split=0.2,
            verbose=True
        )

        # Save Model
        model_dir = Path("models/neural_networks")
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"nn_model_{timestamp}.pth"

        model._save_checkpoint(str(model_path.stem))

        # Validation Accuracy
        val_acc = model.training_history[-1]['val_accuracy']

        return model, val_acc

    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[GPUXGBoostPredictor, float]:
        """Trainiere neues XGBoost Modell"""
        print("\n🌲 Training XGBoost...")

        model = GPUXGBoostPredictor(use_gpu=True)

        # Train/Val Split
        val_size = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.train(X_train, y_train, verbose=True)

        # Validation Accuracy
        val_preds = model.predict_proba(X_val).argmax(axis=1)
        val_acc = (val_preds == y_val).mean()

        print(f"✅ XGBoost Val Accuracy: {val_acc:.4f}")

        # Save Model
        model_dir = Path("models/xgboost")
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"xgb_model_{timestamp}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model.model, f)

        return model, val_acc

    def ab_test_models(
        self,
        new_model_id: str,
        champion_id: str,
        test_data: pd.DataFrame
    ) -> Dict:
        """
        A/B Test: Neues Modell vs Champion

        Returns:
            Dict mit Vergleichs-Metriken
        """
        print(f"\n🔬 A/B Test: {new_model_id} vs {champion_id}")
        print("="*60)

        # TODO: Implementiere A/B Testing auf test_data
        # Vergleiche:
        # - Accuracy
        # - ROI
        # - Sharpe Ratio
        # - Win Rate

        # Placeholder
        results = {
            'new_model_better': True,
            'accuracy_diff': 0.02,
            'roi_diff': 0.05
        }

        return results

    def run_training_cycle(self, force: bool = False):
        """
        Führe kompletten Training-Zyklus aus

        Steps:
        1. Check ob Retrain notwendig
        2. Prepare Data
        3. Train Models
        4. Register in Registry
        5. A/B Test vs Champion
        6. Deploy if better
        """
        print("\n" + "="*60)
        print("🔄 CONTINUOUS TRAINING CYCLE")
        print("="*60)

        # Check
        if not force and not self.should_retrain():
            print("⏸️  Kein Retrain notwendig")
            return

        # Prepare Data
        X, y = self.prepare_training_data()

        # Train Neural Network
        nn_model, nn_acc = self.train_neural_network(X, y, epochs=100)

        # Register
        nn_version = self.registry.register_model(
            model_type='neural_net',
            model_path='models/checkpoints/best_model.pth',
            training_samples=len(X),
            validation_accuracy=nn_acc
        )

        # Train XGBoost
        xgb_model, xgb_acc = self.train_xgboost(X, y)

        # Register
        xgb_version = self.registry.register_model(
            model_type='xgboost',
            model_path='models/xgboost/latest.pkl',
            training_samples=len(X),
            validation_accuracy=xgb_acc
        )

        # A/B Test (wenn Champion existiert)
        champion_nn = self.registry.get_champion('neural_net')
        if champion_nn:
            # ab_results = self.ab_test_models(
            #     nn_version.version_id,
            #     champion_nn.version_id,
            #     test_data=...
            # )
            #
            # if ab_results['new_model_better']:
            #     self.registry.set_champion(nn_version.version_id)
            pass
        else:
            # Kein Champion → Setze als ersten Champion
            self.registry.set_champion(nn_version.version_id)
            self.registry.set_champion(xgb_version.version_id)

        # Update State
        self._save_last_training_date()

        print("\n✅ Training Cycle abgeschlossen!")
        print("="*60)


# ==========================================================
# SCHEDULER FÜR AUTOMATED RETRAINING
# ==========================================================
class TrainingScheduler:
    """
    Scheduler für automatisches Retraining

    Kann im Hintergrund laufen und regelmäßig trainieren
    """

    def __init__(self, engine: ContinuousTrainingEngine):
        self.engine = engine
        self.is_running = False

    def start(self, check_interval_hours: int = 6):
        """Starte Scheduler"""
        print(f"🔄 Training Scheduler gestartet (Check alle {check_interval_hours}h)")

        self.is_running = True

        while self.is_running:
            try:
                # Run Training Cycle
                self.engine.run_training_cycle()

                # Wait
                time.sleep(check_interval_hours * 3600)

            except KeyboardInterrupt:
                print("\n⏹️  Scheduler gestoppt")
                self.is_running = False
                break
            except Exception as e:
                print(f"❌ Error in Scheduler: {e}")
                time.sleep(3600)  # Wait 1h bei Fehler

    def stop(self):
        """Stoppe Scheduler"""
        self.is_running = False


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    print("🔄 CONTINUOUS TRAINING SYSTEM")
    print("="*60)

    if not GPU_AVAILABLE:
        print("❌ GPU Models nicht verfügbar!")
        print("Stelle sicher dass gpu_ml_models.py verfügbar ist")
        exit(1)

    # Erstelle Engine
    engine = ContinuousTrainingEngine(
        database_path="game_database_sportmonks.csv",
        min_new_samples=50,
        retrain_schedule="daily"
    )

    # Manual Training
    print("\n📝 Führe manuellen Training-Cycle aus...")
    engine.run_training_cycle(force=True)

    # Registry Übersicht
    print("\n📊 Model Registry:")
    print("="*60)

    for model_type in ['neural_net', 'xgboost']:
        versions = engine.registry.get_version_history(model_type)
        champion = engine.registry.get_champion(model_type)

        print(f"\n{model_type.upper()}:")
        print(f"  Versionen: {len(versions)}")
        if champion:
            print(f"  Champion: {champion.version_id}")
            print(f"  Val Accuracy: {champion.validation_accuracy:.4f}")
            print(f"  Training Samples: {champion.training_samples}")

    print("\n✅ Test abgeschlossen!")
