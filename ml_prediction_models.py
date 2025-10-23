"""
Machine Learning Prediction Models für Fußball-Wetten

Enthält:
1. XGBoost Classifier
2. Neural Network (PyTorch)
3. Feature Engineering
4. Hybrid Ensemble Model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optionale Imports (installiert werden müssen)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost nicht installiert. pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch nicht installiert. pip install torch")


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
class FeatureEngineer:
    """
    Erstellt Features für ML-Modelle aus Spiel-Daten

    Features:
    - Team Form (letzte N Spiele)
    - xG Statistics
    - Head-to-Head Historie
    - Rest Days
    - Home/Away Performance
    """

    def __init__(self, database: pd.DataFrame):
        """
        Args:
            database: DataFrame mit Spalten: date, home_team, away_team,
                     home_xg, away_xg, home_score, away_score, league
        """
        self.db = database.copy()
        self.db['date'] = pd.to_datetime(self.db['date'])
        self.db = self.db.sort_values('date')

    def calculate_form(
        self,
        team: str,
        before_date: pd.Timestamp,
        n_games: int = 5,
        home_only: bool = False
    ) -> Dict[str, float]:
        """
        Berechne Form-Metriken für ein Team

        Returns:
            Dict mit: avg_goals_scored, avg_goals_conceded, avg_xg_for,
                     avg_xg_against, win_rate, points_per_game
        """
        # Filter Spiele vor dem Datum
        mask = self.db['date'] < before_date

        if home_only:
            team_games = self.db[mask & (self.db['home_team'] == team)]
            goals_for = team_games['home_score']
            goals_against = team_games['away_score']
            xg_for = team_games['home_xg']
            xg_against = team_games['away_xg']
        else:
            home_games = self.db[mask & (self.db['home_team'] == team)]
            away_games = self.db[mask & (self.db['away_team'] == team)]

            goals_for = pd.concat([home_games['home_score'], away_games['away_score']])
            goals_against = pd.concat([home_games['away_score'], away_games['home_score']])
            xg_for = pd.concat([home_games['home_xg'], away_games['away_xg']])
            xg_against = pd.concat([home_games['away_xg'], away_games['home_xg']])

        # Nimm letzte N Spiele
        recent_goals_for = goals_for.tail(n_games)
        recent_goals_against = goals_against.tail(n_games)
        recent_xg_for = xg_for.tail(n_games)
        recent_xg_against = xg_against.tail(n_games)

        # Berechne Win-Rate
        if home_only:
            recent_games = self.db[mask & (self.db['home_team'] == team)].tail(n_games)
            wins = (recent_games['home_score'] > recent_games['away_score']).sum()
            draws = (recent_games['home_score'] == recent_games['away_score']).sum()
        else:
            home_recent = self.db[mask & (self.db['home_team'] == team)].tail(n_games // 2)
            away_recent = self.db[mask & (self.db['away_team'] == team)].tail(n_games // 2)

            home_wins = (home_recent['home_score'] > home_recent['away_score']).sum()
            away_wins = (away_recent['away_score'] > away_recent['home_score']).sum()
            wins = home_wins + away_wins

            home_draws = (home_recent['home_score'] == home_recent['away_score']).sum()
            away_draws = (away_recent['away_score'] == away_recent['home_score']).sum()
            draws = home_draws + away_draws

        n_actual_games = len(recent_goals_for)

        if n_actual_games == 0:
            return {
                'avg_goals_scored': 1.0,
                'avg_goals_conceded': 1.0,
                'avg_xg_for': 1.0,
                'avg_xg_against': 1.0,
                'win_rate': 0.33,
                'points_per_game': 1.0
            }

        win_rate = wins / n_actual_games
        points = wins * 3 + draws
        ppg = points / n_actual_games

        return {
            'avg_goals_scored': float(recent_goals_for.mean() if len(recent_goals_for) > 0 else 1.0),
            'avg_goals_conceded': float(recent_goals_against.mean() if len(recent_goals_against) > 0 else 1.0),
            'avg_xg_for': float(recent_xg_for.mean() if len(recent_xg_for) > 0 else 1.0),
            'avg_xg_against': float(recent_xg_against.mean() if len(recent_xg_against) > 0 else 1.0),
            'win_rate': float(win_rate),
            'points_per_game': float(ppg)
        }

    def create_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Erstelle Feature-Vektor für ein Match

        Returns:
            Feature-Array mit 20 Features
        """
        # Home Team Form
        home_form = self.calculate_form(home_team, match_date, n_games=5)
        home_form_home = self.calculate_form(home_team, match_date, n_games=3, home_only=True)

        # Away Team Form
        away_form = self.calculate_form(away_team, match_date, n_games=5)

        features = [
            # Home Team Overall Form (6 features)
            home_form['avg_goals_scored'],
            home_form['avg_goals_conceded'],
            home_form['avg_xg_for'],
            home_form['avg_xg_against'],
            home_form['win_rate'],
            home_form['points_per_game'],

            # Home Team Home Form (3 features)
            home_form_home['avg_goals_scored'],
            home_form_home['avg_xg_for'],
            home_form_home['win_rate'],

            # Away Team Overall Form (6 features)
            away_form['avg_goals_scored'],
            away_form['avg_goals_conceded'],
            away_form['avg_xg_for'],
            away_form['avg_xg_against'],
            away_form['win_rate'],
            away_form['points_per_game'],

            # Differentials (5 features)
            home_form['avg_xg_for'] - away_form['avg_xg_against'],
            away_form['avg_xg_for'] - home_form['avg_xg_against'],
            home_form['avg_goals_scored'] - away_form['avg_goals_conceded'],
            away_form['avg_goals_scored'] - home_form['avg_goals_conceded'],
            home_form['points_per_game'] - away_form['points_per_game'],
        ]

        return np.array(features, dtype=np.float32)


# ==========================================================
# XGBOOST MODEL
# ==========================================================
class XGBoostMatchPredictor:
    """
    XGBoost Classifier für Match-Outcome Prediction

    Klassen:
    0 = Home Win
    1 = Draw
    2 = Away Win
    """

    def __init__(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost nicht installiert!")

        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )

        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trainiere Modell

        Args:
            X: Features [n_samples, n_features]
            y: Labels [n_samples] (0=Home, 1=Draw, 2=Away)
        """
        self.model.fit(X, y, verbose=False)
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Vorhersage Wahrscheinlichkeiten

        Returns:
            Array [n_samples, 3] mit [P(Home), P(Draw), P(Away)]
        """
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
            'home_home_goals', 'home_home_xg', 'home_home_wr',
            'away_goals_scored', 'away_goals_conceded', 'away_xg_for', 'away_xg_against',
            'away_win_rate', 'away_ppg',
            'diff_xg_home', 'diff_xg_away', 'diff_goals_home', 'diff_goals_away', 'diff_ppg'
        ]

        return dict(zip(feature_names, importance))


# ==========================================================
# NEURAL NETWORK
# ==========================================================
class MatchPredictionNet(nn.Module):
    """
    Feed-Forward Neural Network für Match Prediction

    Architektur:
    Input (20) → FC(128) → ReLU → BatchNorm → Dropout(0.3)
              → FC(64) → ReLU → BatchNorm → Dropout(0.2)
              → FC(32) → ReLU
              → FC(3) → Softmax
    """

    def __init__(self, input_size: int = 20):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return torch.softmax(logits, dim=1)


class NeuralNetworkPredictor:
    """Wrapper für PyTorch Neural Network"""

    def __init__(self, input_size: int = 20):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch nicht installiert!")

        self.model = MatchPredictionNet(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Trainiere Neural Network

        Args:
            X: Features [n_samples, n_features]
            y: Labels [n_samples]
            epochs: Anzahl Trainings-Epochen
            batch_size: Batch-Größe
        """
        # Convert zu Tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Vorhersage Wahrscheinlichkeiten

        Returns:
            Array [n_samples, 3] mit [P(Home), P(Draw), P(Away)]
        """
        if not self.is_trained:
            raise ValueError("Modell muss erst trainiert werden!")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            probs = self.model(X_tensor)

        return probs.numpy()


# ==========================================================
# HYBRID ENSEMBLE MODEL
# ==========================================================
@dataclass
class EnsembleWeights:
    """Gewichte für Ensemble-Modell"""
    poisson: float = 0.4
    xgboost: float = 0.35
    neural_net: float = 0.25


class HybridEnsembleModel:
    """
    Kombiniert mehrere Modelle für robuste Vorhersagen

    Modelle:
    1. Poisson-Modell (Baseline)
    2. XGBoost (Feature-basiert)
    3. Neural Network (Deep Learning)

    Ensemble-Methode: Weighted Average
    """

    def __init__(
        self,
        poisson_model,
        feature_engineer: FeatureEngineer,
        weights: EnsembleWeights = None
    ):
        self.poisson = poisson_model
        self.feature_engineer = feature_engineer
        self.weights = weights or EnsembleWeights()

        # ML Models (werden bei Bedarf trainiert)
        self.xgboost = None
        self.neural_net = None

        if XGBOOST_AVAILABLE:
            self.xgboost = XGBoostMatchPredictor()

        if TORCH_AVAILABLE:
            self.neural_net = NeuralNetworkPredictor()

    def train_ml_models(self, database: pd.DataFrame):
        """
        Trainiere XGBoost und Neural Network auf historischen Daten

        Args:
            database: DataFrame mit historischen Spielen
        """
        print("🤖 Trainiere ML-Modelle...")

        # Feature Engineering
        X_list = []
        y_list = []

        for idx, row in database.iterrows():
            try:
                features = self.feature_engineer.create_match_features(
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

                X_list.append(features)
                y_list.append(label)

            except:
                continue

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"  Training-Samples: {len(X)}")

        # Train XGBoost
        if self.xgboost:
            print("  Training XGBoost...")
            self.xgboost.train(X, y)
            print("  ✅ XGBoost trainiert")

        # Train Neural Network
        if self.neural_net:
            print("  Training Neural Network...")
            self.neural_net.train(X, y, epochs=50, verbose=False)
            print("  ✅ Neural Network trainiert")

        print("✅ ML-Training abgeschlossen!\n")

    def predict(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        match_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Ensemble-Prediction für ein Match

        Returns:
            Dict mit 'Home', 'Draw', 'Away' Wahrscheinlichkeiten
        """
        # 1. Poisson-Modell
        lam_home, lam_away = self.poisson.calculate_lambdas(home_xg, away_xg)
        prob_matrix = self.poisson.calculate_score_probabilities(lam_home, lam_away)
        market_probs = self.poisson.calculate_market_probabilities(prob_matrix)

        poisson_probs = np.array([
            market_probs['3Way Result']['Home'],
            market_probs['3Way Result']['Draw'],
            market_probs['3Way Result']['Away']
        ])

        # 2. XGBoost
        if self.xgboost and self.xgboost.is_trained:
            features = self.feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            xgb_probs = self.xgboost.predict_proba(features.reshape(1, -1))[0]
        else:
            xgb_probs = poisson_probs  # Fallback

        # 3. Neural Network
        if self.neural_net and self.neural_net.is_trained:
            features = self.feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            nn_probs = self.neural_net.predict_proba(features.reshape(1, -1))[0]
        else:
            nn_probs = poisson_probs  # Fallback

        # Ensemble: Weighted Average
        final_probs = (
            self.weights.poisson * poisson_probs +
            self.weights.xgboost * xgb_probs +
            self.weights.neural_net * nn_probs
        )

        # Normalisiere (sollte schon normalisiert sein, aber sicher ist sicher)
        final_probs /= final_probs.sum()

        return {
            'Home': float(final_probs[0]),
            'Draw': float(final_probs[1]),
            'Away': float(final_probs[2])
        }


# ==========================================================
# BEISPIEL-NUTZUNG
# ==========================================================
if __name__ == "__main__":
    print("🤖 ML Prediction Models Test\n")

    # Mock-Daten erstellen
    n_games = 100
    mock_data = {
        'date': pd.date_range('2024-01-01', periods=n_games),
        'home_team': ['Team A'] * 50 + ['Team B'] * 50,
        'away_team': ['Team C'] * 50 + ['Team D'] * 50,
        'home_xg': np.random.uniform(0.5, 3.0, n_games),
        'away_xg': np.random.uniform(0.5, 3.0, n_games),
        'home_score': np.random.randint(0, 4, n_games),
        'away_score': np.random.randint(0, 4, n_games),
        'league': ['Test League'] * n_games
    }

    df = pd.DataFrame(mock_data)

    # Feature Engineering
    print("📊 Feature Engineering Test...")
    engineer = FeatureEngineer(df)

    features = engineer.create_match_features(
        'Team A',
        'Team C',
        pd.Timestamp('2024-02-01')
    )

    print(f"  Features Shape: {features.shape}")
    print(f"  Features: {features[:5]}")
    print("  ✅ Feature Engineering funktioniert\n")

    # XGBoost Test
    if XGBOOST_AVAILABLE:
        print("🌲 XGBoost Test...")
        xgb_model = XGBoostMatchPredictor()

        # Training Data
        X_train = np.random.rand(100, 20)
        y_train = np.random.randint(0, 3, 100)

        xgb_model.train(X_train, y_train)

        # Prediction
        X_test = np.random.rand(5, 20)
        probs = xgb_model.predict_proba(X_test)

        print(f"  Predictions Shape: {probs.shape}")
        print(f"  Sample Prediction: {probs[0]}")
        print(f"  Sum: {probs[0].sum():.4f} (sollte 1.0 sein)")
        print("  ✅ XGBoost funktioniert\n")
    else:
        print("⏭️  XGBoost übersprungen (nicht installiert)\n")

    # Neural Network Test
    if TORCH_AVAILABLE:
        print("🧠 Neural Network Test...")
        nn_model = NeuralNetworkPredictor()

        # Training Data
        X_train = np.random.rand(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 3, 100)

        nn_model.train(X_train, y_train, epochs=10, verbose=False)

        # Prediction
        X_test = np.random.rand(5, 20).astype(np.float32)
        probs = nn_model.predict_proba(X_test)

        print(f"  Predictions Shape: {probs.shape}")
        print(f"  Sample Prediction: {probs[0]}")
        print(f"  Sum: {probs[0].sum():.4f} (sollte 1.0 sein)")
        print("  ✅ Neural Network funktioniert\n")
    else:
        print("⏭️  Neural Network übersprungen (nicht installiert)\n")

    print("✅ Alle Tests abgeschlossen!")
