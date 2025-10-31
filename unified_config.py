"""
🎯 UNIFIED CONFIGURATION SYSTEM
=================================

Zentrale Konfiguration für ALLE Komponenten des AI Dutching Systems.

Verwendung:
    from unified_config import get_config, ConfigManager

    config = get_config()
    print(config.database.game_database)
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()


# ==========================================================
# DATABASE CONFIGURATION
# ==========================================================
@dataclass
class DatabaseConfig:
    """Datenbank Konfiguration"""
    # Main Database (from Hybrid Scraper)
    game_database: str = "game_database_complete.csv"
    game_database_xg_only: str = "game_database_xg_only.csv"
    game_database_odds_only: str = "game_database_odds_only.csv"

    # Correct Score Database
    correct_score_database: str = "correct_score_database.csv"

    # Results & Tracking
    results_dir: str = "results"
    backtest_dir: str = "backtests"


# ==========================================================
# API CONFIGURATION
# ==========================================================
@dataclass
class APIConfig:
    """Sportmonks API Konfiguration"""
    api_token: str = field(default_factory=lambda: os.getenv("SPORTMONKS_API_TOKEN", ""))
    base_url: str = "https://api.sportmonks.com/v3/football"
    request_delay: float = 1.3  # 3000 req/hr = 1 req/1.2s
    request_timeout: int = 60
    max_retries: int = 3
    max_api_calls: int = 2000


# ==========================================================
# CACHE CONFIGURATION
# ==========================================================
@dataclass
class CacheConfig:
    """API Cache Konfiguration"""
    cache_enabled: bool = True
    cache_dir: str = ".api_cache"
    default_ttl: int = 3600  # 1 Stunde

    # TTL pro Endpoint-Typ
    ttl_fixtures: int = 1800  # 30 Minuten
    ttl_odds: int = 300  # 5 Minuten
    ttl_leagues: int = 86400  # 24 Stunden
    ttl_historical: int = 2592000  # 30 Tage

    max_cache_size_mb: int = 500
    enable_compression: bool = True
    verbose: bool = False


# ==========================================================
# ML MODEL CONFIGURATION
# ==========================================================
@dataclass
class MLConfig:
    """Machine Learning Konfiguration"""
    # Model Directories
    model_dir: str = "models"
    registry_dir: str = "models/registry"
    checkpoint_dir: str = "models/checkpoints"

    # Training
    form_window: int = 5  # Letzte N Spiele für Features
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15

    # Neural Network
    nn_epochs: int = 100
    nn_batch_size: int = 64
    nn_learning_rate: float = 0.001
    nn_early_stopping_patience: int = 15

    # XGBoost
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_early_stopping_rounds: int = 50

    # Ensemble Weights
    weight_poisson: float = 0.34
    weight_nn: float = 0.33
    weight_xgb: float = 0.33

    # GPU
    use_gpu: bool = True
    use_mixed_precision: bool = True
    max_batch_size: int = 512


# ==========================================================
# DUTCHING CONFIGURATION
# ==========================================================
@dataclass
class DutchingConfig:
    """Dutching System Konfiguration"""
    # Bankroll Management
    bankroll: float = 1000.0
    kelly_cap: float = 0.25
    max_stake_percent: float = 0.10

    # Edge Requirements
    base_edge: float = -0.08
    adaptive_edge_factor: float = 0.10
    min_odds: float = 1.1
    max_odds: float = 100.0

    # Filters
    min_data_confidence: float = 0.0
    only_finished_games: bool = True

    # Output
    save_results: bool = True
    output_dir: str = "results"


# ==========================================================
# CASHOUT CONFIGURATION
# ==========================================================
@dataclass
class CashoutConfig:
    """Cashout Optimizer Konfiguration"""
    # Deep RL
    use_deep_rl: bool = True
    rl_model_path: str = "models/cashout_dqn.pth"

    # Thresholds
    min_profit_threshold: float = 0.10  # Min 10% Profit
    max_loss_threshold: float = -0.50  # Max 50% Loss

    # Live Monitoring
    check_interval_seconds: int = 60
    enable_auto_cashout: bool = False


# ==========================================================
# PORTFOLIO CONFIGURATION
# ==========================================================
@dataclass
class PortfolioConfig:
    """Portfolio Management Konfiguration"""
    # Exposure Limits
    max_total_exposure: float = 1.0
    max_market_exposure: float = 0.30
    max_league_exposure: float = 0.30
    max_match_exposure: float = 0.10

    # Diversification
    min_markets: int = 2
    min_leagues: int = 3
    max_correlation: float = 0.70

    # Risk
    max_var_95: float = 0.15
    target_sharpe: float = 2.0


# ==========================================================
# ALERT CONFIGURATION
# ==========================================================
@dataclass
class AlertConfig:
    """Alert System Konfiguration"""
    # Telegram
    telegram_enabled: bool = field(default_factory=lambda: bool(os.getenv("TELEGRAM_BOT_TOKEN")))
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # Discord
    discord_enabled: bool = field(default_factory=lambda: bool(os.getenv("DISCORD_WEBHOOK_URL")))
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    # Email
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_sender: str = field(default_factory=lambda: os.getenv("EMAIL_SENDER", ""))
    email_password: str = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD", ""))
    email_recipient: str = field(default_factory=lambda: os.getenv("EMAIL_RECIPIENT", ""))

    # Console
    console_enabled: bool = True

    # Rules
    min_value_bet_ev: float = 0.10
    min_cashout_profit: float = 50.0
    drawdown_threshold: float = 0.15


# ==========================================================
# BACKTESTING CONFIGURATION
# ==========================================================
@dataclass
class BacktestConfig:
    """Backtesting Framework Konfiguration"""
    # Data
    start_date: str = "2023-08-01"
    end_date: str = "2024-11-30"

    # Simulation
    initial_bankroll: float = 1000.0
    slippage: float = 0.02  # 2% Slippage
    commission: float = 0.0  # 0% Kommission (meiste Bookies)

    # Output
    save_results: bool = True
    output_dir: str = "backtests"
    generate_plots: bool = True


# ==========================================================
# CORRECT SCORE CONFIGURATION
# ==========================================================
@dataclass
class CorrectScoreConfig:
    """Correct Score System Konfiguration"""
    # Leagues für Correct Score
    leagues: List[int] = field(default_factory=lambda: [8, 82, 564, 301])  # EPL, BL, LL, L1

    # Filters
    min_probability: float = 0.05  # Min 5% Wahrscheinlichkeit
    max_odds: float = 50.0

    # Features
    use_historical_frequencies: bool = True
    use_team_tendencies: bool = True


# ==========================================================
# GPU CONFIGURATION
# ==========================================================
@dataclass
class GPUConfig:
    """GPU Monitoring & Performance Konfiguration"""
    enable_monitoring: bool = True
    log_interval_seconds: int = 60
    alert_on_high_temp: bool = True
    max_temperature: int = 85  # °C


# ==========================================================
# CONTINUOUS TRAINING CONFIGURATION
# ==========================================================
@dataclass
class ContinuousTrainingConfig:
    """Continuous Training System Konfiguration"""
    enabled: bool = True
    check_interval_hours: int = 24  # Prüfe täglich
    min_new_samples: int = 50  # Min 50 neue Spiele für Retraining
    auto_deploy_better_models: bool = True
    min_improvement: float = 0.01  # Min 1% bessere Accuracy


# ==========================================================
# LEAGUES CONFIGURATION
# ==========================================================
@dataclass
class LeaguesConfig:
    """Verfügbare Ligen"""
    leagues: Dict[str, int] = field(default_factory=lambda: {
        'Premier League': 8,
        'Bundesliga': 82,
        'La Liga': 564,
        'Serie A': 384,
        'Ligue 1': 301,
        'Eredivisie': 72,
        'Championship': 271,
        'Champions League': 2
    })

    # Default Ligen für Dutching
    default_dutching_leagues: List[int] = field(default_factory=lambda: [8, 82, 564, 384, 301])


# ==========================================================
# UNIFIED CONFIG
# ==========================================================
@dataclass
class UnifiedConfig:
    """
    Zentrale Konfiguration für das gesamte System

    Verwendung:
        config = UnifiedConfig()
        print(config.api.api_token)
        print(config.dutching.bankroll)
    """
    # Komponenten
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    dutching: DutchingConfig = field(default_factory=DutchingConfig)
    cashout: CashoutConfig = field(default_factory=CashoutConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    correct_score: CorrectScoreConfig = field(default_factory=CorrectScoreConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    continuous_training: ContinuousTrainingConfig = field(default_factory=ContinuousTrainingConfig)
    leagues: LeaguesConfig = field(default_factory=LeaguesConfig)

    # System
    debug_mode: bool = False
    verbose: bool = True

    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary"""
        return asdict(self)

    def save(self, filepath: str = "config.json"):
        """Speichere Konfiguration"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Konfiguration gespeichert: {filepath}")

    @classmethod
    def load(cls, filepath: str = "config.json") -> 'UnifiedConfig':
        """Lade Konfiguration"""
        if not Path(filepath).exists():
            print(f"⚠️ Konfigurationsdatei nicht gefunden: {filepath}")
            print(f"   Verwende Standard-Konfiguration")
            return cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Rekonstruiere verschachtelte Dataclasses
        config = cls()

        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'ml' in data:
            config.ml = MLConfig(**data['ml'])
        if 'dutching' in data:
            config.dutching = DutchingConfig(**data['dutching'])
        if 'cashout' in data:
            config.cashout = CashoutConfig(**data['cashout'])
        if 'portfolio' in data:
            config.portfolio = PortfolioConfig(**data['portfolio'])
        if 'alert' in data:
            config.alert = AlertConfig(**data['alert'])
        if 'backtest' in data:
            config.backtest = BacktestConfig(**data['backtest'])
        if 'correct_score' in data:
            config.correct_score = CorrectScoreConfig(**data['correct_score'])
        if 'gpu' in data:
            config.gpu = GPUConfig(**data['gpu'])
        if 'continuous_training' in data:
            config.continuous_training = ContinuousTrainingConfig(**data['continuous_training'])
        if 'leagues' in data:
            config.leagues = LeaguesConfig(**data['leagues'])

        if 'debug_mode' in data:
            config.debug_mode = data['debug_mode']
        if 'verbose' in data:
            config.verbose = data['verbose']

        print(f"✅ Konfiguration geladen: {filepath}")
        return config

    def validate(self) -> bool:
        """Validiere Konfiguration"""
        errors = []

        # API Token Check
        if not self.api.api_token:
            errors.append("❌ SPORTMONKS_API_TOKEN nicht gesetzt in .env")

        # Bankroll Check
        if self.dutching.bankroll <= 0:
            errors.append("❌ Bankroll muss > 0 sein")

        # Ensemble Weights Check
        total_weight = self.ml.weight_poisson + self.ml.weight_nn + self.ml.weight_xgb
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"❌ Ensemble Weights müssen 1.0 ergeben (aktuell: {total_weight})")

        # Database Check
        if not Path(self.database.game_database).exists():
            errors.append(f"⚠️ Haupt-Datenbank nicht gefunden: {self.database.game_database}")

        # Model Directory Check
        if not Path(self.ml.model_dir).exists():
            Path(self.ml.model_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Model-Verzeichnis erstellt: {self.ml.model_dir}")

        # Results Directory Check
        if not Path(self.dutching.output_dir).exists():
            Path(self.dutching.output_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Results-Verzeichnis erstellt: {self.dutching.output_dir}")

        # Ausgabe
        if errors:
            print("\n⚠️ KONFIGURATIONS-VALIDIERUNG:")
            for error in errors:
                print(f"  {error}")
            return False
        else:
            print("\n✅ Konfiguration valide!")
            return True


# ==========================================================
# CONFIG MANAGER (Singleton)
# ==========================================================
class ConfigManager:
    """
    Singleton Config Manager

    Verwendung:
        from unified_config import get_config
        config = get_config()
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_config(self, reload: bool = False) -> UnifiedConfig:
        """Hole Konfiguration (Singleton)"""
        if self._config is None or reload:
            # Versuche aus Datei zu laden
            if Path("config.json").exists():
                self._config = UnifiedConfig.load("config.json")
            else:
                self._config = UnifiedConfig()
                print("✅ Standard-Konfiguration erstellt")

        return self._config

    def save_config(self):
        """Speichere aktuelle Konfiguration"""
        if self._config:
            self._config.save()


# Global Singleton Accessor
def get_config(reload: bool = False) -> UnifiedConfig:
    """
    Hole globale Konfiguration

    Usage:
        from unified_config import get_config
        config = get_config()
    """
    manager = ConfigManager()
    return manager.get_config(reload=reload)


# ==========================================================
# CLI INTERFACE
# ==========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Configuration Manager")
    parser.add_argument('--validate', action='store_true', help='Validiere Konfiguration')
    parser.add_argument('--save', action='store_true', help='Speichere Konfiguration')
    parser.add_argument('--show', action='store_true', help='Zeige Konfiguration')

    args = parser.parse_args()

    config = get_config()

    if args.validate:
        config.validate()

    if args.save:
        config.save()

    if args.show:
        print("\n" + "=" * 70)
        print("UNIFIED CONFIGURATION")
        print("=" * 70)
        print(json.dumps(config.to_dict(), indent=2))

    if not any([args.validate, args.save, args.show]):
        print("\n🎯 Unified Configuration System")
        print("=" * 70)
        print("\nUsage:")
        print("  python unified_config.py --validate  # Validiere Konfiguration")
        print("  python unified_config.py --save      # Speichere Konfiguration")
        print("  python unified_config.py --show      # Zeige Konfiguration")
        print("\nIn Python:")
        print("  from unified_config import get_config")
        print("  config = get_config()")
        print("  print(config.api.api_token)")
