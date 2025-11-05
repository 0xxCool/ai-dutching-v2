import pandas as pd
import numpy as np
import time
import os
import datetime as dt
from dataclasses import dataclass, field
from scipy import stats
from difflib import SequenceMatcher
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path
import pickle
import torch
import traceback
import io # Für In-Memory CSV-Verarbeitung
import unicodedata # NEU: Für Umlaute

# --- NEUE IMPORTE FÜR ML-INTEGRATION ---
# Stelle sicher, dass diese .py-Dateien im selben Ordner liegen
from optimized_poisson_model import VectorizedPoissonModel, PoissonConfig
from gpu_ml_models import GPUFeatureEngineer, GPUNeuralNetworkPredictor, GPUXGBoostPredictor, GPUConfig
from continuous_training_system import ModelRegistry
# ----------------------------------------

load_dotenv()

# ==========================================================
# KONFIGURATION (DEBUG AKTIVIERT)
# ==========================================================
@dataclass
class Config(PoissonConfig): # Erbt jetzt von PoissonConfig
    BANKROLL: float = 1000.0
    KELLY_CAP: float = 0.15
    MAX_STAKE_PERCENT: float = 0.10
    BASE_EDGE: float = -0.08
    ADAPTIVE_EDGE_FACTOR: float = 0.10
    MIN_ODDS: float = 1.10
    MAX_ODDS: float = 100.0
    SAVE_RESULTS: bool = True
    DEBUG_MODE: bool = True # *** DEBUG IST JETZT AKTIV ***
    
    ANALYZE_MULTIPLE_MARKETS: bool = True
    USE_FALLBACK_DATA: bool = True
    MIN_DATA_CONFIDENCE: float = 0.0
    
    # Ensemble-Gewichtung
    WEIGHT_POISSON: float = 1.0
    WEIGHT_NN: float = 0.0
    WEIGHT_XGB: float = 0.0
    
    # Toleranz für "Aggressive" Wetten
    AGGRESSIVE_EV_TOLERANCE: float = -0.20 # Zeige Wetten bis -20% EV
    
    OUTPUT_FILE: str = field(default_factory=lambda: f'sportmonks_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv')

# ==========================================================
# UTILITY-KLASSEN (*** TEAMMATCHER VERBESSERT ***)
# ==========================================================
class TeamMatcher:
    """
    Verbesserte Version mit expliziten Mappings für bekannte Teams.
    """
    
    # Bekannte Team-Mappings: Sportmonks → Football-Data
    KNOWN_MAPPINGS = {
        # =================================================================
        # PREMIER LEAGUE
        # =================================================================
        
        # Arsenal
        'arsenal': 'arsenal',
        'arsenal fc': 'arsenal',
        
        # Aston Villa
        'aston villa': 'aston villa',
        'aston villa fc': 'aston villa',
        
        # Bournemouth
        'afc bournemouth': 'bournemouth',
        'bournemouth': 'bournemouth',
        
        # Brentford
        'brentford': 'brentford',
        'brentford fc': 'brentford',
        
        # Brighton
        'brighton & hove albion': 'brighton',
        'brighton and hove albion': 'brighton',
        'brighton': 'brighton',
        
        # Burnley
        'burnley': 'burnley',
        'burnley fc': 'burnley',
        
        # Chelsea
        'chelsea': 'chelsea',
        'chelsea fc': 'chelsea',
        
        # Crystal Palace
        'crystal palace': 'crystal palace',
        'crystal palace fc': 'crystal palace',
        
        # Everton
        'everton': 'everton',
        'everton fc': 'everton',
        
        # Fulham
        'fulham': 'fulham',
        'fulham fc': 'fulham',
        
        # Leeds United
        'leeds united': 'leeds',
        'leeds': 'leeds',
        
        # Liverpool
        'liverpool': 'liverpool',
        'liverpool fc': 'liverpool',
        
        # Manchester City
        'manchester city': 'man city',
        'man city': 'man city',
        
        # Manchester United
        'manchester united': 'man united',
        'man united': 'man united',
        
        # Newcastle United
        'newcastle united': 'newcastle',
        'newcastle': 'newcastle',
        
        # Nottingham Forest
        'nottingham forest': "nott'm forest",
        'nott\'m forest': "nott'm forest",
        'nottm forest': "nott'm forest",
        
        # Sunderland (Championship, aber in deinen Fixtures)
        'sunderland': 'sunderland',
        'sunderland afc': 'sunderland',
        
        # Tottenham
        'tottenham hotspur': 'tottenham',
        'tottenham': 'tottenham',
        
        # West Ham
        'west ham united': 'west ham',
        'west ham': 'west ham',
        
        # Wolverhampton
        'wolverhampton wanderers': 'wolves',
        'wolves': 'wolves',
        
        # =================================================================
        # BUNDESLIGA
        # =================================================================
        
        # Augsburg
        'fc augsburg': 'augsburg',
        'augsburg': 'augsburg',
        
        # Bayern Munich
        'bayern münchen': 'bayern munich',
        'bayern munchen': 'bayern munich',
        'fc bayern münchen': 'bayern munich',
        'fc bayern munchen': 'bayern munich',
        'bayern munich': 'bayern munich',
        
        # Borussia Dortmund
        'borussia dortmund': 'dortmund',
        'bor dortmund': 'dortmund',
        'dortmund': 'dortmund',
        
        # Borussia Mönchengladbach
        'borussia mönchengladbach': "m'gladbach",
        'borussia monchengladbach': "m'gladbach",
        'bor mgladbach': "m'gladbach",
        'mgladbach': "m'gladbach",
        "m'gladbach": "m'gladbach",
        
        # Eintracht Frankfurt
        'eintracht frankfurt': 'ein frankfurt',
        'ein frankfurt': 'ein frankfurt',
        
        # FC Köln
        'fc köln': 'fc koln',
        'fc koln': 'fc koln',
        '1 fc köln': 'fc koln',
        '1 fc koln': 'fc koln',
        'cologne': 'fc koln',
        
        # Freiburg
        'sc freiburg': 'freiburg',
        'freiburg': 'freiburg',
        
        # Hamburger SV (2. Bundesliga, aber in deinen Fixtures)
        'hamburger sv': 'hamburg',
        'hamburg': 'hamburg',
        
        # Heidenheim
        'heidenheim': 'heidenheim',
        'fc heidenheim': 'heidenheim',
        '1 fc heidenheim': 'heidenheim',
        
        # Hoffenheim
        'tsg hoffenheim': 'hoffenheim',
        'hoffenheim': 'hoffenheim',
        
        # Leverkusen
        'bayer 04 leverkusen': 'leverkusen',
        'bayer leverkusen': 'leverkusen',
        'leverkusen': 'leverkusen',
        
        # Mainz
        'fsv mainz 05': 'mainz',
        'mainz 05': 'mainz',
        'mainz': 'mainz',
        
        # RB Leipzig
        'rb leipzig': 'rb leipzig',
        'rasenballsport leipzig': 'rb leipzig',
        
        # St. Pauli
        'st pauli': 'st pauli',
        'fc st pauli': 'st pauli',
        'st. pauli': 'st pauli',
        
        # Stuttgart
        'vfb stuttgart': 'stuttgart',
        'stuttgart': 'stuttgart',
        
        # Union Berlin
        'union berlin': 'union berlin',
        'fc union berlin': 'union berlin',
        '1 fc union berlin': 'union berlin',
        
        # Werder Bremen
        'werder bremen': 'werder bremen',
        'sv werder bremen': 'werder bremen',
        
        # Wolfsburg
        'vfl wolfsburg': 'wolfsburg',
        'wolfsburg': 'wolfsburg',
        
        # =================================================================
        # LIGUE 1
        # =================================================================
        
        # Angers
        'angers sco': 'angers',
        'angers': 'angers',
        
        # Auxerre
        'aj auxerre': 'auxerre',
        'auxerre': 'auxerre',
        
        # Brest
        'stade brestois': 'brest',
        'brest': 'brest',
        
        # Le Havre
        'le havre ac': 'le havre',
        'le havre': 'le havre',
        
        # Lens
        'rc lens': 'lens',
        'lens': 'lens',
        
        # Lille
        'losc lille': 'lille',
        'lille': 'lille',
        
        # Lorient
        'fc lorient': 'lorient',
        'lorient': 'lorient',
        
        # Lyon
        'olympique lyonnais': 'lyon',
        'lyon': 'lyon',
        
        # Marseille
        'olympique marseille': 'marseille',
        'olympique de marseille': 'marseille',
        'marseille': 'marseille',
        
        # Metz
        'fc metz': 'metz',
        'metz': 'metz',
        
        # Monaco
        'as monaco': 'monaco',
        'monaco': 'monaco',
        
        # Nantes
        'fc nantes': 'nantes',
        'nantes': 'nantes',
        
        # Nice
        'ogc nice': 'nice',
        'nice': 'nice',
        
        # Paris Saint-Germain
        'paris saint germain': 'paris sg',
        'paris saint-germain': 'paris sg',
        'paris sg': 'paris sg',
        'psg': 'paris sg',
        
        # Paris FC (unterschiedlich zu PSG!)
        'paris': 'paris fc',
        'paris fc': 'paris fc',
        
        # Reims
        'stade de reims': 'reims',
        'reims': 'reims',
        
        # Rennes
        'stade rennais': 'rennes',
        'rennes': 'rennes',
        
        # Strasbourg
        'rc strasbourg': 'strasbourg',
        'strasbourg': 'strasbourg',
        
        # Toulouse
        'toulouse fc': 'toulouse',
        'toulouse': 'toulouse',
        
        # =================================================================
        # LA LIGA
        # =================================================================
        
        'atletico madrid': 'ath madrid',
        'athletic bilbao': 'ath bilbao',
        'barcelona': 'barcelona',
        'fc barcelona': 'barcelona',
        'real betis': 'betis',
        'celta de vigo': 'celta',
        'rcd espanyol': 'espanyol',
        'getafe': 'getafe',
        'getafe cf': 'getafe',
        'granada': 'granada',
        'granada cf': 'granada',
        'osasuna': 'osasuna',
        'ca osasuna': 'osasuna',
        'rayo vallecano': 'vallecano',
        'real madrid': 'real madrid',
        'real sociedad': 'sociedad',
        'sevilla': 'sevilla',
        'sevilla fc': 'sevilla',
        'valencia': 'valencia',
        'valencia cf': 'valencia',
        'villarreal': 'villarreal',
        'villarreal cf': 'villarreal',
        
        # =================================================================
        # SERIE A
        # =================================================================
        
        'atalanta': 'atalanta',
        'atalanta bc': 'atalanta',
        'bologna': 'bologna',
        'bologna fc': 'bologna',
        'cagliari': 'cagliari',
        'cagliari calcio': 'cagliari',
        'empoli': 'empoli',
        'empoli fc': 'empoli',
        'fiorentina': 'fiorentina',
        'acf fiorentina': 'fiorentina',
        'frosinone': 'frosinone',
        'frosinone calcio': 'frosinone',
        'genoa': 'genoa',
        'genoa cfc': 'genoa',
        'hellas verona': 'verona',
        'verona': 'verona',
        'inter': 'inter',
        'inter milan': 'inter',
        'fc internazionale milano': 'inter',
        'internazionale': 'inter',
        'juventus': 'juventus',
        'juventus fc': 'juventus',
        'lazio': 'lazio',
        'ss lazio': 'lazio',
        'lecce': 'lecce',
        'us lecce': 'lecce',
        'milan': 'milan',
        'ac milan': 'milan',
        'monza': 'monza',
        'ac monza': 'monza',
        'napoli': 'napoli',
        'ssc napoli': 'napoli',
        'roma': 'roma',
        'as roma': 'roma',
        'salernitana': 'salernitana',
        'us salernitana': 'salernitana',
        'sassuolo': 'sassuolo',
        'us sassuolo': 'sassuolo',
        'torino': 'torino',
        'torino fc': 'torino',
        'udinese': 'udinese',
        'udinese calcio': 'udinese',
    }
    
    @staticmethod
    def normalize(name: str) -> str:
        """
        Normalisiert Teamnamen für das Matching.
        
        WICHTIG: Diese Funktion wird NACH der Prüfung auf KNOWN_MAPPINGS verwendet!
        """
        if pd.isna(name):
            return ""
        
        # 1. Umlaute und Akzente entfernen
        try:
            name = ''.join(c for c in unicodedata.normalize('NFD', name)
                           if unicodedata.category(c) != 'Mn')
        except TypeError:
            pass
        
        # 2. Kleinbuchstaben
        name = str(name).lower().strip()
        
        # 3. Prüfe auf bekannte Mappings BEVOR wir weitere Ersetzungen machen
        # (das verhindert, dass "Manchester United" zu "man utd" wird und dann nicht mehr matched)
        if name in TeamMatcher.KNOWN_MAPPINGS:
            return TeamMatcher.KNOWN_MAPPINGS[name]
        
        # 4. Satzzeichen entfernen (aber VORSICHTIG)
        name = name.replace('.', '').replace("'", "").replace('&', ' ')
        name = name.replace('-', ' ')  # "Saint-Germain" → "Saint Germain"
        
        # 5. Standard-Ersetzungen (NACH den bekannten Mappings!)
        replacements = {
            ' fc': '',
            ' cf': '',
            ' afc': '',
            ' united': '',
            ' city': '',
            ' de': '',
            ' do': '',
            ' dos': '',
            ' das': '',
            ' da': '',
            ' ac': '',
            ' as': '',
            ' us': '',
            ' ss': '',
            ' sc': '',
            ' sv': '',
            ' vfb': '',
            ' vfl': '',
            ' tsv': '',
            ' fsv': '',
            ' spvgg': '',
            ' real': '',
            ' atletico': '',
            ' athletic': '',
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # 6. Doppelte Leerzeichen entfernen
        name = ' '.join(name.split())
        
        return name.strip()
    
    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei normalisierten Strings."""
        return SequenceMatcher(None, a, b).ratio()
    
    @staticmethod
    def find_best_match(team: str, teams_list: List[str], threshold: float = 0.85) -> Optional[str]:
        """
        Findet das beste Match für ein Team in einer Liste.
        
        ÄNDERUNG: Threshold von 0.5 auf 0.85 erhöht!
        
        WARUM?
        - 0.5: Zu niedrig → viele falsche Matches
        - 0.7: Immer noch zu niedrig → "Heidenheim" = "Hoffenheim" (0.70)
        - 0.85: Perfekt → nur sehr ähnliche Teams werden gemacht
        
        Args:
            team: Der zu suchende Teamname (von Sportmonks)
            teams_list: Liste von Teamnamen (von Football-Data)
            threshold: Minimum-Ähnlichkeit (Standard: 0.85, vorher 0.5)
        
        Returns:
            Bester Match oder None
        """
        
        # 1. Normalisiere den Suchbegriff
        team_norm = TeamMatcher.normalize(team)
        
        # 2. Prüfe auf EXAKTE Übereinstimmung (schnell)
        for candidate in teams_list:
            candidate_norm = TeamMatcher.normalize(candidate)
            if team_norm == candidate_norm:
                return candidate
        
        # 3. Prüfe auf bekannte Mappings
        if team_norm in TeamMatcher.KNOWN_MAPPINGS:
            mapped_name = TeamMatcher.KNOWN_MAPPINGS[team_norm]
            for candidate in teams_list:
                candidate_norm = TeamMatcher.normalize(candidate)
                if mapped_name == candidate_norm:
                    return candidate
        
        # 4. Fuzzy Matching mit Ähnlichkeit (ERHÖHTER THRESHOLD!)
        best_match = None
        best_score = 0
        
        for candidate in teams_list:
            candidate_norm = TeamMatcher.normalize(candidate)
            score = SequenceMatcher(None, team_norm, candidate_norm).ratio()
            
            if score > best_score and score >= threshold:  # ← THRESHOLD HIER!
                best_score = score
                best_match = candidate
        
        return best_match


# ==========================================================
# DUTCHING CALCULATOR (Unverändert)
# ==========================================================
class OptimizedDutchingCalculator:
    
    def __init__(self, config: Config): 
        self.config = config
        
    def _calculate_confidence(self, evs: List[float], probs: List[float]) -> float:
        if not evs or not probs: return 0
        max_ev = max(evs)
        avg_prob = sum(probs) / len(probs)
        return max(0, min(1, (max_ev + 1) * avg_prob))
    
    def calculate_value_bet(self, odds: List[float], probs: List[float]) -> Tuple[List[float], Dict]:
        """
        NEUE VERSION: Gibt IMMER ein Ergebnis zurück (auch für aggressive Wetten)
        
        Labels:
        - "Konservativ (Value)": Positiver EV, Kelly-Stakes berechnet
        - "Aggressiv (Neg-EV)": Negativer EV (bis -20%), symbolische Stakes (1% Bankroll)
        - "Nicht wetten (Neg-EV)": Sehr negativer EV (< -20%), keine Stakes
        """
        
        if not all([odds, probs, len(odds) == len(probs)]): 
            return [], {}
        
        valid_bets = [(o, p) for o, p in zip(odds, probs) 
                    if self.config.MIN_ODDS <= o <= self.config.MAX_ODDS and o > 0]
        if not valid_bets: 
            return [], {}
        
        odds, probs = zip(*valid_bets)
        evs = [p * o - 1 for p, o in zip(probs, odds)]
        confidence = self._calculate_confidence(evs, probs)
        dynamic_min_edge = self.config.BASE_EDGE + (confidence * self.config.ADAPTIVE_EDGE_FACTOR)
        
        best_ev_idx = evs.index(max(evs))
        best_ev = evs[best_ev_idx]
        
        stakes = [0.0] * len(odds)
        bet_label = "Nicht wetten (Neg-EV)"
        
        # KONSERVATIV: Positiver EV → Kelly-Stakes
        if best_ev > dynamic_min_edge:
            bet_label = "Konservativ (Value)"
            kelly_f = min(best_ev / (odds[best_ev_idx] - 1), self.config.KELLY_CAP)
            total_stake = self.config.BANKROLL * kelly_f
            stakes[best_ev_idx] = total_stake
        
        # AGGRESSIV: Negativer EV, aber nicht zu schlecht → Symbolische Stakes
        elif best_ev > self.config.AGGRESSIVE_EV_TOLERANCE:
            bet_label = "Aggressiv (Neg-EV)"
            # Setze 1% der Bankroll als symbolischen Einsatz für die Analyse
            total_stake = self.config.BANKROLL * 0.01
            stakes[best_ev_idx] = total_stake
        
        # NICHT WETTEN: Sehr negativer EV
        else:
            bet_label = "Nicht wetten (Neg-EV)"
            # Stakes bleiben 0
        
        total_stake = sum(stakes)
        profit = best_ev * total_stake
        roi = (profit / total_stake * 100) if total_stake > 0 else 0
        
        metrics = {
            'bet_label': bet_label,
            'expected_value': best_ev, 
            'roi': roi, 
            'profit': profit, 
            'confidence': confidence, 
            'required_edge': dynamic_min_edge
        }
        
        return list(stakes), metrics

# ==========================================================
# XGDatabase (Unverändert)
# ==========================================================
class XGDatabase:
    
    def __init__(self, filepath: str, config: Config):
        self.config = config
        self.game_database = self._load_data_from_csv(filepath)
        self.global_avg_xg = self.game_database['home_xg'].mean() if not self.game_database.empty else 1.35

    def _load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            print(f"⚠️ Datenbank '{filepath}' nicht gefunden. Verwende Durchschnittswerte.")
            return pd.DataFrame()
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Datenbank geladen: {len(df)} historische Spiele")
            return df.sort_values(by='date')
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return pd.DataFrame()

    def get_xg_based_on_form(self, home_team: str, away_team: str, 
                            form_window: int = 8, debug: bool = False) -> Tuple[float, float, float]:
        
        if self.game_database.empty:
            if self.config.USE_FALLBACK_DATA:
                return self.global_avg_xg, self.global_avg_xg, 0.5
            return self.global_avg_xg, self.global_avg_xg, 0.0
        
        all_teams = pd.concat([self.game_database['home_team'], self.game_database['away_team']]).unique()
        
        home_team_match = TeamMatcher.find_best_match(home_team, all_teams, threshold=0.85)
        away_team_match = TeamMatcher.find_best_match(away_team, all_teams, threshold=0.85)
        
        data_confidence = 0.0
        
        if home_team_match:
            home_games = self.game_database[self.game_database['home_team'] == home_team_match].tail(form_window)
            if not home_games.empty:
                home_form_xg_for = home_games['home_xg'].mean()
                home_form_xg_against = home_games['away_xg'].mean()
                data_confidence += 0.5
            else: home_form_xg_for = home_form_xg_against = self.global_avg_xg
        else: home_form_xg_for = home_form_xg_against = self.global_avg_xg
        
        if away_team_match:
            away_games = self.game_database[self.game_database['away_team'] == away_team_match].tail(form_window)
            if not away_games.empty:
                away_form_xg_for = away_games['away_xg'].mean()
                away_form_xg_against = away_games['home_xg'].mean()
                data_confidence += 0.5
            else: away_form_xg_for = away_form_xg_against = self.global_avg_xg
        else: away_form_xg_for = away_form_xg_against = self.global_avg_xg
        
        if self.config.USE_FALLBACK_DATA and data_confidence < 0.5:
            data_confidence = 0.3
        
        home_lambda = (home_form_xg_for + away_form_xg_against) / 2
        away_lambda = (away_form_xg_for + home_form_xg_against) / 2
        
        return home_lambda, away_lambda, data_confidence

# ==========================================================
# ANALYZER (Unverändert)
# ==========================================================
class ComprehensiveAnalyzer:
    
    def __init__(self, config: Config):
        self.config = config
        self.xg_db = XGDatabase("game_database_complete.csv", config)
        self.poisson = VectorizedPoissonModel(config)
        self.dutching = OptimizedDutchingCalculator(config)
        self.matches_analyzed = 0
        self.matches_with_odds = 0
        self.matches_with_data = 0
        self.matches_profitable = 0
        self.matches_aggressive = 0
        print("\n🤖 Lade trainierte ML-Modelle...")
        self.registry = ModelRegistry()
        self.gpu_config = GPUConfig()
        if self.xg_db.game_database.empty:
            print("❌ WARNUNG: Historische Datenbank ist leer. ML-Features können nicht berechnet werden.")
            self.feature_engineer = None
            self.nn_model = None
            self.xgb_model = None
        else:
            self.feature_engineer = GPUFeatureEngineer(self.xg_db.game_database, self.gpu_config.device)
            self.nn_model = self._load_champion_model('neural_net')
            self.xgb_model = self._load_champion_model('xgboost')
        self.ensemble_weights = {'poisson': self.config.WEIGHT_POISSON, 'nn': self.config.WEIGHT_NN, 'xgb': self.config.WEIGHT_XGB}
        if not self.nn_model or not self.xgb_model:
            print("⚠️ WARNUNG: Konnte nicht alle ML-Modelle laden. Verwende reines Poisson-Modell.")
            self.ensemble_weights = {'poisson': 1.0, 'nn': 0.0, 'xgb': 0.0}

    def _load_champion_model(self, model_type: str) -> Optional[object]:
        champion_version = self.registry.get_champion(model_type)
        if not champion_version: print(f"  ❌ Kein 'Champion'-Modell für '{model_type}' gefunden."); return None
        model_path = champion_version.model_path
        if not os.path.exists(model_path): print(f"  ❌ Champion-Modell-Datei fehlt: {model_path}"); return None
        try:
            if model_type == 'neural_net':
                model = GPUNeuralNetworkPredictor(input_size=20, gpu_config=self.gpu_config)
                model.load_checkpoint(Path(model_path).stem); model.model.eval()
                print(f"  ✅ Champion 'neural_net' geladen: {champion_version.version_id}"); return model
            elif model_type == 'xgboost':
                model = GPUXGBoostPredictor(use_gpu=True)
                with open(model_path, 'rb') as f: model.model = pickle.load(f)
                model.is_trained = True
                print(f"  ✅ Champion 'xgboost' geladen: {champion_version.version_id}"); return model
        except Exception as e: print(f"  ❌ Fehler beim Laden von Champion-Modell {model_path}: {e}"); return None
        return None

    def _get_ensemble_probabilities(self, home, away, match_date, base_home_xg, base_away_xg) -> np.ndarray:
        lam_home, lam_away = self.poisson.calculate_lambdas(base_home_xg, base_away_xg)
        prob_matrix = self.poisson.calculate_score_probabilities(lam_home, lam_away)
        market_probs_poisson = self.poisson.calculate_market_probabilities(prob_matrix)['3Way Result']
        poisson_probs = np.array([market_probs_poisson['Home'], market_probs_poisson['Draw'], market_probs_poisson['Away']])
        if self.feature_engineer and self.nn_model and self.xgb_model:
            try:
                features_tensor = self.feature_engineer.create_match_features(home, away, match_date)
                features_np = features_tensor.cpu().numpy().reshape(1, -1)
                nn_probs = self.nn_model.predict_proba(features_np)[0]
                xgb_probs = self.xgb_model.predict_proba(features_np)[0]
                final_probs = (self.ensemble_weights['poisson'] * poisson_probs + self.ensemble_weights['nn'] * nn_probs + self.ensemble_weights['xgb'] * xgb_probs)
                return final_probs / final_probs.sum()
            except Exception as e:
                print(f"⚠️ WARNUNG: ML-Feature-Erstellung/Vorhersage fehlgeschlagen für {home} vs {away}. Verwende reines Poisson. Fehler: {e}")
                return poisson_probs
        return poisson_probs

    def analyze_match(self, fixture: Dict, odds_data: Dict) -> List[Dict]:
        self.matches_analyzed += 1
        try:
            home = fixture['participants'][0]['name']
            away = fixture['participants'][1]['name']
            match_date = pd.to_datetime(fixture['starting_at'])
        except Exception as e: print(f"(DEBUG: Konnte Fixture-Daten nicht parsen: {e})"); return []
        base_home, base_away, data_confidence = self.xg_db.get_xg_based_on_form(home, away, debug=self.config.DEBUG_MODE)
        if data_confidence < self.config.MIN_DATA_CONFIDENCE: return []
        self.matches_with_data += 1
        ensemble_probs_array = self._get_ensemble_probabilities(home, away, match_date, base_home, base_away)
        market_probs = {'3Way Result': {'Home': ensemble_probs_array[0], 'Draw': ensemble_probs_array[1], 'Away': ensemble_probs_array[2]}}
        results = []
        market_name = '3Way Result'
        if market_name not in odds_data or not odds_data[market_name]: return []
        odds = odds_data[market_name]['odds']
        selections = odds_data[market_name]['selections']
        probs = [market_probs[market_name][sel] for sel in selections]
        if not all(p > 0 for p in probs): return []
        stakes, metrics = self.dutching.calculate_value_bet(odds, probs)
        if metrics:
            if sum(stakes) > 0: self.matches_profitable += 1
            else: self.matches_aggressive += 1
            results.append({'date': dt.datetime.now(dt.UTC), 'home': home, 'away': away, 'market': market_name, 'selections': selections, 'odds': odds, 'probabilities': probs, 'stakes': stakes, 'total_stake': sum(stakes), 'profit': metrics['profit'], 'metrics': metrics})
        return results
    
    def print_summary(self):
        print(f"\n{'='*70}")
        print("📊 ANALYSE-STATISTIKEN")
        print(f"{'='*70}")
        print(f"  Analysierte Spiele: {self.matches_analyzed}")
        print(f"  Spiele mit Quoten: {self.matches_with_odds}")
        print(f"  Spiele mit Daten: {self.matches_with_data}")
        print(f"  Wetten (Konservativ): {self.matches_profitable}")
        print(f"  Wetten (Aggressiv): {self.matches_aggressive}")
        print(f"{'='*70}\n")

# ==========================================================
# ResultFormatter (Unverändert)
# ==========================================================
class ResultFormatter:
    
    def format_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        if not results: return pd.DataFrame()
        formatted = []
        for r in results:
            bet_selection = "N/A (Aggressiv)"
            if r['total_stake'] > 0:
                try: stake_idx = next(i for i, s in enumerate(r['stakes']) if s > 0); bet_selection = r['selections'][stake_idx]
                except StopIteration: bet_selection = "Fehler"
            formatted.append({'Date': r['date'].strftime('%Y-%m-%d %H:%M'), 'Match': f"{r['home']} vs {r['away']}", 'Bet_Type': r['metrics']['bet_label'], 'Market': r['market'], 'Selection': bet_selection, 'Odds': str([f"{o:.2f}" for o in r['odds']]), 'Probabilities': str([f"{p:.3f}" for p in r['probabilities']]), 'Total_Stake': f"€{r['total_stake']:.2f}", 'Expected_Profit': f"€{r['profit']:.2f}", 'ROI': f"{r['metrics']['roi']:.1f}%", 'EV': f"{r['metrics']['expected_value']:.4f}"})
        df = pd.DataFrame(formatted)
        df = df.sort_values(by=['Bet_Type', 'EV'], ascending=[True, False])
        return df
    
    def display_results(self, df: pd.DataFrame):
        if df.empty: print("\n❌ Keine Wetten (konservativ oder aggressiv) gefunden.\n"); return
        print("\n" + "="*70); print("💰 PROFITABLE & AGGRESSIVE WETTEN (1X2)"); print("="*70)
        print(df.to_string(index=False)); print("="*70 + "\n")

# ==========================================================
# SPORTMONKS API CLIENT (*** KORRIGIERT: SportmonksFixtureClient ***)
# ==========================================================
class SportmonksFixtureClient:
    """Holt NUR Fixture-Listen von Sportmonks, keine Quoten"""
    def __init__(self, api_token: str, config: Config):
        self.api_token = api_token
        self.config = config
        self.base_url = "https://api.sportmonks.com/v3/football"
        self.api_calls = 0
        self.max_api_calls = 2000
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
            """Zentrale Request-Funktion mit Retry-Logik & 1.3s Delay"""
            if self.api_calls >= self.max_api_calls:
                print(f"\n⚠️ API-Limit erreicht ({self.api_calls} Calls)")
                return None
                
            if params is None: params = {}
            params['api_token'] = self.api_token
            url = f"{self.base_url}/{endpoint}"
            
            # --- NEUES DEBUGGING (1/5) ---
            # Erstelle eine "sichere" Version der Parameter für den Log (ohne Token)
            safe_params = params.copy()
            if 'api_token' in safe_params:
                safe_params['api_token'] = '***REDACTED***'
            print(f"\n[DEBUG _make_request] Ziel-URL: {url}")
            print(f"[DEBUG _make_request] Parameter: {safe_params}")
            # --- ENDE DEBUGGING ---
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # --- NEUES DEBUGGING (2/5) ---
                    print(f"[DEBUG _make_request] Starte Versuch {attempt+1}/{max_retries} (Timeout: 20s)...")
                    # --- ENDE DEBUGGING ---
                    
                    response = self.session.get(url, params=params, timeout=20)
                    
                    # --- NEUES DEBUGGING (3/5) ---
                    print(f"[DEBUG _make_request] Request BEENDET. Status: {response.status_code}")
                    # --- ENDE DEBUGGING ---
                    
                    self.api_calls += 1
                    
                    if response.status_code == 429:
                        print(f"⚠️ Rate Limit (Status 429) - warte {2**attempt}s...")
                        time.sleep(2 ** attempt)
                        continue
                    
                    response.raise_for_status() # Löst Fehler aus bei 4xx/5xx
                    
                    print(f"[DEBUG _make_request] Status OK. Warte 1.3s...") # NEU
                    time.sleep(1.3)
                    
                    # --- NEUES DEBUGGING (4/5) ---
                    print(f"[DEBUG _make_request] Lese JSON-Antwort...")
                    # --- ENDE DEBUGGING ---
                    
                    json_data = response.json()
                    
                    print(f"[DEBUG _make_request] JSON gelesen. Gebe Daten zurück.") # NEU
                    return json_data
                    
                except requests.exceptions.Timeout:
                    # --- NEUES DEBUGGING (5/5) ---
                    print(f"❌ API-Fehler (Versuch {attempt+1}): TIMEOUT nach 20s.")
                    if attempt == max_retries - 1:
                        print(f"❌ API-Fehler nach {max_retries} Timeouts. Gebe None zurück.")
                        return None
                    time.sleep(1)
                    
                except requests.exceptions.RequestException as e:
                    print(f"❌ API-Fehler (Versuch {attempt+1}): {e}") # Verbessertes Logging
                    if attempt == max_retries - 1:
                        print(f"❌ API-Fehler nach {max_retries} Versuchen. Gebe None zurück.")
                        return None
                    time.sleep(1)
            return None
    
    def get_fixtures(self, date_from: str, date_to: str, league_ids: List[int]) -> List[Dict]:
        """
        Hole Fixtures für bestimmte Ligen und Zeitraum.
        
        METHODE: Kim's empfohlener zwei-stufiger Ansatz
        1. GET /leagues/{id}?include=seasons → finde current_season_id
        2. GET /seasons/{season_id}?include=fixtures → hole alle Spiele
        3. Filtere lokal nach Datum und Status
        """
        
        all_fixtures = []
        date_from_dt = dt.datetime.strptime(date_from, "%Y-%m-%d").date()
        date_to_dt = dt.datetime.strptime(date_to, "%Y-%m-%d").date()
        
        for league_id in league_ids:
            print(f"\n[DEBUG get_fixtures] Verarbeite Liga {league_id}...")
            
            # SCHRITT 1: Hole aktuelle Season-ID
            print(f"[DEBUG get_fixtures] Suche current_season_id für Liga {league_id}...")
            league_data = self._make_request(f"leagues/{league_id}", {'include': 'seasons'})
            
            if not league_data or 'data' not in league_data:
                print(f"[DEBUG get_fixtures] FEHLER: Keine Daten für Liga {league_id}")
                continue
            
            seasons = league_data['data'].get('seasons', [])
            if not seasons:
                print(f"[DEBUG get_fixtures] FEHLER: Keine Seasons für Liga {league_id}")
                continue
            
            # Suche nach is_current_season=True
            current_season = next((s for s in seasons if s.get('is_current_season') == True), None)
            
            if not current_season:
                print(f"[DEBUG get_fixtures] WARNUNG: Liga {league_id} hat keine 'is_current_season' Flag. Versuche es mit der letzten Saison.")
                current_season = seasons[-1]
            
            season_id = current_season.get('id')
            season_name = current_season.get('name', 'Unknown')
            print(f"[DEBUG get_fixtures] -> Aktuelle Saison ID ist: {season_id} ({season_name})")
            
            # SCHRITT 2: Hole ALLE Fixtures dieser Season
            print(f"[DEBUG get_fixtures] Hole alle Fixtures für Season {season_id}...")
            season_data = self._make_request(
                f"seasons/{season_id}",
                {'include': 'fixtures.participants;fixtures.league', 'per_page': 500}
            )
            
            if not season_data or 'data' not in season_data:
                print(f"[DEBUG get_fixtures] FEHLER: Keine Season-Daten für {season_id}")
                continue
            
            fixtures = season_data['data'].get('fixtures', [])
            if not fixtures:
                print(f"[DEBUG get_fixtures] WARNUNG: Keine Fixtures in Season {season_id}")
                continue
            
            print(f"[DEBUG get_fixtures] {len(fixtures)} Fixtures (gesamt) in Season gefunden.")
            
            # SCHRITT 3: Filtere lokal nach Datum und Status
            filtered = []
            for fixture in fixtures:
                # Prüfe Status: state_id == 1 bedeutet "Not Started"
                if fixture.get('state_id') != 1:
                    continue
                
                # Prüfe Datum
                try:
                    fixture_date_str = fixture.get('starting_at', '')
                    if not fixture_date_str:
                        continue
                    
                    fixture_date = dt.datetime.fromisoformat(
                        fixture_date_str.replace('Z', '+00:00')
                    ).date()
                    
                    if date_from_dt <= fixture_date <= date_to_dt:
                        filtered.append(fixture)
                
                except Exception as e:
                    if self.config.DEBUG_MODE:
                        print(f"[DEBUG get_fixtures] Konnte Datum nicht parsen für Fixture {fixture.get('id')}: {e}")
                    continue
            
            print(f"[DEBUG get_fixtures] -> {len(filtered)} Fixtures im Datumsbereich {date_from} bis {date_to} und 'Not Started'.")
            all_fixtures.extend(filtered)
        
        print(f"\n[DEBUG get_fixtures] === GESAMT: {len(all_fixtures)} Fixtures aus allen Ligen ===")
        return all_fixtures

# ==========================================================
# FOOTBALL-DATA.CO.UK CLIENT (*** MIT NEUEM DEBUGGING ***)
# ==========================================================
class FootballDataClient:
    """Holt LIVE-Quoten von Football-Data.co.uk CSVs"""

    def __init__(self):
        self.session = requests.Session()
        self.live_odds_db = pd.DataFrame() # Hier speichern wir alle Quoten

        # Liga-Mappings: Sportmonks Liga-ID → Football-Data Code
        self.league_mappings = {
            8: {'code': 'E0', 'name': 'Premier League'},
            82: {'code': 'D1', 'name': 'Bundesliga'},
            564: {'code': 'SP1', 'name': 'La Liga'},
            301: {'code': 'F1', 'name': 'Ligue 1'},
            384: {'code': 'I1', 'name': 'Serie A'},
        }
    
    def _get_current_season_code(self) -> str:
        """Ermittelt den Saison-Code (z.B. '2526')"""
        now = dt.datetime.now()
        if now.month >= 8: # Ab August
            start_year_short = now.strftime('%y')
            end_year_short = (now + dt.timedelta(days=365)).strftime('%y')
        else: # Vor August (z.B. Mai)
            start_year_short = (now - dt.timedelta(days=365)).strftime('%y')
            end_year_short = now.strftime('%y')
        return f"{start_year_short}{end_year_short}"

    def load_live_odds(self, league_ids_to_load: List[int]):
        """Lädt die CSVs der aktuellen Saison für die Top-5-Ligen"""
        
        print("\n💰 Lade Live-Quoten von Football-Data.co.uk...")
        season_code = self._get_current_season_code()
        
        all_odds = []

        for league_id in league_ids_to_load:
            if league_id not in self.league_mappings:
                continue

            league_config = self.league_mappings[league_id]
            code = league_config['code']
            name = league_config['name']
            
            url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
            
            try:
                print(f"   Lade {name} (Saison {season_code})...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                df = self._standardize_columns(df, name)
                all_odds.append(df)
                time.sleep(1)

            except Exception as e:
                print(f"   ⚠️ Fehler beim Laden von {name}: {e}")
                print(f"   URL: {url}")
                continue

        if all_odds:
            self.live_odds_db = pd.concat(all_odds, ignore_index=True)
            self.live_odds_db['home_norm'] = self.live_odds_db['home_team'].apply(TeamMatcher.normalize)
            self.live_odds_db['away_norm'] = self.live_odds_db['away_team'].apply(TeamMatcher.normalize)
            print(f"✅ {len(self.live_odds_db)} Live-Quoten geladen und für Matching vorbereitet.")
        else:
            print("❌ Es konnten keine Live-Quoten von Football-Data geladen werden.")

    def _standardize_columns(self, df: pd.DataFrame, league_name: str) -> pd.DataFrame:
        """
        FIXED VERSION: Behält 'date' als datetime (nicht date-Objekt)
        """
        
        if 'Date' in df.columns: 
            date_col = 'Date'
        elif 'date' in df.columns: 
            date_col = 'date'
        else: 
            return pd.DataFrame()

        required_cols = [date_col, 'HomeTeam', 'AwayTeam']
        if not all(col in df.columns for col in required_cols):
            print(f"   ⚠️ CSV für {league_name} fehlen Spalten (HomeTeam, AwayTeam).")
            return pd.DataFrame()

        standard_df = pd.DataFrame()
        
        # WICHTIGE ÄNDERUNG: Entferne .dt.date!
        try:
            standard_df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
            # ↑ KEIN .dt.date mehr! Bleibt datetime!
        except:
            try: 
                standard_df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%y', errors='coerce')
                # ↑ KEIN .dt.date mehr! Bleibt datetime!
            except: 
                print(f"   ⚠️ Datumsformat in {league_name} konnte nicht gelesen werden.")
                return pd.DataFrame()
        
        standard_df['home_team'] = df['HomeTeam']
        standard_df['away_team'] = df['AwayTeam']
        standard_df['league'] = league_name

        if 'B365H' in df.columns:
            standard_df['odds_home'] = pd.to_numeric(df['B365H'], errors='coerce')
            standard_df['odds_draw'] = pd.to_numeric(df['B365D'], errors='coerce')
            standard_df['odds_away'] = pd.to_numeric(df['B365A'], errors='coerce')
        elif 'PSH' in df.columns:
            standard_df['odds_home'] = pd.to_numeric(df['PSH'], errors='coerce')
            standard_df['odds_draw'] = pd.to_numeric(df['PSD'], errors='coerce')
            standard_df['odds_away'] = pd.to_numeric(df['PSA'], errors='coerce')
        else:
            print(f"   ⚠️ CSV für {league_name} hat keine Quoten-Spalten.")
            return pd.DataFrame()

        standard_df = standard_df.dropna(subset=['date', 'home_team', 'away_team'])
        
        return standard_df

    
    def find_odds_for_fixture(self, fixture: Dict, debug_mode: bool = False) -> Dict:
        """
        FIXED VERSION mit korrektem datetime-Handling.
        """
        
        try:
            # Extrahiere Team-Namen aus Sportmonks Fixture
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                if debug_mode:
                    print(f"[DEBUG] FEHLER: Fixture {fixture.get('id')} hat < 2 Teams")
                return {}
            
            home_team_sm = participants[0].get('name', '')
            away_team_sm = participants[1].get('name', '')
            fixture_date_str = fixture.get('starting_at', '')
            
            if not all([home_team_sm, away_team_sm, fixture_date_str]):
                if debug_mode:
                    print(f"[DEBUG] FEHLER: Unvollständige Fixture-Daten")
                return {}
            
            # Parse Datum (als datetime, nicht date!)
            try:
                import datetime as dt
                fixture_datetime = dt.datetime.fromisoformat(
                    fixture_date_str.replace('Z', '+00:00')
                )
                # Für Vergleich: Nur das Datum (aber datetime bleibt datetime!)
                fixture_date = fixture_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            except Exception as e:
                if debug_mode:
                    print(f"[DEBUG] FEHLER: Konnte Datum nicht parsen: {e}")
                return {}
            
            if debug_mode:
                print(f"\n[DEBUG] Suche Match für: {home_team_sm} vs {away_team_sm} (SM Datum: {fixture_date.date()})")
            
            # Normalisiere die Teamnamen
            home_norm_sm = TeamMatcher.normalize(home_team_sm)
            away_norm_sm = TeamMatcher.normalize(away_team_sm)
            
            if debug_mode:
                print(f"        Normalisiert SM: '{home_norm_sm}' vs '{away_norm_sm}'")
            
            # ===== STRATEGIE 1: EXAKTE MATCHES =====
            exact_matches = self.live_odds_db[
                (self.live_odds_db['home_norm'] == home_norm_sm) &
                (self.live_odds_db['away_norm'] == away_norm_sm)
            ]
            
            if not exact_matches.empty:
                # Finde das Match mit dem nächsten Datum
                exact_matches = exact_matches.copy()
                exact_matches['date_diff'] = abs(
                    (exact_matches['date'] - fixture_date).dt.total_seconds() / 86400
                )
                best_match_row = exact_matches.nsmallest(1, 'date_diff').iloc[0]
                
                home_odd = best_match_row['odds_home']
                draw_odd = best_match_row['odds_draw']
                away_odd = best_match_row['odds_away']
                
                if all([pd.notna(home_odd), pd.notna(draw_odd), pd.notna(away_odd),
                    home_odd > 0, draw_odd > 0, away_odd > 0]):
                    
                    if debug_mode:
                        print(f"        -> ✅ EXAKTES MATCH! Quoten: H={home_odd:.2f}, D={draw_odd:.2f}, A={away_odd:.2f}")
                    
                    return {
                        '3Way Result': {
                            'odds': [float(home_odd), float(draw_odd), float(away_odd)],
                            'selections': ['Home', 'Draw', 'Away']
                        }
                    }
            
            # ===== STRATEGIE 2: FUZZY MATCH auf Home-Team =====
            if debug_mode:
                print(f"        -> Kein exaktes Match. Versuche Fuzzy-Matching (Threshold: 0.5)...")
            
            all_home_teams = self.live_odds_db['home_team'].unique().tolist()
            
            home_match = TeamMatcher.find_best_match(
                home_team_sm,
                all_home_teams,
                threshold=0.85
            )
            
            if not home_match:
                if debug_mode:
                    print(f"        -> ❌ Kein Home-Team-Match gefunden für '{home_team_sm}'")
                return {}
            
            if debug_mode:
                home_match_norm = TeamMatcher.normalize(home_match)
                similarity = TeamMatcher.similarity(home_norm_sm, home_match_norm)
                print(f"        -> Home-Match gefunden: '{home_match}' (Ähnlichkeit: {similarity:.2f})")
            
            home_team_matches = self.live_odds_db[
                self.live_odds_db['home_team'] == home_match
            ]
            
            if home_team_matches.empty:
                if debug_mode:
                    print(f"        -> ❌ Keine Spiele für Home-Team '{home_match}' in FDC-DB")
                return {}
            
            # ===== STRATEGIE 3: FUZZY MATCH auf Away-Team =====
            possible_away_teams = home_team_matches['away_team'].unique().tolist()
            
            away_match = TeamMatcher.find_best_match(
                away_team_sm,
                possible_away_teams,
                threshold=0.85
            )
            
            if not away_match:
                if debug_mode:
                    print(f"        -> ❌ Kein Away-Team-Match gefunden für '{away_team_sm}'")
                    print(f"        -> Mögliche Away-Teams für '{home_match}':")
                    for candidate in possible_away_teams[:5]:
                        candidate_norm = TeamMatcher.normalize(candidate)
                        sim = TeamMatcher.similarity(away_norm_sm, candidate_norm)
                        print(f"           • '{candidate}' (Ähnlichkeit: {sim:.2f})")
                return {}
            
            if debug_mode:
                away_match_norm = TeamMatcher.normalize(away_match)
                similarity = TeamMatcher.similarity(away_norm_sm, away_match_norm)
                print(f"        -> Away-Match gefunden: '{away_match}' (Ähnlichkeit: {similarity:.2f})")
            
            # Finde das beste Spiel (nach Datum)
            final_matches = home_team_matches[
                home_team_matches['away_team'] == away_match
            ]
            
            if final_matches.empty:
                if debug_mode:
                    print(f"        -> ❌ Kein Spiel gefunden für '{home_match}' vs '{away_match}'")
                return {}
            
            # WICHTIG: Berechne Datumsdifferenz korrekt
            final_matches = final_matches.copy()
            final_matches['date_diff'] = abs(
                (final_matches['date'] - fixture_date).dt.total_seconds() / 86400
            )
            best_match_row = final_matches.nsmallest(1, 'date_diff').iloc[0]
            
            home_odd = best_match_row['odds_home']
            draw_odd = best_match_row['odds_draw']
            away_odd = best_match_row['odds_away']
            
            if all([pd.notna(home_odd), pd.notna(draw_odd), pd.notna(away_odd),
                home_odd > 0, draw_odd > 0, away_odd > 0]):
                
                if debug_mode:
                    date_diff = best_match_row['date_diff']
                    print(f"        -> ✅ FUZZY MATCH! Quoten: H={home_odd:.2f}, D={draw_odd:.2f}, A={away_odd:.2f}")
                    print(f"           (FDC Datum: {best_match_row['date'].date()}, Differenz: {date_diff:.0f} Tage)")
                
                return {
                    '3Way Result': {
                        'odds': [float(home_odd), float(draw_odd), float(away_odd)],
                        'selections': ['Home', 'Draw', 'Away']
                    }
                }
            
            else:
                if debug_mode:
                    print(f"        -> ❌ Match gefunden, aber Quoten ungültig (H={home_odd}, D={draw_odd}, A={away_odd})")
                return {}
        
        except Exception as e:
            if debug_mode:
                print(f"        -> ❌ Fehler beim Matchen: {e}")
                import traceback
                traceback.print_exc()
            return {}

# ==========================================================
# HAUPTSYSTEM (*** HAUPTSCHLEIFE KORRIGIERT ***)
# ==========================================================
class SportmonksDutchingSystem:
    
    def __init__(self, config: Config = None):
        print("Initialisiere Dutching System (mit Football-Data Quoten)...")
        self.config = config or Config()
        self.api_token = os.getenv("SPORTMONKS_API_TOKEN")
        
        if not self.api_token:
            print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
            raise ValueError("API Token fehlt")
        
        self.fixture_client = SportmonksFixtureClient(self.api_token, self.config)
        self.odds_client = FootballDataClient()
        self.analyzer = ComprehensiveAnalyzer(self.config)
        self.formatter = ResultFormatter()
        self.results = []
        
        self.league_ids = [
            8,    # Premier League
            82,   # Bundesliga
            564,  # La Liga
            384,  # Serie A
            301   # Ligue 1
        ]
        
        self.odds_client.load_live_odds(self.league_ids)
        
        print("✅ Dutching System initialisiert.")
    
    def run(self):
            print("\n" + "="*70)
            print("🚀 SPORTMONKS DUTCHING SYSTEM (mit FDC Quoten) WIRD GESTARTET")
            print("="*70 + "\n")
            
            if self.odds_client.live_odds_db.empty:
                print("❌ ABBRUCH: Keine Live-Quoten von Football-Data geladen...")
                return
            
            now = dt.datetime.now(dt.timezone.utc)
            date_from = now.strftime("%Y-%m-%d")
            date_to = (now + dt.timedelta(days=14)).strftime("%Y-%m-%d")
            
            print(f"Suche Spiele von {date_from} bis {date_to}...")
            print(f"Ligen: {len(self.league_ids)} (Top 5 Ligen)")
            
            # 1. Hole Fixtures von Sportmonks (Direkte Datumsabfrage, 1 Seite)
            print("Hole Fixtures von Sportmonks (direkte Abfrage)...")
            
            try:
                fixtures = self.fixture_client.get_fixtures(date_from, date_to, self.league_ids)

                # Filtere auf bekannte Top-Ligen
                top_league_names = ['Premier League', 'Bundesliga', 'Ligue 1']

                filtered_fixtures = [
                    f for f in fixtures 
                    if f.get('league', {}).get('name', '') in top_league_names
                ]

                print(f"✅ {len(filtered_fixtures)} Top-Liga-Spiele (von {len(fixtures)} gesamt)")
                fixtures = filtered_fixtures

            except Exception as e:
                print(f"     ❌ Fehler beim Abrufen der Fixtures: {e}")
                traceback.print_exc()
                fixtures = []

            if not fixtures:
                print("ℹ️  Keine *kommenden* Spiele im Datumsbereich gefunden.")
                print(f"\n📡 API-Nutzung (Sportmonks): {self.fixture_client.api_calls} Calls")
                return
            
            print(f"✅ {len(fixtures)} relevante Spiele für die Analyse gefunden.\n")
            
            # Der Rest des Skripts (ab hier) ist unverändert
            
            league_counts = {}
            for f in fixtures:
                # Stelle sicher, dass die verschachtelte Struktur korrekt gelesen wird
                if 'league' in f and f['league']:
                    league = f['league'].get('name', 'Unknown')
                else:
                    league = 'Unknown (League-Daten fehlen)'
                league_counts[league] = league_counts.get(league, 0) + 1
            
            print("Verteilung nach Ligen (gefiltert):")
            for league, count in sorted(league_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {league}: {count} Spiele")
            
            print(f"\nAnalysiere Spiele (Matche mit Football-Data Quoten)...\n")
            
            all_results = []
            
            # *** KORRIGIERTE HAUPTSCHLEIFE ***
            for fixture in tqdm(fixtures, desc="Fortschritt"):
                
                # 2. Hole Quoten von Football-Data (aus dem Speicher)
                odds_data = self.odds_client.find_odds_for_fixture(fixture, debug_mode=self.config.DEBUG_MODE)
                
                if odds_data.get('3Way Result'):
                    self.analyzer.matches_with_odds += 1
                    
                    # 3. Analysiere (wie zuvor)
                    match_results = self.analyzer.analyze_match(fixture, odds_data)
                    if match_results:
                        all_results.extend(match_results)
                # else:
                    # Das Debugging passiert jetzt in der Funktion 'find_odds_for_fixture'
                    pass
            
            # 4. Ergebnisse mit verbesserter Ausgabe
                self.analyzer.print_summary()
                self.results = all_results
                
                df = self.formatter.format_dataframe(self.results)
                self.formatter.display_results(df)
                
                if self.results:
                    # Kategorisiere Wetten
                    conservative_results = [r for r in self.results if r['metrics']['bet_label'] == "Konservativ (Value)"]
                    aggressive_results = [r for r in self.results if r['metrics']['bet_label'] == "Aggressiv (Neg-EV)"]
                    no_bet_results = [r for r in self.results if r['metrics']['bet_label'] == "Nicht wetten (Neg-EV)"]
                    
                    print("\n📊 ZUSAMMENFASSUNG (Alle Analysierten Spiele)")
                    print("="*70)
                    print(f"  • Konservative Wetten (positiver EV): {len(conservative_results)}")
                    print(f"  • Aggressive Wetten (negativer EV): {len(aggressive_results)}")
                    print(f"  • Nicht wetten (sehr negativer EV): {len(no_bet_results)}")
                    print(f"  • GESAMT analysierte Spiele: {len(self.results)}")
                    
                    # Details für konservative Wetten
                    if conservative_results:
                        total_stake = sum(r['total_stake'] for r in conservative_results)
                        total_profit = sum(r['profit'] for r in conservative_results)
                        avg_roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
                        
                        print(f"\n  💰 KONSERVATIVE WETTEN (Empfohlen):")
                        print(f"    • Gesamteinsatz: €{total_stake:.2f}")
                        print(f"    • Erwarteter Profit: €{total_profit:.2f}")
                        print(f"    • Durchschnittlicher ROI: {avg_roi:.1f}%")
                        
                        selections = {}
                        for r in conservative_results:
                            try:
                                stake_idx = next(i for i, s in enumerate(r['stakes']) if s > 0)
                                sel = r['selections'][stake_idx]
                                selections[sel] = selections.get(sel, 0) + 1
                            except StopIteration: 
                                pass
                        
                        print(f"\n    Wetten pro Selektion:")
                        for sel, count in sorted(selections.items(), key=lambda x: x[1], reverse=True):
                            print(f"      • {sel}: {count}x")
                    
                    # Details für aggressive Wetten
                    if aggressive_results:
                        print(f"\n  ⚠️  AGGRESSIVE WETTEN (Höheres Risiko):")
                        print(f"    • Diese Wetten haben negativen EV")
                        print(f"    • Nur für erfahrene Wetter mit Risikotoleranz")
                        
                        # Zeige die 3 besten aggressiven Wetten
                        best_aggressive = sorted(aggressive_results, key=lambda x: x['metrics']['expected_value'], reverse=True)[:3]
                        
                        print(f"\n    Top 3 Aggressive Wetten:")
                        for i, r in enumerate(best_aggressive, 1):
                            stake_idx = next((j for j, s in enumerate(r['stakes']) if s > 0), 0)
                            selection = r['selections'][stake_idx] if stake_idx < len(r['selections']) else "Unknown"
                            
                            print(f"      {i}. {r['home']} vs {r['away']}")
                            print(f"         Selection: {selection} @ {r['odds'][stake_idx]:.2f}")
                            print(f"         EV: {r['metrics']['expected_value']:.4f} | ROI: {r['metrics']['roi']:.1f}%")
                    
                    # Hinweis zu "Nicht wetten"
                    if no_bet_results:
                        print(f"\n  ❌ NICHT WETTEN:")
                        print(f"    • {len(no_bet_results)} Spiele mit sehr negativem EV")
                        print(f"    • Diese sollten vermieden werden")
                    
                    if self.config.SAVE_RESULTS:
                        df.to_csv(self.config.OUTPUT_FILE, index=False)
                        print(f"\n💾 Alle Ergebnisse gespeichert: {self.config.OUTPUT_FILE}")
                
                else:
                    print("\n❌ Keine Spiele analysiert.")
                
                print(f"\n📡 API-Nutzung (Sportmonks): {self.fixture_client.api_calls} von {self.fixture_client.max_api_calls} Calls")
                print("\n" + "="*70)
                print("✅ ANALYSE ABGESCHLOSSEN")
                print("="*70 + "\n")

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    try:
        config = Config()
        system = SportmonksDutchingSystem(config)
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen durch Benutzer")
    except Exception as e:
        print(f"\n\n❌ FEHLER: {e}")

        traceback.print_exc()

