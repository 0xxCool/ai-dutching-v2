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

# --- NEUE IMPORTE FÜR ML-INTEGRATION ---
from optimized_poisson_model import VectorizedPoissonModel, PoissonConfig
from gpu_ml_models import GPUFeatureEngineer, GPUNeuralNetworkPredictor, GPUXGBoostPredictor, GPUConfig
from continuous_training_system import ModelRegistry
# ----------------------------------------

load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class Config(PoissonConfig): # Erbt jetzt von PoissonConfig
    BANKROLL: float = 1000.0
    KELLY_CAP: float = 0.25
    MAX_STAKE_PERCENT: float = 0.10
    BASE_EDGE: float = -0.08
    ADAPTIVE_EDGE_FACTOR: float = 0.10
    MIN_ODDS: float = 1.1
    MAX_ODDS: float = 100.0
    SAVE_RESULTS: bool = True
    DEBUG_MODE: bool = False
    
    ANALYZE_MULTIPLE_MARKETS: bool = True
    USE_FALLBACK_DATA: bool = True
    MIN_DATA_CONFIDENCE: float = 0.0
    
    # Ensemble-Gewichtung
    WEIGHT_POISSON: float = 0.34
    WEIGHT_NN: float = 0.33
    WEIGHT_XGB: float = 0.33
    
    OUTPUT_FILE: str = field(default_factory=lambda: f'sportmonks_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv')

# ==========================================================
# UTILITY-KLASSEN
# ==========================================================
class TeamMatcher:
    # ... (Keine Änderungen hier, Code bleibt gleich) ...
    @staticmethod
    def normalize(name: str) -> str:
        replacements = {
            'FC': '', 'CF': '', 'AFC': '', 'United': 'Utd', 'City': '',
            'de': '', 'do': '', 'dos': '', 'das': '', 'da': '',
            'AC': '', 'AS': '', 'US': '', 'SS': '', 'SC': '',
            'Real': '', 'Atletico': '', 'Athletic': '',
            'SV': '', 'VfB': '', 'VfL': '', 'TSV': '', 'FSV': '',
        }
        name = name.strip().lower()
        for old, new in replacements.items(): 
            name = name.replace(old.lower(), new.lower())
        return ' '.join(name.split())
    
    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, TeamMatcher.normalize(a), TeamMatcher.normalize(b)).ratio()

    @staticmethod
    def find_best_match(team: str, teams_list: List[str], threshold: float = 0.6) -> Optional[str]:
        best_match, best_score = None, 0
        for candidate in teams_list:
            score = TeamMatcher.similarity(team, candidate)
            if score > best_score and score >= threshold:
                best_score, best_match = score, candidate
        return best_match

# AdvancedPoissonModel-Klasse wird ENTFERNT
# Wir importieren stattdessen VectorizedPoissonModel

class OptimizedDutchingCalculator:
    # ... (Keine Änderungen hier, Code bleibt gleich) ...
    def __init__(self, config: Config): 
        self.config = config
        
    def _calculate_confidence(self, evs: List[float], probs: List[float]) -> float:
        if not evs or not probs: return 0
        max_ev = max(evs)
        avg_prob = sum(probs) / len(probs)
        return max(0, min(1, (max_ev + 1) * avg_prob))
    
    def calculate_value_bet(self, odds: List[float], probs: List[float]) -> Tuple[List[float], Dict]:
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
        
        if evs[best_ev_idx] > dynamic_min_edge:
            kelly_f = min(evs[best_ev_idx] / (odds[best_ev_idx] - 1), self.config.KELLY_CAP)
            total_stake = self.config.BANKROLL * kelly_f
            stakes = [0] * len(odds)
            stakes[best_ev_idx] = total_stake
        else: 
            return [], {}
        
        if not stakes or sum(stakes) == 0: 
            return [], {}
        
        total_stake = sum(stakes)
        profit = evs[best_ev_idx] * total_stake
        roi = (profit / total_stake * 100) if total_stake > 0 else 0
        
        metrics = {
            'expected_value': max(evs), 
            'roi': roi, 
            'profit': profit, 
            'confidence': confidence, 
            'required_edge': dynamic_min_edge
        }
        
        return list(stakes), metrics

# ==========================================================
# XGDatabase
# ==========================================================
class XGDatabase:
    # ... (Keine Änderungen hier, Code bleibt gleich) ...
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
        
        all_teams = pd.concat([
            self.game_database['home_team'], 
            self.game_database['away_team']
        ]).unique()
        
        home_team_match = TeamMatcher.find_best_match(home_team, all_teams, threshold=0.6)
        away_team_match = TeamMatcher.find_best_match(away_team, all_teams, threshold=0.6)
        
        data_confidence = 0.0
        
        if home_team_match:
            home_games = self.game_database[
                self.game_database['home_team'] == home_team_match
            ].tail(form_window)
            
            if not home_games.empty:
                home_form_xg_for = home_games['home_xg'].mean()
                home_form_xg_against = home_games['away_xg'].mean()
                data_confidence += 0.5
            else: 
                home_form_xg_for = home_form_xg_against = self.global_avg_xg
        else: 
            home_form_xg_for = home_form_xg_against = self.global_avg_xg
        
        if away_team_match:
            away_games = self.game_database[
                self.game_database['away_team'] == away_team_match
            ].tail(form_window)
            
            if not away_games.empty:
                away_form_xg_for = away_games['away_xg'].mean()
                away_form_xg_against = away_games['home_xg'].mean()
                data_confidence += 0.5
            else: 
                away_form_xg_for = away_form_xg_against = self.global_avg_xg
        else: 
            away_form_xg_for = away_form_xg_against = self.global_avg_xg
        
        if self.config.USE_FALLBACK_DATA and data_confidence < 0.5:
            data_confidence = 0.3
        
        home_lambda = (home_form_xg_for + away_form_xg_against) / 2
        away_lambda = (away_form_xg_for + home_form_xg_against) / 2
        
        return home_lambda, away_lambda, data_confidence

# ==========================================================
# ANALYZER (STARK ÜBERARBEITET)
# ==========================================================
class ComprehensiveAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.xg_db = XGDatabase("game_database_sportmonks.csv", config) 
        self.poisson = VectorizedPoissonModel(config) # Ersetzt
        self.dutching = OptimizedDutchingCalculator(config)
        self.matches_analyzed = 0
        self.matches_with_odds = 0
        self.matches_with_data = 0
        self.matches_profitable = 0
        
        # --- NEUE ML-MODELL-INTEGRATION ---
        print("\n🤖 Lade trainierte ML-Modelle...")
        self.registry = ModelRegistry()
        self.gpu_config = GPUConfig()
        
        # Stelle sicher, dass die DB für den Feature Engineer geladen ist
        if self.xg_db.game_database.empty:
            print("❌ WARNUNG: Historische Datenbank ist leer. ML-Features können nicht berechnet werden.")
            self.feature_engineer = None
            self.nn_model = None
            self.xgb_model = None
        else:
            self.feature_engineer = GPUFeatureEngineer(self.xg_db.game_database, self.gpu_config.device)
            self.nn_model = self._load_champion_model('neural_net')
            self.xgb_model = self._load_champion_model('xgboost')
        
        self.ensemble_weights = {
            'poisson': self.config.WEIGHT_POISSON,
            'nn': self.config.WEIGHT_NN,
            'xgb': self.config.WEIGHT_XGB
        }
        if not self.nn_model or not self.xgb_model:
            print("⚠️ WARNUNG: Konnte nicht alle ML-Modelle laden. Verwende reines Poisson-Modell.")
            self.ensemble_weights = {'poisson': 1.0, 'nn': 0.0, 'xgb': 0.0}
        # ------------------------------------

    def _load_champion_model(self, model_type: str) -> Optional[object]:
        """Lädt das beste trainierte Modell aus der Registry."""
        champion_version = self.registry.get_champion(model_type)
        if not champion_version:
            print(f"  ❌ Kein 'Champion'-Modell für '{model_type}' gefunden.")
            return None
        
        model_path = champion_version.model_path
        if not os.path.exists(model_path):
            print(f"  ❌ Champion-Modell-Datei fehlt: {model_path}")
            return None
            
        try:
            if model_type == 'neural_net':
                # (Annahme: 20 Features, wie in gpu_ml_models.py definiert)
                model = GPUNeuralNetworkPredictor(input_size=20, gpu_config=self.gpu_config)
                model.load_checkpoint(Path(model_path).stem) # Lade via Checkpoint-Namen
                model.model.eval() # Setze in Inferenzmodus
                print(f"  ✅ Champion 'neural_net' geladen: {champion_version.version_id}")
                return model
            
            elif model_type == 'xgboost':
                model = GPUXGBoostPredictor(use_gpu=True) # Versucht GPU
                with open(model_path, 'rb') as f:
                    model.model = pickle.load(f)
                model.is_trained = True
                print(f"  ✅ Champion 'xgboost' geladen: {champion_version.version_id}")
                return model
                
        except Exception as e:
            print(f"  ❌ Fehler beim Laden von Champion-Modell {model_path}: {e}")
            return None
        
        return None

    def _get_ensemble_probabilities(self, home, away, match_date, base_home_xg, base_away_xg) -> np.ndarray:
        """Kombiniert Poisson, NN und XGBoost zu einer Vorhersage."""
        
        # 1. Poisson-Modell
        lam_home, lam_away = self.poisson.calculate_lambdas(base_home_xg, base_away_xg)
        prob_matrix = self.poisson.calculate_score_probabilities(lam_home, lam_away)
        market_probs_poisson = self.poisson.calculate_market_probabilities(prob_matrix)['3Way Result']
        poisson_probs = np.array([market_probs_poisson['Home'], market_probs_poisson['Draw'], market_probs_poisson['Away']])

        # 2. ML-Modelle (NN & XGB)
        if self.feature_engineer and self.nn_model and self.xgb_model:
            try:
                # Erstelle Feature-Vektor
                features_tensor = self.feature_engineer.create_match_features(home, away, match_date)
                features_np = features_tensor.cpu().numpy().reshape(1, -1)
                
                # NN-Vorhersage
                nn_probs = self.nn_model.predict_proba(features_np)[0]
                
                # XGB-Vorhersage
                xgb_probs = self.xgb_model.predict_proba(features_np)[0]
                
                # 3. Ensemble (Gewichtete Mittelung)
                final_probs = (
                    self.ensemble_weights['poisson'] * poisson_probs +
                    self.ensemble_weights['nn'] * nn_probs +
                    self.ensemble_weights['xgb'] * xgb_probs
                )
                
                return final_probs / final_probs.sum() # Normalisieren
            
            except Exception as e:
                print(f"⚠️ WARNUNG: ML-Feature-Erstellung/Vorhersage fehlgeschlagen für {home} vs {away}. Verwende reines Poisson. Fehler: {e}")
                return poisson_probs # Fallback auf reines Poisson
        
        # Fallback, wenn ML-Modelle nicht geladen wurden
        return poisson_probs

    def analyze_match(self, fixture: Dict, odds_data: Dict) -> List[Dict]:
        self.matches_analyzed += 1
        
        try:
            home = fixture['participants'][0]['name']
            away = fixture['participants'][1]['name']
            match_date = pd.to_datetime(fixture['starting_at'])
        except Exception as e:
            print(f"(DEBUG: Konnte Fixture-Daten nicht parsen: {e})")
            return []
        
        # 1. Hole Basis-xG für Poisson
        base_home, base_away, data_confidence = self.xg_db.get_xg_based_on_form(
            home, away, debug=self.config.DEBUG_MODE
        )
        
        if data_confidence < self.config.MIN_DATA_CONFIDENCE:
            return []
        
        self.matches_with_data += 1
        
        # 2. Hole Ensemble-Wahrscheinlichkeiten (Poisson + NN + XGB)
        ensemble_probs_array = self._get_ensemble_probabilities(home, away, match_date, base_home, base_away)
        
        # Konvertiere Array zurück in Dictionary-Struktur
        market_probs = {
            '3Way Result': {
                'Home': ensemble_probs_array[0],
                'Draw': ensemble_probs_array[1],
                'Away': ensemble_probs_array[2]
            }
            # TODO: Erweitere dies, wenn du auch O/U, BTTS etc. mit den ML-Modellen trainierst.
            # Aktuell trainieren die ML-Modelle nur 1X2 (3 Klassen).
        }
        
        results = []
        
        # Analysiere nur 3Way Result, da nur hierfür Ensemble-Probs vorhanden sind
        market_name = '3Way Result'
        
        if market_name not in odds_data or not odds_data[market_name]:
            return []
        
        odds = odds_data[market_name]['odds']
        selections = odds_data[market_name]['selections']
        
        probs = [market_probs[market_name][sel] for sel in selections]
        
        if not all(p > 0 for p in probs):
            return []
        
        stakes, metrics = self.dutching.calculate_value_bet(odds, probs)
        
        if stakes and sum(stakes) > 0:
            self.matches_profitable += 1
            
            results.append({
                'date': dt.datetime.now(dt.UTC),
                'home': home,
                'away': away,
                'market': market_name,
                'selections': selections,
                'odds': odds,
                'probabilities': probs,
                'stakes': stakes,
                'total_stake': sum(stakes),
                'profit': metrics['profit'],
                'metrics': metrics
            })
        
        return results
    
    def print_summary(self):
        # ... (Keine Änderungen hier, Code bleibt gleich) ...
        print(f"\n{'='*70}")
        print("📊 ANALYSE-STATISTIKEN")
        print(f"{'='*70}")
        print(f"  Analysierte Spiele: {self.matches_analyzed}")
        print(f"  Spiele mit Quoten: {self.matches_with_odds}")
        print(f"  Spiele mit Daten: {self.matches_with_data}")
        print(f"  Profitable Wetten: {self.matches_profitable}")
        print(f"{'='*70}\n")

# ==========================================================
# ResultFormatter
# ==========================================================
class ResultFormatter:
    # ... (Keine Änderungen hier, Code bleibt gleich) ...
    def format_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()
        
        formatted = []
        for r in results:
            formatted.append({
                'Date': r['date'].strftime('%Y-%m-%d %H:%M'),
                'Match': f"{r['home']} vs {r['away']}",
                'Market': r['market'],
                'Selection': str(r['selections']),
                'Odds': str([f"{o:.2f}" for o in r['odds']]),
                'Probabilities': str([f"{p:.3f}" for p in r['probabilities']]),
                'Stakes': str([f"€{s:.2f}" for s in r['stakes']]),
                'Total_Stake': f"€{r['total_stake']:.2f}",
                'Expected_Profit': f"€{r['profit']:.2f}",
                'ROI': f"{r['metrics']['roi']:.1f}%",
                'EV': f"{r['metrics']['expected_value']:.4f}"
            })
        return pd.DataFrame(formatted)
    
    def display_results(self, df: pd.DataFrame):
        if df.empty:
            print("\n❌ Keine profitablen Wetten gefunden.\n")
            return
        
        print("\n" + "="*70)
        print("💰 PROFITABLE WETTEN")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")

# ==========================================================
# SPORTMONKS API CLIENT (ÜBERARBEITET)
# ==========================================================
class SportmonksClient:
    def __init__(self, api_token: str, config: Config):
        self.api_token = api_token
        self.config = config
        self.base_url = "https://api.sportmonks.com/v3/football"
        self.api_calls = 0
        self.max_api_calls = 2000  # Sicherheitslimit
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Zentrale Request-Funktion mit Retry-Logik & 1.3s Delay"""
        if self.api_calls >= self.max_api_calls:
            print(f"\n⚠️ API-Limit erreicht ({self.api_calls} Calls)")
            return None
            
        if params is None:
            params = {}
        
        params['api_token'] = self.api_token
        url = f"{self.base_url}/{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=20)
                self.api_calls += 1
                
                if response.status_code == 429:  # Rate Limit
                    wait_time = 2 ** attempt
                    print(f"⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                # WICHTIG: Delay NACH dem Request
                time.sleep(1.3) # Respektiere 3000 req/hr Limit
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ API-Fehler nach {max_retries} Versuchen: {e}")
                    return None
                time.sleep(1)
        
        return None
    
    def get_fixtures(self, date_from: str, date_to: str, league_ids: List[int]) -> List[Dict]:
        """Hole Fixtures für bestimmte Ligen und Zeitraum (NEUE V3-SYNTAX)"""
        
        # Konvertiere Liga-IDs in CSV-String
        leagues_str = ",".join(map(str, league_ids))
        
        # KORREKTER ENDPUNKT: /fixtures/between/{FROM_DATE}/{TO_DATE}
        endpoint = f"fixtures/between/{date_from}/{date_to}"
        
        params = {
            'leagues': leagues_str,
            'include': 'participants;league' # league für Logging
        }
        
        all_fixtures = []
        page = 1
        
        while True:
            if self.api_calls >= self.max_api_calls:
                break
            
            params['page'] = page
            data = self._make_request(endpoint, params)
            
            if not data or 'data' not in data:
                break
            
            fixtures = data['data']
            if not fixtures:
                break
            
            # Filter nur "Not Started" Status
            upcoming = [f for f in fixtures if f.get('state_id') == 1]
            all_fixtures.extend(upcoming)
            
            # Paginierungs-Check
            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break
                
            page += 1
            
        return all_fixtures
    
    def get_odds_for_fixture(self, fixture_id: int) -> Dict:
        """Hole Quoten für ein spezifisches Spiel"""
        params = {
            'include': 'bookmaker;market'
        }
        
        # Verwende den robusten _make_request
        data = self._make_request(f'odds/pre-match/fixtures/{fixture_id}', params)
        
        if not data or 'data' not in data:
            return self._empty_odds_dict()
        
        return self._parse_sportmonks_odds(data['data'])
    
    def _parse_sportmonks_odds(self, odds_data: List[Dict]) -> Dict:
        """Parse Sportmonks Odds in unser Format"""
        result = self._empty_odds_dict()
        
        # Finde den "primären" Bookie (z.B. Pinnacle, ID 2)
        primary_odds = [o for o in odds_data if o.get('bookmaker', {}).get('id') == 2]
        if not primary_odds:
             # Fallback: Nimm den ersten verfügbaren Bookie
             primary_odds = odds_data

        for odds_item in primary_odds:
            market = odds_item.get('market')
            if not market:
                continue

            market_name = market.get('name', '')
            
            # Hole Odds-Werte
            bookmaker_odds_list = odds_item.get('bookmaker', {}).get('odds', [])
            if not bookmaker_odds_list:
                continue
            
            # Nimm den ersten (und oft einzigen) Satz von Odds-Daten
            odds_values = bookmaker_odds_list[0].get('odds', [])
            if not odds_values:
                continue

            # 3Way Result (1X2)
            if market_name == '3Way Result' and not result.get('3Way Result'):
                home_odd = next((o['value'] for o in odds_values if o['label'] == 'Home'), 0)
                draw_odd = next((o['value'] for o in odds_values if o['label'] == 'Draw'), 0)
                away_odd = next((o['value'] for o in odds_values if o['label'] == 'Away'), 0)
                
                if all([home_odd, draw_odd, away_odd]):
                    result['3Way Result'] = {
                        'odds': [float(home_odd), float(draw_odd), float(away_odd)],
                        'selections': ['Home', 'Draw', 'Away']
                    }
            
            # Goals Over/Under (Suche speziell nach 2.5)
            elif 'Goals Over/Under' in market_name and market.get('handicap') == '2.5' and not result.get('Goals Over/Under'):
                over_odd = next((o['value'] for o in odds_values if o['label'] == 'Over'), 0)
                under_odd = next((o['value'] for o in odds_values if o['label'] == 'Under'), 0)
                
                if all([over_odd, under_odd]):
                    result['Goals Over/Under'] = {
                        'odds': [float(over_odd), float(under_odd)],
                        'selections': ['Over 2.5', 'Under 2.5']
                    }
            
            # Both Teams Score
            elif 'Both Teams Score' in market_name and not result.get('Both Teams Score'):
                yes_odd = next((o['value'] for o in odds_values if o['label'] == 'Yes'), 0)
                no_odd = next((o['value'] for o in odds_values if o['label'] == 'No'), 0)
                
                if all([yes_odd, no_odd]):
                    result['Both Teams Score'] = {
                        'odds': [float(yes_odd), float(no_odd)],
                        'selections': ['Yes', 'No']
                    }
            
            # Double Chance
            elif 'Double Chance' in market_name and not result.get('Double Chance'):
                hd = next((o['value'] for o in odds_values if o['label'] == 'Home/Draw'), 0)
                ha = next((o['value'] for o in odds_values if o['label'] == 'Home/Away'), 0)
                da = next((o['value'] for o in odds_values if o['label'] == 'Draw/Away'), 0)
                
                if all([hd, ha, da]):
                    result['Double Chance'] = {
                        'odds': [float(hd), float(ha), float(da)],
                        'selections': ['Home/Draw', 'Home/Away', 'Draw/Away']
                    }
        
        return result
    
    def _empty_odds_dict(self) -> Dict:
        return {
            '3Way Result': None,
            'Double Chance': None,
            'Goals Over/Under': None,
            'Both Teams Score': None
        }

# ==========================================================
# HAUPTSYSTEM
# ==========================================================
class SportmonksDutchingSystem:
    # ... (Fast keine Änderungen hier, nutzt die neuen Klassen) ...
    def __init__(self, config: Config = None):
        print("Initialisiere Dutching System...")
        self.config = config or Config()
        self.api_token = os.getenv("SPORTMONKS_API_TOKEN")
        
        if not self.api_token:
            print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
            raise ValueError("API Token fehlt")
        
        self.client = SportmonksClient(self.api_token, self.config)
        self.analyzer = ComprehensiveAnalyzer(self.config)
        self.formatter = ResultFormatter()
        self.results = []
        print("✅ Dutching System initialisiert.")
    
    def run(self):
        print("\n" + "="*70)
        print("🚀 SPORTMONKS DUTCHING SYSTEM WIRD GESTARTET")
        print("="*70 + "\n")
        
        now = dt.datetime.now(dt.timezone.utc)
        date_from = now.strftime("%Y-%m-%d")
        date_to = (now + dt.timedelta(days=14)).strftime("%Y-%m-%d")
        
        league_ids = [
            8, 82, 564, 384, 301, 72, 271, 2
        ]
        
        print(f"Suche Spiele von {date_from} bis {date_to}...")
        print(f"Ligen: {len(league_ids)}\n")
        
        fixtures = self.client.get_fixtures(date_from, date_to, league_ids)
        
        if not fixtures:
            print("ℹ️  Keine Spiele gefunden.")
            print(f"\n📡 API-Nutzung: {self.client.api_calls} Calls")
            return
        
        print(f"✅ {len(fixtures)} Spiele gefunden\n")
        
        league_counts = {}
        for f in fixtures:
            league = f.get('league', {}).get('name', 'Unknown')
            league_counts[league] = league_counts.get(league, 0) + 1
        
        print("Verteilung nach Ligen:")
        for league, count in sorted(league_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {league}: {count} Spiele")
        
        print(f"\nAnalysiere Spiele...\n")
        
        all_results = []
        
        for fixture in tqdm(fixtures, desc="Fortschritt"):
            fixture_id = fixture['id']
            
            odds_data = self.client.get_odds_for_fixture(fixture_id)
            
            if odds_data.get('3Way Result'): # Analysiere nur wenn Hauptmarkt vorhanden
                self.analyzer.matches_with_odds += 1
                match_results = self.analyzer.analyze_match(fixture, odds_data)
                if match_results:
                    all_results.extend(match_results)
        
        self.analyzer.print_summary()
        self.results = sorted(all_results, key=lambda x: x['profit'], reverse=True)
        df = self.formatter.format_dataframe(self.results)
        self.formatter.display_results(df)
        
        if self.results:
            total_stake = sum(r['total_stake'] for r in self.results)
            total_profit = sum(r['profit'] for r in self.results)
            avg_roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
            
            print("📊 ZUSAMMENFASSUNG")
            print("="*70)
            print(f"  • Gefundene Wetten: {len(self.results)}")
            print(f"  • Gesamteinsatz: €{total_stake:.2f}")
            print(f"  • Erwarteter Profit: €{total_profit:.2f}")
            print(f"  • Durchschnittlicher ROI: {avg_roi:.1f}%")
            
            markets = {}
            for r in self.results:
                market = r['market']
                if market not in markets:
                    markets[market] = 0
                markets[market] += 1
            
            print(f"\n  Wetten pro Markt:")
            for market, count in sorted(markets.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {market}: {count}")

            if self.config.SAVE_RESULTS:
                df.to_csv(self.config.OUTPUT_FILE, index=False)
                print(f"\n💾 Ergebnisse gespeichert: {self.config.OUTPUT_FILE}")
        
        print(f"\n📡 API-Nutzung: {self.client.api_calls} von {self.client.max_api_calls} Calls")
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