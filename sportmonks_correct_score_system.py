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
import itertools

load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class CorrectScoreConfig:
    BANKROLL: float = 1000.0
    KELLY_CAP: float = 0.20  # Konservativer für Correct Score
    MAX_STAKE_PERCENT: float = 0.08
    BASE_EDGE: float = -0.05  # Höherer Edge-Threshold
    ADAPTIVE_EDGE_FACTOR: float = 0.12
    MIN_ODDS: float = 3.0  # Correct Scores haben höhere Quoten
    MAX_ODDS: float = 500.0
    MAX_GOALS_HOME: int = 5
    MAX_GOALS_AWAY: int = 5
    HOME_ADVANTAGE: float = 0.15
    SAVE_RESULTS: bool = True
    DEBUG_MODE: bool = False
    
    # Correct Score spezifisch
    MIN_PROBABILITY: float = 0.01  # 1% Minimum für Correct Score
    TOP_N_SCORES: int = 15  # Analysiere Top 15 wahrscheinlichste Scores
    USE_FALLBACK_DATA: bool = True
    MIN_DATA_CONFIDENCE: float = 0.0
    
    # NEU: Toleranz für "Aggressive" Wetten
    AGGRESSIVE_EV_TOLERANCE: float = -0.20 # Zeige Wetten bis -20% EV
    
    OUTPUT_FILE: str = field(default_factory=lambda: f'correct_score_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv')

# ==========================================================
# TEAM MATCHER
# ==========================================================
class TeamMatcher:
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

# ==========================================================
# POISSON-MODELL für Correct Score
# ==========================================================
class CorrectScorePoissonModel:
    """Spezialisiertes Poisson-Modell für Correct Score Vorhersagen"""
    
    def __init__(self, config: CorrectScoreConfig): 
        self.config = config
        
    def calculate_lambdas(self, base_home: float, base_away: float) -> Tuple[float, float]:
        """Berechne angepasste Lambdas mit Home Advantage"""
        adj_home = base_home * (1 + self.config.HOME_ADVANTAGE)
        adj_away = base_away
        return max(0.3, min(4.0, adj_home)), max(0.3, min(4.0, adj_away))
    
    def calculate_score_probabilities(self, lam_home: float, lam_away: float) -> Dict[str, float]:
        """
        Berechne Wahrscheinlichkeiten für alle möglichen Correct Scores
        Rückgabe: {'0-0': 0.15, '1-0': 0.18, ...}
        """
        probs = {}
        
        for h in range(self.config.MAX_GOALS_HOME + 1):
            for a in range(self.config.MAX_GOALS_AWAY + 1):
                # Basis-Poisson-Wahrscheinlichkeit
                prob = stats.poisson.pmf(h, lam_home) * stats.poisson.pmf(a, lam_away)
                
                # Empirische Anpassungen (basierend auf realen Daten)
                if h == 0 and a == 0:  # 0-0 tritt häufiger auf
                    prob *= 1.12
                elif h == 1 and a == 1:  # 1-1 auch üblich
                    prob *= 1.08
                elif h == 1 and a == 0:  # 1-0 häufiges Ergebnis
                    prob *= 1.05
                elif h == 2 and a == 1:  # 2-1 üblich
                    prob *= 1.03
                
                score = f"{h}-{a}"
                probs[score] = prob
        
        # Normalisiere Wahrscheinlichkeiten
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def get_top_probable_scores(self, probs: Dict[str, float], n: int = 15) -> List[Tuple[str, float]]:
        """Gib die N wahrscheinlichsten Scores zurück"""
        sorted_scores = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]

# ==========================================================
# VALUE BET CALCULATOR (*** ANGEPASST ***)
# ==========================================================
class CorrectScoreValueCalculator:
    """Berechnet Value Bets für Correct Scores"""
    
    def __init__(self, config: CorrectScoreConfig): 
        self.config = config
        
    def _calculate_confidence(self, ev: float, prob: float) -> float:
        """Berechne Confidence Score"""
        if prob < self.config.MIN_PROBABILITY:
            return 0
        return max(0, min(1, (ev + 1) * prob * 5))  # *5 für Correct Score Scaling
    
    def calculate_value_bet(self, score: str, odds: float, prob: float) -> Tuple[float, Dict]:
        """
        Berechne Value Bet für einen einzelnen Correct Score
        
        Returns:
            stake: Empfohlener Einsatz (kann 0 sein)
            metrics: Dict mit Metriken
        """
        
        stake = 0.0
        
        # Prüfe Quoten-Range
        if odds < self.config.MIN_ODDS or odds > self.config.MAX_ODDS:
            return 0, {}
        
        # Prüfe Mindest-Wahrscheinlichkeit
        if prob < self.config.MIN_PROBABILITY:
            return 0, {}
        
        # Berechne Expected Value
        ev = prob * odds - 1
        
        # Berechne Confidence
        confidence = self._calculate_confidence(ev, prob)
        
        # Dynamischer Edge-Threshold
        dynamic_min_edge = self.config.BASE_EDGE + (confidence * self.config.ADAPTIVE_EDGE_FACTOR)
        
        # --- NEUE LOGIK: Klassifizieren statt filtern ---
        bet_label = "Nicht wetten (Neg-EV)"
        kelly_fraction = 0.0

        if ev > dynamic_min_edge:
            # KONSERVATIV: Das ist eine Value Bet nach unseren Regeln
            bet_label = "Konservativ (Value)"
            kelly_fraction = ev / (odds - 1)
            kelly_fraction = min(kelly_fraction, self.config.KELLY_CAP)
            
            stake = self.config.BANKROLL * kelly_fraction
            max_stake = self.config.BANKROLL * self.config.MAX_STAKE_PERCENT
            stake = min(stake, max_stake)

        elif ev > self.config.AGGRESSIVE_EV_TOLERANCE:
            # AGGRESSIV: Kein echter Value, aber in unserer Toleranz
            bet_label = "Aggressiv (Neg-EV)"
            stake = 0.0 # Wir setzen nicht, aber wir loggen es
        
        # else: bet_label bleibt "Nicht wetten (Neg-EV)"

        # Metriken
        profit = ev * stake
        roi = (profit / stake * 100) if stake > 0 else 0
        
        metrics = {
            'bet_label': bet_label,  # <-- NEUES FELD
            'expected_value': ev,
            'roi': roi,
            'profit': profit,
            'confidence': confidence,
            'required_edge': dynamic_min_edge,
            'probability': prob,
            'kelly_fraction': kelly_fraction
        }
        
        # Gib nur Metriken zurück, wenn wir sie anzeigen wollen
        if bet_label == "Nicht wetten (Neg-EV)":
             return 0, {} 

        return stake, metrics

# ==========================================================
# DATABASE
# ==========================================================
class CorrectScoreDatabase:
    """Lädt historische Correct Score Daten"""
    
    def __init__(self, filepath: str, config: CorrectScoreConfig):
        self.config = config
        # *** NUTZT JETZT DIE NEUE KOMPLETTE DATENBANK ***
        self.game_database = self._load_data(filepath)
        self.global_avg_xg = 1.35
        
        if not self.game_database.empty and 'home_xg' in self.game_database.columns:
            self.global_avg_xg = self.game_database['home_xg'].mean()
    
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Lade Datenbank"""
        if not os.path.exists(filepath):
            print(f"⚠️ '{filepath}' nicht gefunden. Verwende Durchschnitte.")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Datenbank geladen: {len(df)} Spiele")
            return df.sort_values(by='date') if 'date' in df.columns else df
        except Exception as e:
            print(f"❌ Fehler: {e}")
            return pd.DataFrame()
    
    def get_xg_based_on_form(self, home_team: str, away_team: str, 
                            form_window: int = 8) -> Tuple[float, float, float]:
        """Hole xG basierend auf Form"""
        
        if self.game_database.empty:
            if self.config.USE_FALLBACK_DATA:
                return self.global_avg_xg, self.global_avg_xg, 0.5
            return self.global_avg_xg, self.global_avg_xg, 0.0
        
        all_teams = pd.concat([
            self.game_database['home_team'], 
            self.game_database['away_team']
        ]).unique()
        
        home_match = TeamMatcher.find_best_match(home_team, all_teams)
        away_match = TeamMatcher.find_best_match(away_team, all_teams)
        
        confidence = 0.0
        
        # Home Team
        if home_match and 'home_xg' in self.game_database.columns:
            home_games = self.game_database[
                self.game_database['home_team'] == home_match
            ].tail(form_window)
            
            if not home_games.empty:
                home_for = home_games['home_xg'].mean()
                home_against = home_games['away_xg'].mean()
                confidence += 0.5
            else:
                home_for = home_against = self.global_avg_xg
        else:
            home_for = home_against = self.global_avg_xg
        
        # Away Team
        if away_match and 'away_xg' in self.game_database.columns:
            away_games = self.game_database[
                self.game_database['away_team'] == away_match
            ].tail(form_window)
            
            if not away_games.empty:
                away_for = away_games['away_xg'].mean()
                away_against = away_games['home_xg'].mean()
                confidence += 0.5
            else:
                away_for = away_against = self.global_avg_xg
        else:
            away_for = away_against = self.global_avg_xg
        
        if self.config.USE_FALLBACK_DATA and confidence < 0.5:
            confidence = 0.3
        
        home_lambda = (home_for + away_against) / 2
        away_lambda = (away_for + home_against) / 2
        
        return home_lambda, away_lambda, confidence

# ==========================================================
# ANALYZER (*** ANGEPASST ***)
# ==========================================================
class CorrectScoreAnalyzer:
    """Analysiert Matches für Correct Score Wetten"""
    
    def __init__(self, config: CorrectScoreConfig):
        self.config = config
        # *** NUTZT JETZT DIE NEUE KOMPLETTE DATENBANK ***
        self.database = CorrectScoreDatabase("game_database_complete.csv", config)
        self.poisson = CorrectScorePoissonModel(config)
        self.calculator = CorrectScoreValueCalculator(config)
        
        self.matches_analyzed = 0
        self.matches_with_odds = 0
        self.matches_with_data = 0
        self.bets_found = 0
        self.aggressive_bets = 0
    
    def analyze_match(self, fixture: Dict, odds_data: Dict) -> List[Dict]:
        """Analysiere ein Match für Correct Score Wetten"""
        
        self.matches_analyzed += 1
        
        # Team-Namen
        home = fixture['participants'][0]['name']
        away = fixture['participants'][1]['name']
        
        # Hole xG-basierte Lambdas
        base_home, base_away, confidence = self.database.get_xg_based_on_form(home, away)
        
        if confidence < self.config.MIN_DATA_CONFIDENCE:
            return []
        
        self.matches_with_data += 1
        
        # Berechne angepasste Lambdas
        lam_home, lam_away = self.poisson.calculate_lambdas(base_home, base_away)
        
        # Berechne Wahrscheinlichkeiten für alle Scores
        score_probs = self.poisson.calculate_score_probabilities(lam_home, lam_away)
        
        # Hole Top N wahrscheinlichste Scores
        top_scores = self.poisson.get_top_probable_scores(score_probs, self.config.TOP_N_SCORES)
        
        results = []
        
        # Prüfe ob Correct Score Odds verfügbar
        if not odds_data.get('Correct Score'):
            return []
        
        self.matches_with_odds += 1
        
        available_odds = odds_data['Correct Score']
        
        # Analysiere jeden Top-Score
        for score, prob in top_scores:
            # Prüfe ob Quote verfügbar
            if score not in available_odds:
                continue
            
            odds = available_odds[score]
            
            if odds <= 0:
                continue
            
            # Berechne Value Bet
            stake, metrics = self.calculator.calculate_value_bet(score, odds, prob)
            
            # *** ANGEPASSTE LOGIK ***
            # Füge hinzu, wenn Metriken zurückgegeben wurden (also Konservativ ODER Aggressiv)
            if metrics:
                if stake > 0:
                    self.bets_found += 1
                else:
                    self.aggressive_bets += 1
                
                if self.config.DEBUG_MODE:
                    print(f"\n  ✅ {metrics['bet_label']}: {home} vs {away}")
                    print(f"     Score: {score}")
                    print(f"     Odds: {odds:.2f}, Prob: {prob:.4f}, EV: {metrics['expected_value']:.3f}")
                    print(f"     Stake: €{stake:.2f}")
                
                results.append({
                    'date': dt.datetime.now(dt.UTC),
                    'home': home,
                    'away': away,
                    'correct_score': score,
                    'odds': odds,
                    'probability': prob,
                    'stake': stake,
                    'profit': metrics['profit'],
                    'metrics': metrics,
                    'lambdas': {'home': lam_home, 'away': lam_away}
                })
        
        return results
    
    def print_summary(self):
        """Ausgabe Statistiken"""
        print(f"\n{'='*70}")
        print("📊 ANALYSE-STATISTIKEN")
        print(f"{'='*70}")
        print(f"  Analysierte Spiele: {self.matches_analyzed}")
        print(f"  Spiele mit Quoten: {self.matches_with_odds}")
        print(f"  Spiele mit Daten: {self.matches_with_data}")
        print(f"  Gefundene Wetten (Konservativ): {self.bets_found}")
        print(f"  Gefundene Wetten (Aggressiv): {self.aggressive_bets}")
        print(f"{'='*70}\n")

# ==========================================================
# SPORTMONKS CLIENT
# ==========================================================
class SportmonksCorrectScoreClient:
    """Client für Sportmonks API mit Correct Score Odds"""
    
    def __init__(self, api_token: str, config: CorrectScoreConfig):
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
        """Hole Fixtures für bestimmte Ligen und Zeitraum (V3-SYNTAX)"""
        
        leagues_str = ",".join(map(str, league_ids))
        
        endpoint = f"fixtures/between/{date_from}/{date_to}"
        
        params = {
            'leagues': leagues_str,
            'include': 'participants;league'
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
    
    def get_correct_score_odds(self, fixture_id: int) -> Dict:
        """
        Hole Correct Score Quoten für ein Fixture
        Returns:
            Dict mit Correct Scores: {'1-0': 7.5, '2-1': 9.0, ...}
        """
        params = {
            'include': 'bookmaker;market'
        }
        
        data = self._make_request(f'odds/pre-match/fixtures/{fixture_id}', params)
        
        if not data or 'data' not in data:
            return {}
        
        return self._parse_correct_score_odds(data['data'])
    
    def _parse_correct_score_odds(self, odds_data: List[Dict]) -> Dict:
        """Parse Correct Score Odds aus API Response"""
        
        correct_scores = {}
        
        # Finde den "primären" Bookie (z.B. Pinnacle, ID 2)
        primary_odds_data = [o for o in odds_data if o.get('bookmaker', {}).get('id') == 2]
        if not primary_odds_data:
             primary_odds_data = odds_data # Fallback

        for odds_item in primary_odds_data:
            market = odds_item.get('market')
            if not market:
                continue

            market_name = market.get('name', '')
            
            # Suche nach "Correct Score" Market
            if 'Correct Score' in market_name or 'Exact Score' in market_name:
                
                bookmaker_odds_list = odds_item.get('bookmaker', {}).get('odds', [])
                if not bookmaker_odds_list:
                    continue
                
                odds_values = bookmaker_odds_list[0].get('odds', [])
                if not odds_values:
                    continue

                for odd in odds_values:
                    label = odd.get('label', '')  # z.B. "1-0", "2-1", etc.
                    value = odd.get('value', 0)   # Quote
                    
                    if value and value > 0 and '-' in label:
                        # Normalisiere Label (z.B. "1 - 0" -> "1-0")
                        score = "".join(label.split())
                        correct_scores[score] = float(value)
                
                # Nimm den ersten Bookie mit Correct Score Daten
                if correct_scores:
                    break
        
        return {'Correct Score': correct_scores} if correct_scores else {}

# ==========================================================
# RESULT FORMATTER (*** ANGEPASST ***)
# ==========================================================
class ResultFormatter:
    """Formatiert Ergebnisse für Ausgabe"""
    
    def format_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()
        
        formatted = []
        for r in results:
            formatted.append({
                'Date': r['date'].strftime('%Y-%m-%d %H:%M'),
                'Match': f"{r['home']} vs {r['away']}",
                'Bet_Type': r['metrics']['bet_label'], # <-- NEUE SPALTE
                'Correct_Score': r['correct_score'],
                'Odds': f"{r['odds']:.2f}",
                'Probability': f"{r['probability']:.4f} ({r['probability']*100:.2f}%)",
                'Stake': f"€{r['stake']:.2f}",
                'Expected_Profit': f"€{r['profit']:.2f}",
                'EV': f"{r['metrics']['expected_value']:.4f}",
            })
        
        df = pd.DataFrame(formatted)
        # Sortiere, sodass Konservative oben sind
        df = df.sort_values(by=['Bet_Type', 'EV'], ascending=[True, False])
        return df
    
    def display_results(self, df: pd.DataFrame):
        if df.empty:
            print("\n❌ Keine Wetten (konservativ oder aggressiv) gefunden.\n")
            return
        
        print("\n" + "="*70)
        print("⚽ PROFITABLE & AGGRESSIVE CORRECT SCORE WETTEN")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")

# ==========================================================
# HAUPT-SYSTEM
# ==========================================================
class CorrectScoreBettingSystem:
    """Hauptsystem für Correct Score Wetten"""
    
    def __init__(self, config: CorrectScoreConfig = None):
        print("Initialisiere Correct Score System...")
        self.config = config or CorrectScoreConfig()
        self.api_token = os.getenv("SPORTMONKS_API_TOKEN")
        
        if not self.api_token:
            print("❌ FEHLER: SPORTMONKS_API_TOKEN fehlt!")
            raise ValueError("API Token fehlt")
        
        self.client = SportmonksCorrectScoreClient(self.api_token, self.config)
        self.analyzer = CorrectScoreAnalyzer(self.config)
        self.formatter = ResultFormatter()
        self.results = []
        print("✅ Correct Score System initialisiert.")
    
    def run(self):
        print("\n" + "="*70)
        print("⚽ CORRECT SCORE BETTING SYSTEM")
        print("="*70 + "\n")
        
        # Datum-Range
        now = dt.datetime.now(dt.UTC)
        date_from = now.strftime("%Y-%m-%d")
        date_to = (now + dt.timedelta(days=14)).strftime("%Y-%m-%d")
        
        # Top Ligen
        league_ids = [8, 82, 564, 384, 301, 72, 271, 2, 390, 501]
        
        print(f"Suche Spiele: {date_from} bis {date_to}")
        print(f"Ligen: {len(league_ids)}\n")
        
        fixtures = self.client.get_fixtures(date_from, date_to, league_ids)
        
        if not fixtures:
            print("ℹ️  Keine Spiele gefunden.")
            return
        
        print(f"✅ {len(fixtures)} Spiele gefunden\n")
        
        # Zeige Verteilung
        league_counts = {}
        for f in fixtures:
            league = f.get('league', {}).get('name', 'Unknown')
            league_counts[league] = league_counts.get(league, 0) + 1
        
        print("Verteilung:")
        for league, count in sorted(league_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {league}: {count}")
        
        print(f"\nAnalysiere Spiele...\n")
        
        all_results = []
        
        for fixture in tqdm(fixtures, desc="Fortschritt"):
            fixture_id = fixture['id']
            
            odds_data = self.client.get_correct_score_odds(fixture_id)
            
            if odds_data.get('Correct Score'):
                match_results = self.analyzer.analyze_match(fixture, odds_data)
                if match_results:
                    all_results.extend(match_results)
        
        self.analyzer.print_summary()
        
        # Sortiere (passiert jetzt im Formatter)
        self.results = all_results
        
        df = self.formatter.format_dataframe(self.results)
        self.formatter.display_results(df)
        
        if self.results:
            # Nur "Konservative" Wetten für die Profit-Zusammenfassung zählen
            conservative_results = [r for r in self.results if r['stake'] > 0]
            
            if conservative_results:
                total_stake = sum(r['stake'] for r in conservative_results)
                total_profit = sum(r['profit'] for r in conservative_results)
                avg_roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
                
                print("📊 ZUSAMMENFASSUNG (Nur Konservative Wetten)")
                print("="*70)
                print(f"  • Gefundene Wetten: {len(conservative_results)}")
                print(f"  • Gesamteinsatz: €{total_stake:.2f}")
                print(f"  • Erwarteter Profit: €{total_profit:.2f}")
                print(f"  • Durchschnittlicher ROI: {avg_roi:.1f}%")
                
                # Top Scores
                scores = {}
                for r in conservative_results:
                    score = r['correct_score']
                    scores[score] = scores.get(score, 0) + 1
                
                print(f"\n  Häufigste Scores (Konservativ):")
                for score, count in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    • {score}: {count}x")
            
            else:
                print("📊 ZUSAMMENFASSUNG (Nur Konservative Wetten)")
                print("="*70)
                print("  • Keine konservativen Wetten mit positivem Stake gefunden.")


            if self.config.SAVE_RESULTS:
                df.to_csv(self.config.OUTPUT_FILE, index=False)
                print(f"\n💾 Ergebnisse (inkl. Aggressiv): {self.config.OUTPUT_FILE}")
        
        print(f"\n📡 API-Nutzung: {self.client.api_calls} von {self.client.max_api_calls}")
        print("\n" + "="*70)
        print("✅ ANALYSE ABGESCHLOSSEN")
        print("="*70 + "\n")

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    try:
        config = CorrectScoreConfig()
        system = CorrectScoreBettingSystem(config)
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen")
    except Exception as e:
        print(f"\n\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()