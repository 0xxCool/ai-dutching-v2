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

load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class Config:
    BANKROLL: float = 1000.0
    KELLY_CAP: float = 0.25
    MAX_STAKE_PERCENT: float = 0.10
    BASE_EDGE: float = -0.08
    ADAPTIVE_EDGE_FACTOR: float = 0.10
    MIN_ODDS: float = 1.1
    MAX_ODDS: float = 100.0
    MAX_GOALS: int = 5
    HOME_ADVANTAGE: float = 0.15
    SAVE_RESULTS: bool = True
    DEBUG_MODE: bool = False
    
    ANALYZE_MULTIPLE_MARKETS: bool = True
    USE_FALLBACK_DATA: bool = True
    MIN_DATA_CONFIDENCE: float = 0.0
    
    OUTPUT_FILE: str = field(default_factory=lambda: f'sportmonks_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv')

# ==========================================================
# UTILITY-KLASSEN
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

class AdvancedPoissonModel:
    def __init__(self, config: Config): 
        self.config = config
        
    def calculate_lambdas(self, base_home: float, base_away: float) -> Tuple[float, float]:
        adj_home = base_home * (1 + self.config.HOME_ADVANTAGE)
        adj_away = base_away
        return max(0.3, min(4.0, adj_home)), max(0.3, min(4.0, adj_away))
    
    def calculate_probabilities(self, lam_home: float, lam_away: float) -> pd.Series:
        probs = {}
        for h in range(self.config.MAX_GOALS + 1):
            for a in range(self.config.MAX_GOALS + 1):
                prob = stats.poisson.pmf(h, lam_home) * stats.poisson.pmf(a, lam_away)
                if h == 0 and a == 0: prob *= 1.08
                elif h == 1 and a == 1: prob *= 1.05
                probs[f"{h}-{a}"] = prob
        total = sum(probs.values())
        return pd.Series({k: v / total for k, v in probs.items()})
    
    def calculate_market_probabilities(self, probs: pd.Series) -> Dict:
        home_win = sum(probs[s] for s in probs.index if int(s.split('-')[0]) > int(s.split('-')[1]))
        draw = sum(probs[s] for s in probs.index if int(s.split('-')[0]) == int(s.split('-')[1]))
        away_win = 1 - home_win - draw
        
        over_05 = sum(probs[s] for s in probs.index if sum(map(int, s.split('-'))) > 0.5)
        over_15 = sum(probs[s] for s in probs.index if sum(map(int, s.split('-'))) > 1.5)
        over_25 = sum(probs[s] for s in probs.index if sum(map(int, s.split('-'))) > 2.5)
        over_35 = sum(probs[s] for s in probs.index if sum(map(int, s.split('-'))) > 3.5)
        
        btts = sum(probs[s] for s in probs.index 
                   if int(s.split('-')[0]) > 0 and int(s.split('-')[1]) > 0)
        
        home_or_draw = home_win + draw
        home_or_away = home_win + away_win
        draw_or_away = draw + away_win
        
        return {
            '3Way Result': {'Home': home_win, 'Draw': draw, 'Away': away_win},
            'Double Chance': {
                'Home/Draw': home_or_draw,
                'Home/Away': home_or_away,
                'Draw/Away': draw_or_away
            },
            'Goals Over/Under': {
                'Over 0.5': over_05, 'Under 0.5': 1 - over_05,
                'Over 1.5': over_15, 'Under 1.5': 1 - over_15,
                'Over 2.5': over_25, 'Under 2.5': 1 - over_25,
                'Over 3.5': over_35, 'Under 3.5': 1 - over_35
            },
            'Both Teams Score': {
                'Yes': btts, 'No': 1 - btts
            }
        }

class OptimizedDutchingCalculator:
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
# ANALYZER
# ==========================================================
class ComprehensiveAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        # Verwende korrekte Datenbank-Datei (von sportmonks_xg_scraper.py)
        self.xg_db = XGDatabase("game_database_sportmonks.csv", config) 
        self.poisson = AdvancedPoissonModel(config)
        self.dutching = OptimizedDutchingCalculator(config)
        self.matches_analyzed = 0
        self.matches_with_odds = 0
        self.matches_with_data = 0
        self.matches_profitable = 0
    
    def analyze_match(self, fixture: Dict, odds_data: Dict) -> List[Dict]:
        self.matches_analyzed += 1
        
        # Sportmonks verwendet 'participants' statt 'teams'
        home = fixture['participants'][0]['name']
        away = fixture['participants'][1]['name']
        
        base_home, base_away, data_confidence = self.xg_db.get_xg_based_on_form(
            home, away, debug=self.config.DEBUG_MODE
        )
        
        if data_confidence < self.config.MIN_DATA_CONFIDENCE:
            return []
        
        self.matches_with_data += 1
        
        lam_home, lam_away = self.poisson.calculate_lambdas(base_home, base_away)
        score_probs = self.poisson.calculate_probabilities(lam_home, lam_away)
        market_probs = self.poisson.calculate_market_probabilities(score_probs)
        
        results = []
        
        markets_to_analyze = ['3Way Result']
        if self.config.ANALYZE_MULTIPLE_MARKETS:
            markets_to_analyze.extend(['Double Chance', 'Goals Over/Under', 'Both Teams Score'])
        
        for market_name in markets_to_analyze:
            if market_name not in odds_data or not odds_data[market_name]:
                continue
            
            odds = odds_data[market_name]['odds']
            selections = odds_data[market_name]['selections']
            
            # Mappe Sportmonks selections auf unsere Wahrscheinlichkeiten
            probs = []
            for sel in selections:
                if market_name in market_probs and sel in market_probs[market_name]:
                    probs.append(market_probs[market_name][sel])
                else:
                    probs.append(0)
            
            if not all(p > 0 for p in probs):
                continue
            
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
# SPORTMONKS API CLIENT
# ==========================================================
class SportmonksClient:
    def __init__(self, api_token: str, config: Config):
        self.api_token = api_token
        self.config = config
        self.base_url = "https://api.sportmonks.com/v3/football"
        self.api_calls = 0
        self.max_api_calls = 2000  # Sportmonks: 3000/hour = ~2000 sicher
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Zentrale Request-Funktion mit Error Handling"""
        if self.api_calls >= self.max_api_calls:
            print(f"\n⚠️ API-Limit erreicht ({self.api_calls} Calls)")
            return {}
        
        url = f"{self.base_url}/{endpoint}"
        
        if params is None:
            params = {}
        
        params['api_token'] = self.api_token
        
        try:
            response = requests.get(url, params=params, timeout=15)
            self.api_calls += 1
            
            if response.status_code == 429:
                print(f"\n⚠️ Rate Limit erreicht!")
                return {}
            
            response.raise_for_status()
            data = response.json()
            
            time.sleep(0.2)  # Rate limiting
            
            return data
            
        except requests.exceptions.RequestException as e:
            if self.config.DEBUG_MODE:
                print(f"API Error: {e}")
            return {}
    
    def get_fixtures(self, date_from: str, date_to: str, league_ids: List[int]) -> List[Dict]:
        """Hole Fixtures für bestimmte Ligen und Zeitraum"""
        all_fixtures = []
        
        for league_id in league_ids:
            if self.api_calls >= self.max_api_calls:
                break
            
            params = {
                'filters': f'leagueIds:{league_id};fixtureStartingAt:{date_from},{date_to}',
                'include': 'participants'
            }
            
            data = self._make_request('fixtures', params)
            
            if data and 'data' in data:
                fixtures = data['data']
                # Filter nur "Not Started" Status
                upcoming = [f for f in fixtures if f.get('state', {}).get('state') == 'NS']
                all_fixtures.extend(upcoming)
        
        return all_fixtures
    
    def get_odds_for_fixture(self, fixture_id: int) -> Dict:
        """Hole Quoten für ein spezifisches Spiel"""
        params = {
            'include': 'bookmaker;market'
        }
        
        data = self._make_request(f'odds/pre-match/fixtures/{fixture_id}', params)
        
        if not data or 'data' not in data:
            return self._empty_odds_dict()
        
        return self._parse_sportmonks_odds(data['data'])
    
    def _parse_sportmonks_odds(self, odds_data: List[Dict]) -> Dict:
        """Parse Sportmonks Odds in unser Format"""
        result = self._empty_odds_dict()
        
        for odds_item in odds_data:
            if 'market' not in odds_item:
                continue
            
            market = odds_item['market']
            market_name = market.get('name', '')
            
            # 3Way Result (1X2)
            if market_name == '3Way Result':
                bookmaker_odds = odds_item.get('bookmaker', {})
                if 'odds' in bookmaker_odds:
                    odds_values = bookmaker_odds['odds']
                    
                    home_odd = next((o['value'] for o in odds_values if o['label'] == 'Home'), 0)
                    draw_odd = next((o['value'] for o in odds_values if o['label'] == 'Draw'), 0)
                    away_odd = next((o['value'] for o in odds_values if o['label'] == 'Away'), 0)
                    
                    if all([home_odd, draw_odd, away_odd]):
                        result['3Way Result'] = {
                            'odds': [float(home_odd), float(draw_odd), float(away_odd)],
                            'selections': ['Home', 'Draw', 'Away']
                        }
            
            # Goals Over/Under
            elif 'Goals Over/Under' in market_name:
                bookmaker_odds = odds_item.get('bookmaker', {})
                if 'odds' in bookmaker_odds:
                    odds_values = bookmaker_odds['odds']
                    
                    over_odd = next((o['value'] for o in odds_values if 'Over' in o['label']), 0)
                    under_odd = next((o['value'] for o in odds_values if 'Under' in o['label']), 0)
                    
                    # Extrahiere Line (z.B. "2.5")
                    line_label = next((o['label'] for o in odds_values if 'Over' in o['label']), '')
                    
                    if all([over_odd, under_odd]):
                        result['Goals Over/Under'] = {
                            'odds': [float(over_odd), float(under_odd)],
                            'selections': [line_label, line_label.replace('Over', 'Under')]
                        }
            
            # Both Teams Score
            elif 'Both Teams Score' in market_name:
                bookmaker_odds = odds_item.get('bookmaker', {})
                if 'odds' in bookmaker_odds:
                    odds_values = bookmaker_odds['odds']
                    
                    yes_odd = next((o['value'] for o in odds_values if o['label'] == 'Yes'), 0)
                    no_odd = next((o['value'] for o in odds_values if o['label'] == 'No'), 0)
                    
                    if all([yes_odd, no_odd]):
                        result['Both Teams Score'] = {
                            'odds': [float(yes_odd), float(no_odd)],
                            'selections': ['Yes', 'No']
                        }
            
            # Double Chance
            elif 'Double Chance' in market_name:
                bookmaker_odds = odds_item.get('bookmaker', {})
                if 'odds' in bookmaker_odds:
                    odds_values = bookmaker_odds['odds']
                    
                    hd = next((o['value'] for o in odds_values if 'Home' in o['label'] and 'Draw' in o['label']), 0)
                    ha = next((o['value'] for o in odds_values if 'Home' in o['label'] and 'Away' in o['label']), 0)
                    da = next((o['value'] for o in odds_values if 'Draw' in o['label'] and 'Away' in o['label']), 0)
                    
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
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.api_token = os.getenv("SPORTMONKS_API_TOKEN")
        
        if not self.api_token:
            print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
            print("\nBitte erstellen Sie eine .env Datei mit:")
            print("SPORTMONKS_API_TOKEN=your_token_here")
            raise ValueError("API Token fehlt")
        
        self.client = SportmonksClient(self.api_token, self.config)
        self.analyzer = ComprehensiveAnalyzer(self.config)
        self.formatter = ResultFormatter()
        self.results = []
    
    def run(self):
        print("\n" + "="*70)
        print("🚀 SPORTMONKS DUTCHING SYSTEM")
        print("="*70 + "\n")
        
        # Berechne Datum-Range
        now = dt.datetime.now(dt.UTC)
        date_from = now.strftime("%Y-%m-%d")
        date_to = (now + dt.timedelta(days=14)).strftime("%Y-%m-%d")
        
        # Top Europäische Ligen (Sportmonks League IDs)
        league_ids = [
            8,      # Premier League
            82,     # Bundesliga
            564,    # La Liga
            384,    # Serie A
            301,    # Ligue 1
            72,     # Eredivisie
            271,    # Primeira Liga
            2,      # Championship
            1489    # Serie A Brasil (wenn verfügbar)
        ]
        
        print(f"Suche Spiele von {date_from} bis {date_to}...")
        print(f"Ligen: {len(league_ids)}\n")
        
        fixtures = self.client.get_fixtures(date_from, date_to, league_ids)
        
        if not fixtures:
            print("ℹ️  Keine Spiele gefunden.")
            print(f"\n📡 API-Nutzung: {self.client.api_calls} Calls")
            return
        
        print(f"✅ {len(fixtures)} Spiele gefunden\n")
        
        # Zeige Verteilung
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
            
            if odds_data.get('3Way Result'):
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
        import traceback
        traceback.print_exc()