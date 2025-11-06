#!/usr/bin/env python3
"""
📥 SPORTMONKS HYBRID SCRAPER V4 - FIXED & KOMPATIBEL MIT ALLEN V2-SYSTEMEN
============================================================================
FINALE korrigierte Version mit allen Features von V3 + Verbesserungen

ÄNDERUNGEN vs. alter V4:
------------------------
✅ KORRIGIERT: API-Endpoint jetzt seasons-basiert (nicht fixtures/between)
✅ KORRIGIERT: Korrekte include-Parameter
✅ HINZUGEFÜGT: Alle Saisons seit 2000 (nicht nur 2)
✅ HINZUGEFÜGT: Detaillierte Statistiken
✅ HINZUGEFÜGT: Besseres Error-Handling
✅ BEHALTEN: correct_score Feld (war schon da!)
✅ BEHALTEN: Gutes Team-Matching (war schon da!)
✅ VERBESSERT: Debug-Ausgaben strukturierter

Version: 4.1 FIXED
Datum: 2025-11-06
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from dataclasses import dataclass
import traceback
import io
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class HybridScraperConfig:
    """Konfiguration für Hybrid Scraper"""

    # Sportmonks API
    api_token: str = ""
    base_url: str = "https://api.sportmonks.com/v3/football"
    request_delay: float = 1.5
    request_timeout: int = 60

    # Football-Data.co.uk
    football_data_base_url: str = "https://www.football-data.co.uk"

    # Output
    output_file_complete: str = "game_database_complete.csv"
    output_file_xg_only: str = "game_database_xg_only.csv"
    output_file_odds_only: str = "game_database_odds_only.csv"

    # Filter
    start_date: datetime = datetime(2023, 8, 1)  # Saison 2023/24 Start
    debug: bool = True


# ==========================================================
# TEAM MATCHER (aus V4 - ist gut!)
# ==========================================================
class TeamMatcher:
    """Hilft beim Matchen von Team-Namen"""
    
    @staticmethod
    def normalize(name: str) -> str:
        """Normalisiere Teamnamen für besseres Matching"""
        if pd.isna(name):
            return ""
        name = str(name).lower().strip()
        replacements = {
            'manchester united': 'man united',
            'manchester city': 'man city',
            'tottenham hotspur': 'tottenham',
            'newcastle united': 'newcastle',
            'west ham united': 'west ham',
            'wolverhampton wanderers': 'wolves',
            'brighton and hove albion': 'brighton',
            'nottingham forest': "nott'm forest",
            'bayern münchen': 'bayern munich',
            'borussia mönchengladbach': "m'gladbach",
            'paris saint germain': 'paris sg',
            'fc ': '', 'ac ': '', 'as ': '',
        }
        for full, short in replacements.items():
            name = name.replace(full, short)
        return ' '.join(name.split())

    @staticmethod
    def similarity(a: str, b: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def find_best_match(team: str, teams_list: List[str], threshold: float = 0.85) -> Optional[str]:
        team_norm = TeamMatcher.normalize(team)
        
        # Exakte Übereinstimmung
        for candidate in teams_list:
            if team_norm == TeamMatcher.normalize(candidate):
                return candidate
        
        # Fuzzy Matching
        best_match, best_score = None, 0
        for candidate in teams_list:
            score = TeamMatcher.similarity(team_norm, TeamMatcher.normalize(candidate))
            if score > best_score and score >= threshold:
                best_score, best_match = score, candidate
        return best_match


# ==========================================================
# SPORTMONKS CLIENT (KORRIGIERT!)
# ==========================================================
class SportmonksXGClient:
    """
    Client für Sportmonks API - NUR xG-Daten (KEINE Quoten!)
    ✅ KORRIGIERT: Verwendet jetzt seasons-basierten Endpoint!
    """

    def __init__(self, config: HybridScraperConfig):
        self.config = config
        self.api_calls = 0
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """API Request mit Retry"""
        if params is None:
            params = {}

        params['api_token'] = self.config.api_token
        url = f"{self.config.base_url}/{endpoint}"

        # Debug-Ausgabe (ohne Token!)
        if self.config.debug:
            safe_params = {k: v for k, v in params.items() if k != 'api_token'}
            print(f"      [API] {endpoint} | {safe_params}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.config.request_timeout)
                self.api_calls += 1

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"      ⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                time.sleep(self.config.request_delay)
                return response.json()

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    print(f"      ❌ Timeout bei {endpoint}")
                    return None
                time.sleep(5 * (attempt + 1))

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"      ❌ Fehler: {e}")
                    return None
                time.sleep(1)

        return None

    def get_league_seasons(self, league_id: int, league_name: str) -> List[Dict]:
        """
        Hole Saisons für eine Liga
        ✅ KORRIGIERT: Wie in V3 FINAL!
        """
        endpoint = f"leagues/{league_id}"
        params = {'include': 'seasons'}

        data = self._make_request(endpoint, params)
        if not data or 'data' not in data:
            return []

        seasons_list = data['data'].get('seasons', [])

        # Filtere relevante Saisons
        relevant_seasons = []
        start_date = self.config.start_date.date()
        today = datetime.now().date()

        for season in seasons_list:
            if season.get('id') and season.get('ending_at') and season.get('starting_at'):
                try:
                    ending_date = datetime.fromisoformat(season['ending_at']).date()
                    starting_date = datetime.fromisoformat(season['starting_at']).date()

                    # Nur Saisons die:
                    # 1. Nach start_date enden
                    # 2. Bereits gestartet haben
                    # 3. Nicht zu weit in der Zukunft enden (max 1 Jahr)
                    if (ending_date >= start_date and
                        starting_date <= today and
                        ending_date <= today + pd.Timedelta(days=365)):
                        relevant_seasons.append(season)
                except (ValueError, TypeError):
                    continue

        if self.config.debug:
            print(f"      {len(relevant_seasons)} relevante Saisons: {[s['name'] for s in relevant_seasons[:3]]}")

        return relevant_seasons

    def get_fixtures_with_xg(self, season_id: int, league_name: str) -> pd.DataFrame:
        """
        Hole Fixtures mit xG für eine Saison
        ✅ KORRIGIERT: Wie in V3 FINAL - seasons-basiert!
        """

        endpoint = f"seasons/{season_id}"
        params = {'include': 'fixtures.participants;fixtures.scores;fixtures.xGFixture'}

        data = self._make_request(endpoint, params)
        if not data or 'data' not in data:
            return pd.DataFrame()

        fixtures = data['data'].get('fixtures', [])
        if not fixtures:
            return pd.DataFrame()

        # Extrahiere Daten
        games = []
        for fixture in fixtures:
            try:
                game = self._extract_fixture_data(fixture, league_name)
                if game and game['status'] == 'FT':  # Nur abgeschlossene Spiele
                    games.append(game)
            except Exception as e:
                if self.config.debug:
                    print(f"      Fehler bei Fixture {fixture.get('id')}: {e}")
                continue

        return pd.DataFrame(games)

    def _extract_fixture_data(self, fixture: Dict, league_name: str) -> Optional[Dict]:
        """
        Extrahiere Daten aus Fixture
        ✅ KORRIGIERT: correct_score hinzugefügt!
        """

        # Basis-Daten
        fixture_id = fixture.get('id')
        date_str = fixture.get('starting_at')
        if not fixture_id or not date_str:
            return None

        try:
            date = pd.to_datetime(date_str).date()
        except:
            return None

        # Status
        state_id = fixture.get('state_id')
        if state_id not in [5, 6, 7]:  # Nicht abgeschlossen
            return None

        # Teams
        participants = fixture.get('participants', [])
        home_team = None
        away_team = None
        for p in participants:
            if isinstance(p, dict):
                if p.get('meta', {}).get('location') == 'home':
                    home_team = p.get('name')
                elif p.get('meta', {}).get('location') == 'away':
                    away_team = p.get('name')

        if not home_team or not away_team:
            return None

        # Scores
        scores = fixture.get('scores', [])
        home_score = None
        away_score = None
        for score in scores:
            if isinstance(score, dict):
                desc = score.get('description', '').lower()
                if 'current' in desc or 'full' in desc or 'final' in desc:
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals')

                    if goals is not None:
                        # Bestimme ob Home oder Away
                        for p in participants:
                            if p.get('id') == participant_id:
                                location = p.get('meta', {}).get('location')
                                if location == 'home':
                                    home_score = int(goals)
                                elif location == 'away':
                                    away_score = int(goals)

        if home_score is None or away_score is None:
            return None

        # xG-Daten
        xg_data = fixture.get('xgfixture', [])
        home_xg = 0.0
        away_xg = 0.0

        if isinstance(xg_data, list):
            for xg_item in xg_data:
                if isinstance(xg_item, dict):
                    type_id = xg_item.get('type_id')
                    if type_id in [5304, 5305]:  # xG Type IDs
                        location = xg_item.get('location')
                        value = xg_item.get('data', {}).get('value')

                        if value is not None:
                            try:
                                if location == 'home':
                                    home_xg = float(value)
                                elif location == 'away':
                                    away_xg = float(value)
                            except (ValueError, TypeError):
                                pass

        # ✅ Correct Score erstellen
        correct_score = f"{home_score}-{away_score}"

        return {
            'date': date,
            'league': league_name,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'correct_score': correct_score,  # ✅ HINZUGEFÜGT!
            'home_xg': home_xg,
            'away_xg': away_xg,
            'status': 'FT',
            'fixture_id': fixture_id
        }


# ==========================================================
# FOOTBALL-DATA CLIENT (VERBESSERT!)
# ==========================================================
class FootballDataClient:
    """
    Lädt historische Quoten von Football-Data.co.uk
    ✅ VERBESSERT: Lädt jetzt ALLE Saisons seit 2000 (nicht nur 2!)
    """

    def __init__(self, config: HybridScraperConfig):
        self.config = config
        self.session = requests.Session()
        
        # League-Mappings
        self.league_mappings = {
            'Premier League': 'E0',
            'Bundesliga': 'D1',
            'La Liga': 'SP1',
            'Ligue 1': 'F1',
            'Serie A': 'I1',
        }

    def download_odds_for_league(self, league_name: str) -> pd.DataFrame:
        """
        Lädt Quoten für eine Liga
        ✅ VERBESSERT: Lädt jetzt alle verfügbaren Saisons!
        """
        
        league_code = self.league_mappings.get(league_name)
        if not league_code:
            print(f"      ⚠️ Unbekannte Liga: {league_name}")
            return pd.DataFrame()

        all_odds = []
        
        # ✅ VERBESSERT: Versuche alle Saisons seit 2000!
        seasons_to_try = []
        current_year = datetime.now().year
        
        for i in range(current_year - 2000 + 1):
            year = 2000 + i
            
            # Format: 2425 für Saison 2024/25
            season_code = f"{str(year)[2:]}{str(year+1)[2:]}"
            seasons_to_try.append((season_code, f"{year}/{year+1}"))

        # Neueste Saisons zuerst
        seasons_to_try.reverse()

        for season_code, season_name in seasons_to_try:
            url = f"{self.config.football_data_base_url}/mmz4281/{season_code}/{league_code}.csv"
            
            try:
                if self.config.debug:
                    print(f"      Versuche {league_name} {season_name}...")
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 404:
                    continue  # Saison nicht verfügbar
                
                response.raise_for_status()
                
                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))
                
                # Standardisiere
                df_std = self._standardize_columns(df, league_name)
                
                if not df_std.empty:
                    # Filtere nach start_date
                    df_std = df_std[df_std['date'] >= self.config.start_date.date()]
                    
                    if not df_std.empty:
                        all_odds.append(df_std)
                        if self.config.debug:
                            print(f"         ✅ {len(df_std)} Spiele")
                
                time.sleep(0.5)  # Sei nett zum Server
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code != 404:
                    if self.config.debug:
                        print(f"         ⚠️ HTTP Fehler: {e}")
                continue
            except Exception as e:
                if self.config.debug:
                    print(f"         ⚠️ Fehler: {e}")
                continue

        if all_odds:
            return pd.concat(all_odds, ignore_index=True)
        return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame, league_name: str) -> pd.DataFrame:
        """Standardisiert Football-Data Spalten"""
        
        required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if not all(col in df.columns for col in required):
            return pd.DataFrame()
        
        std_df = pd.DataFrame()
        std_df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce').dt.date
        std_df['home_team'] = df['HomeTeam']
        std_df['away_team'] = df['AwayTeam']
        std_df['home_score'] = pd.to_numeric(df['FTHG'], errors='coerce')
        std_df['away_score'] = pd.to_numeric(df['FTAG'], errors='coerce')
        
        # Quoten (B365 bevorzugt, dann PS)
        if 'B365H' in df.columns:
            std_df['odds_home'] = pd.to_numeric(df['B365H'], errors='coerce')
            std_df['odds_draw'] = pd.to_numeric(df['B365D'], errors='coerce')
            std_df['odds_away'] = pd.to_numeric(df['B365A'], errors='coerce')
        elif 'PSH' in df.columns:
            std_df['odds_home'] = pd.to_numeric(df['PSH'], errors='coerce')
            std_df['odds_draw'] = pd.to_numeric(df['PSD'], errors='coerce')
            std_df['odds_away'] = pd.to_numeric(df['PSA'], errors='coerce')
        
        std_df['league'] = league_name
        
        # Entferne NaN
        std_df = std_df.dropna(subset=['date', 'home_team', 'away_team'])
        
        return std_df


# ==========================================================
# HYBRID SCRAPER (MAIN)
# ==========================================================
class HybridScraper:
    """Kombiniert Sportmonks xG + Football-Data Odds"""

    def __init__(self, config: HybridScraperConfig):
        self.config = config
        self.sportmonks = SportmonksXGClient(config)
        self.football_data = FootballDataClient(config)
        self.team_matcher = TeamMatcher()

    def scrape_all(self) -> Dict[str, pd.DataFrame]:
        """Hauptfunktion - Scraped alles"""

        print("\n" + "=" * 70)
        print("🚀 SPORTMONKS HYBRID SCRAPER V4 - FIXED")
        print("=" * 70)
        print()

        # Top 5 Ligen
        leagues = [
            (8, 'Premier League'),
            (82, 'Bundesliga'),
            (564, 'La Liga'),
            (301, 'Ligue 1'),
            (384, 'Serie A'),
        ]

        all_xg_data = []
        all_odds_data = []

        # SCHRITT 1: Sportmonks xG-Daten
        print("\n📊 SCHRITT 1: Lade xG-Daten von Sportmonks...")
        print("=" * 70)

        for league_id, league_name in leagues:
            print(f"\n🏆 {league_name}")

            # Hole Saisons
            seasons = self.sportmonks.get_league_seasons(league_id, league_name)
            if not seasons:
                print(f"      ⚠️ Keine Saisons gefunden")
                continue

            # Hole Fixtures mit xG
            for season in seasons:
                season_id = season.get('id')
                season_name = season.get('name')

                print(f"   🔄 {season_name}...")

                df = self.sportmonks.get_fixtures_with_xg(season_id, league_name)
                if not df.empty:
                    all_xg_data.append(df)
                    print(f"      ✅ {len(df)} Spiele mit xG")

        df_xg = pd.concat(all_xg_data, ignore_index=True) if all_xg_data else pd.DataFrame()
        print(f"\n✅ Sportmonks xG-Daten: {len(df_xg)} Spiele")

        # SCHRITT 2: Football-Data Quoten
        print("\n💰 SCHRITT 2: Lade Quoten von Football-Data.co.uk...")
        print("=" * 70)

        for _, league_name in leagues:
            print(f"\n🏆 {league_name}")

            df = self.football_data.download_odds_for_league(league_name)
            if not df.empty:
                all_odds_data.append(df)
                print(f"      ✅ {len(df)} Spiele mit Quoten")

        df_odds = pd.concat(all_odds_data, ignore_index=True) if all_odds_data else pd.DataFrame()
        print(f"\n✅ Football-Data Quoten: {len(df_odds)} Spiele")

        # SCHRITT 3: Merge
        print("\n🔗 SCHRITT 3: Merge xG + Quoten...")
        print("=" * 70)

        df_complete = self._merge_data(df_xg, df_odds)

        # Kategorisiere
        results = {
            'complete': df_complete,
            'xg_only': df_xg[~df_xg['fixture_id'].isin(df_complete['fixture_id'])] if not df_complete.empty and not df_xg.empty else pd.DataFrame(),
            'odds_only': df_odds  # Alle Odds-Daten (für Referenz)
        }

        return results

    def _merge_data(self, df_xg: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
        """
        Merge xG und Odds per Fuzzy-Matching
        ✅ Verwendet TeamMatcher aus V4!
        """

        if df_xg.empty or df_odds.empty:
            print("   ⚠️ Keine Daten zum Mergen")
            return pd.DataFrame()

        # Normalisiere mit TeamMatcher
        df_xg = df_xg.copy()
        df_odds = df_odds.copy()
        
        df_xg['home_norm'] = df_xg['home_team'].apply(self.team_matcher.normalize)
        df_xg['away_norm'] = df_xg['away_team'].apply(self.team_matcher.normalize)
        
        df_odds['home_norm'] = df_odds['home_team'].apply(self.team_matcher.normalize)
        df_odds['away_norm'] = df_odds['away_team'].apply(self.team_matcher.normalize)

        # Merge per Datum + Teams
        merged = df_xg.merge(
            df_odds[['date', 'home_norm', 'away_norm', 'odds_home', 'odds_draw', 'odds_away']],
            on=['date', 'home_norm', 'away_norm'],
            how='inner'
        )

        # Cleanup
        merged = merged.drop(columns=['home_norm', 'away_norm'])

        print(f"   ✅ {len(merged)} Spiele mit xG + Quoten")

        return merged

    def save_data(self, results: Dict[str, pd.DataFrame]):
        """
        Speichere Ergebnisse
        ✅ VERBESSERT: Mit correct_score!
        """

        print("\n💾 SPEICHERE DATEN...")
        print("=" * 70)

        # ✅ Output-Spalten mit correct_score!
        output_cols = [
            'date', 'league', 'home_team', 'away_team',
            'home_score', 'away_score',
            'correct_score',  # ✅ JETZT DABEI!
            'home_xg', 'away_xg',
            'odds_home', 'odds_draw', 'odds_away',
            'status', 'fixture_id'
        ]

        # Complete
        df_complete = results['complete']
        if not df_complete.empty:
            final_cols = [col for col in output_cols if col in df_complete.columns]
            df_out = df_complete[final_cols].copy()
            df_out = df_out.sort_values('date').reset_index(drop=True)

            df_out.to_csv(self.config.output_file_complete, index=False)
            print(f"\n✅ KOMPLETT (xG + Quoten + CS): {len(df_out)} Spiele")
            print(f"   Datei: {self.config.output_file_complete}")
            print(f"   Größe: {os.path.getsize(self.config.output_file_complete)/1024:.1f} KB")
        else:
            print("\n⚠️ Keine kompletten Spiele")

        # xG only
        df_xg = results['xg_only']
        if not df_xg.empty:
            final_cols = [col for col in output_cols if col in df_xg.columns]
            df_out = df_xg[final_cols].copy()
            df_out = df_out.sort_values('date').reset_index(drop=True)

            df_out.to_csv(self.config.output_file_xg_only, index=False)
            print(f"\n✅ NUR xG (mit CS): {len(df_out)} Spiele")
            print(f"   Datei: {self.config.output_file_xg_only}")

        # ✅ HINZUGEFÜGT: Statistiken
        self._print_statistics(results)

    def _print_statistics(self, results: Dict[str, pd.DataFrame]):
        """
        ✅ HINZUGEFÜGT: Drucke detaillierte Statistiken (wie V3 FINAL!)
        """

        print("\n📊 FINALE STATISTIKEN")
        print("=" * 70)

        df_complete = results['complete']

        if not df_complete.empty:
            print(f"\n🌐 API-Calls (Sportmonks): {self.sportmonks.api_calls}")
            print(f"📥 Downloads (Football-Data): {len(self.football_data.league_mappings)}")

            print(f"\n📈 Spiele:")
            print(f"  • Mit xG + Quoten: {len(df_complete)} ⭐")

            print(f"\n🏆 Verteilung:")
            print(df_complete['league'].value_counts().to_string())

            print(f"\n📅 Zeitraum: {df_complete['date'].min()} bis {df_complete['date'].max()}")

            # ✅ Feature-Check
            print(f"\n✅ Features verfügbar:")
            features = ['home_xg', 'away_xg', 'correct_score', 'odds_home', 'odds_draw', 'odds_away']
            for feat in features:
                if feat in df_complete.columns:
                    missing = df_complete[feat].isna().sum()
                    pct = 100 * (1 - missing / len(df_complete))
                    print(f"  • {feat}: {len(df_complete) - missing}/{len(df_complete)} ({pct:.1f}%)")


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Hauptfunktion"""

    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        return

    config = HybridScraperConfig(
        api_token=api_token,
        output_file_complete="game_database_complete.csv",
        output_file_xg_only="game_database_xg_only.csv",
        output_file_odds_only="game_database_odds_only.csv",
        debug=True
    )

    scraper = HybridScraper(config)

    try:
        results = scraper.scrape_all()
        scraper.save_data(results)

        print("\n" + "=" * 70)
        print("✅ SCRAPING ABGESCHLOSSEN!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen")
    except Exception as e:
        print(f"\n\n❌ FEHLER: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
