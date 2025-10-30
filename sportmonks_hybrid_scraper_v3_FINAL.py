#!/usr/bin/env python3
"""
HYBRID SCRAPER v3.0 - SPORTMONKS xG + FOOTBALL-DATA.CO.UK ODDS
================================================================
Die FINALE Lösung für ML-Training Daten!

Warum dieser Ansatz?
--------------------
Sportmonks API speichert KEINE historischen Pre-Match Odds.
Lösung: Kombiniere 2 Quellen:
  1. Sportmonks: xG-Daten (mit API)
  2. Football-Data.co.uk: Historische Quoten (kostenlos!)

Was macht dieser Scraper?
--------------------------
1. Lädt xG-Daten von Sportmonks (wie bisher)
2. Downloaded historische Quoten von Football-Data.co.uk (CSV)
3. Merged beide Datenquellen per Datum + Teams
4. Erstellt perfekte Trainings-Datenbank!

Output:
-------
- game_database_complete.csv (xG + Odds) ← FÜR ML-TRAINING!
- game_database_xg_only.csv (nur xG)
- game_database_odds_only.csv (nur Odds)

Version: 3.0 FINAL
Datum: 2025-10-30
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from dataclasses import dataclass
import traceback
import io

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
# SPORTMONKS CLIENT (NUR xG)
# ==========================================================
class SportmonksXGClient:
    """Client für Sportmonks API - NUR xG-Daten (KEINE Quoten!)"""

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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.config.request_timeout)
                self.api_calls += 1

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                time.sleep(self.config.request_delay)
                return response.json()

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    print(f"❌ Timeout bei {endpoint}")
                    return None
                time.sleep(5 * (attempt + 1))

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ Fehler: {e}")
                    return None
                time.sleep(1)

        return None

    def get_league_seasons(self, league_id: int, league_name: str) -> List[Dict]:
        """Hole Saisons für eine Liga"""
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
            print(f"   {len(relevant_seasons)} relevante Saisons: {[s['name'] for s in relevant_seasons[:3]]}")

        return relevant_seasons

    def get_fixtures_with_xg(self, season_id: int, league_name: str) -> pd.DataFrame:
        """Hole Fixtures mit xG für eine Saison"""

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
                    print(f"   Fehler bei Fixture {fixture.get('id')}: {e}")
                continue

        return pd.DataFrame(games)

    def _extract_fixture_data(self, fixture: Dict, league_name: str) -> Optional[Dict]:
        """Extrahiere Daten aus Fixture"""

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
                if 'current' in desc or 'full' in desc:
                    participant_id = score.get('participant_id')
                    goals = score.get('score', {}).get('goals')

                    if goals is not None and participants:
                        if participants[0].get('id') == participant_id:
                            home_score = int(goals)
                        elif len(participants) > 1 and participants[1].get('id') == participant_id:
                            away_score = int(goals)

        # xG-Daten
        xg_data = fixture.get('xgfixture', [])
        home_xg = 0.0
        away_xg = 0.0

        if isinstance(xg_data, list):
            for xg_item in xg_data:
                if isinstance(xg_item, dict) and xg_item.get('type_id') == 5304:
                    location = xg_item.get('location')
                    value = xg_item.get('data', {}).get('value')

                    if value is not None:
                        try:
                            if location == 'home':
                                home_xg = float(value)
                            elif location == 'away':
                                away_xg = float(value)
                        except:
                            pass

        return {
            'fixture_id': fixture_id,
            'date': date,
            'league': league_name,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'status': 'FT'
        }


# ==========================================================
# FOOTBALL-DATA.CO.UK CLIENT
# ==========================================================
class FootballDataClient:
    """Client für Football-Data.co.uk (historische Quoten)"""

    def __init__(self, config: HybridScraperConfig):
        self.config = config
        self.session = requests.Session()

        # Liga-Mappings: Sportmonks Name → Football-Data Code
        self.league_mappings = {
            'Premier League': {
                'code': 'E0',
                'seasons': {
                    '2023/2024': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
                    '2024/2025': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
                }
            },
            'Bundesliga': {
                'code': 'D1',
                'seasons': {
                    '2023/2024': 'https://www.football-data.co.uk/mmz4281/2324/D1.csv',
                    '2024/2025': 'https://www.football-data.co.uk/mmz4281/2425/D1.csv',
                }
            },
            'La Liga': {
                'code': 'SP1',
                'seasons': {
                    '2023/2024': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
                    '2024/2025': 'https://www.football-data.co.uk/mmz4281/2425/SP1.csv',
                }
            },
            'Ligue 1': {
                'code': 'F1',
                'seasons': {
                    '2023/2024': 'https://www.football-data.co.uk/mmz4281/2324/F1.csv',
                    '2024/2025': 'https://www.football-data.co.uk/mmz4281/2425/F1.csv',
                }
            },
            'Serie A': {
                'code': 'I1',
                'seasons': {
                    '2023/2024': 'https://www.football-data.co.uk/mmz4281/2324/I1.csv',
                    '2024/2025': 'https://www.football-data.co.uk/mmz4281/2425/I1.csv',
                }
            },
        }

    def download_odds_for_league(self, league_name: str) -> pd.DataFrame:
        """Download historische Quoten für eine Liga"""

        if league_name not in self.league_mappings:
            return pd.DataFrame()

        league_config = self.league_mappings[league_name]
        all_odds = []

        for season_name, url in league_config['seasons'].items():
            try:
                print(f"   Downloading {league_name} {season_name} von Football-Data.co.uk...")

                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))

                # Standardisiere Spaltennamen
                df = self._standardize_columns(df, league_name)

                all_odds.append(df)

                time.sleep(1)  # Sei nett zum Server

            except Exception as e:
                print(f"   ⚠️ Fehler bei {league_name} {season_name}: {e}")
                continue

        if all_odds:
            return pd.concat(all_odds, ignore_index=True)
        return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame, league_name: str) -> pd.DataFrame:
        """Standardisiere Spaltennamen"""

        # Erforderliche Spalten
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

        # Prüfe ob vorhanden
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Extrahiere relevante Spalten
        standard_df = pd.DataFrame()
        standard_df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce').dt.date
        standard_df['home_team'] = df['HomeTeam']
        standard_df['away_team'] = df['AwayTeam']
        standard_df['home_score'] = pd.to_numeric(df['FTHG'], errors='coerce')
        standard_df['away_score'] = pd.to_numeric(df['FTAG'], errors='coerce')

        # Quoten (Bet365 als Standard)
        if 'B365H' in df.columns:
            standard_df['odds_home'] = pd.to_numeric(df['B365H'], errors='coerce')
        if 'B365D' in df.columns:
            standard_df['odds_draw'] = pd.to_numeric(df['B365D'], errors='coerce')
        if 'B365A' in df.columns:
            standard_df['odds_away'] = pd.to_numeric(df['B365A'], errors='coerce')

        standard_df['league'] = league_name

        # Entferne Zeilen mit fehlenden Daten
        standard_df = standard_df.dropna(subset=['date', 'home_team', 'away_team'])

        return standard_df


# ==========================================================
# HYBRID SCRAPER
# ==========================================================
class HybridScraper:
    """Kombiniert Sportmonks xG + Football-Data Odds"""

    def __init__(self, config: HybridScraperConfig):
        self.config = config
        self.sportmonks = SportmonksXGClient(config)
        self.football_data = FootballDataClient(config)

    def scrape_all(self) -> Dict[str, pd.DataFrame]:
        """Scrape alle Daten"""

        print("\n" + "=" * 70)
        print("🚀 HYBRID SCRAPER v3.0 - Sportmonks xG + Football-Data Odds")
        print("=" * 70)

        # Ligen
        leagues = [
            (8, 'Premier League'),
            (82, 'Bundesliga'),
            (564, 'La Liga'),
            (301, 'Ligue 1'),
            # (384, 'Serie A'),  # Optional
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
                print(f"   ⚠️ Keine Saisons gefunden")
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
                print(f"   ✅ {len(df)} Spiele mit Quoten")

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
        """Merge xG und Odds per Fuzzy-Matching"""

        if df_xg.empty or df_odds.empty:
            print("   ⚠️ Keine Daten zum Mergen")
            return pd.DataFrame()

        # Team-Name-Normalisierung
        def normalize_team(name):
            """Normalisiere Teamnamen für besseres Matching"""
            if pd.isna(name):
                return ""
            name = str(name).lower()
            replacements = {
                'manchester united': 'man united',
                'manchester city': 'man city',
                'tottenham hotspur': 'tottenham',
                'newcastle united': 'newcastle',
                'west ham united': 'west ham',
                'wolverhampton wanderers': 'wolves',
                'brighton and hove albion': 'brighton',
                'nottingham forest': "nott'm forest",
            }
            for full, short in replacements.items():
                if full in name:
                    return short
            return name

        # Normalisiere
        df_xg['home_norm'] = df_xg['home_team'].apply(normalize_team)
        df_xg['away_norm'] = df_xg['away_team'].apply(normalize_team)
        df_odds['home_norm'] = df_odds['home_team'].apply(normalize_team)
        df_odds['away_norm'] = df_odds['away_team'].apply(normalize_team)

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
        """Speichere Ergebnisse"""

        print("\n💾 SPEICHERE DATEN...")
        print("=" * 70)

        output_cols = [
            'date', 'league', 'home_team', 'away_team',
            'home_score', 'away_score',
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
            print(f"\n✅ KOMPLETT (xG + Quoten): {len(df_out)} Spiele")
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
            print(f"\n✅ NUR xG: {len(df_out)} Spiele")
            print(f"   Datei: {self.config.output_file_xg_only}")

        # Statistiken
        self._print_statistics(results)

    def _print_statistics(self, results: Dict[str, pd.DataFrame]):
        """Drucke Statistiken"""

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

            # Feature-Check
            print(f"\n✅ Features verfügbar:")
            features = ['home_xg', 'away_xg', 'odds_home', 'odds_draw', 'odds_away']
            for feat in features:
                if feat in df_complete.columns:
                    missing = df_complete[feat].isna().sum()
                    print(f"  • {feat}: {len(df_complete) - missing}/{len(df_complete)} ({100*(1-missing/len(df_complete)):.1f}%)")


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
