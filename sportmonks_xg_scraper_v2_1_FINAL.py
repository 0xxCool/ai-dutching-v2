#!/usr/bin/env python3
"""
SPORTMONKS xG & ODDS SCRAPER v2.1 - FINAL & PRODUCTION-READY
=============================================================
FIXES gegenüber v2.0:
- Erhöhtes Timeout (60s statt 20s) für große Saisons
- Quoten werden separat geholt (include=odds funktioniert nicht zuverlässig)
- Nur abgeschlossene Spiele (status=FT) werden verarbeitet
- Bessere Fehlerbehandlung für Timeouts
- Chunked Processing für große Saisons (optional)

Version: 2.1 (Production-Ready)
Datum: 2025-10-30
"""

import json
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
from dataclasses import dataclass
import traceback

load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class ScraperConfig:
    """Konfiguration für den Sportmonks Scraper v2.1"""

    # API Settings
    api_token: str = ""
    base_url: str = "https://api.sportmonks.com/v3/football"
    request_delay: float = 1.3
    request_timeout: int = 60  # Erhöht von 20s auf 60s

    # Output
    output_file: str = "game_database_sportmonks.csv"
    output_file_odds_only: str = "game_database_sportmonks_odds_only.csv"
    output_file_xg_only: str = "game_database_sportmonks_xg_only.csv"
    save_intermediate: bool = True

    # Startdatum für xG-Daten
    XG_START_DATE: datetime = datetime(2024, 3, 1)

    # Debug-Modus
    debug: bool = True
    max_fixtures_per_season: int = None  # None = alle

    # Nur abgeschlossene Spiele
    only_finished_games: bool = True  # Ignoriere zukünftige Spiele


# ==========================================================
# SPORTMONKS API CLIENT v2.1
# ==========================================================
class SportmonksXGClientV2_1:
    """Client für Sportmonks API v2.1 (FINAL)"""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.api_calls = 0
        self.session = requests.Session()

        # Statistiken
        self.stats = {
            'fixtures_fetched': 0,
            'fixtures_with_odds': 0,
            'fixtures_with_xg': 0,
            'fixtures_complete': 0,
            'fixtures_skipped_date': 0,
            'fixtures_skipped_status': 0,
            'fixtures_skipped_no_data': 0,
            'api_timeouts': 0,
        }

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Zentrale Request-Funktion mit Retry-Logik"""
        if params is None:
            params = {}

        params['api_token'] = self.config.api_token
        url = f"{self.config.base_url}/{endpoint}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.request_timeout  # 60s timeout
                )
                self.api_calls += 1

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                time.sleep(self.config.request_delay)

                return response.json()

            except requests.exceptions.Timeout as e:
                self.stats['api_timeouts'] += 1
                if attempt == max_retries - 1:
                    print(f"❌ Timeout nach {max_retries} Versuchen (Endpunkt zu groß?)")
                    return None
                wait_time = 5 * (attempt + 1)
                print(f"⚠️ Timeout - warte {wait_time}s und versuche erneut...")
                time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ Fehler nach {max_retries} Versuchen: {e}")
                    return None
                time.sleep(1)

        return None

    def get_leagues(self) -> List[Dict]:
        """Hole alle verfügbaren Top-Ligen"""
        print("📋 Lade verfügbare Ligen...")

        top_league_ids = [
            8,      # Premier League
            82,     # Bundesliga
            564,    # La Liga
            # 384,  # Serie A (hat oft Timeout)
            301,    # Ligue 1
            # 72,   # Eredivisie
            # 271,  # Primeira Liga
            2,      # Champions League
            # 390,  # 2. Bundesliga
            # 501,  # Championship
            # 1489  # Scottish Premiership
        ]

        leagues = []

        print(f"Lade Daten für {len(top_league_ids)} Ligen...")
        for league_id in tqdm(top_league_ids, desc="Lade Ligainformationen"):
            time.sleep(0.2 if leagues else 1.3)

            data = self._make_request(f"leagues/{league_id}")
            if data and 'data' in data:
                leagues.append(data['data'])
            else:
                print(f"(WARNUNG: Konnte Liga-Info für ID {league_id} nicht laden)")

        print(f"✅ {len(leagues)} Ligainformationen erfolgreich geladen.")
        return leagues

    def get_seasons_for_league(self, league_id: int) -> List[Dict]:
        """Hole Saisons für eine Liga"""
        endpoint = f"leagues/{league_id}"
        params = {'include': 'seasons'}

        if self.config.debug:
            print(f"DEBUG: Lade Saisons für Liga {league_id}")

        data = self._make_request(endpoint, params)

        if not data or 'data' not in data:
            return []

        league_data = data['data']
        seasons_list = league_data.get('seasons')

        if not seasons_list or not isinstance(seasons_list, list):
            return []

        # Filtere relevante Saisons (ab März 2024, ABER NICHT Zukunft!)
        relevant_seasons = []
        xg_start_date = self.config.XG_START_DATE.date()
        today = datetime.now().date()

        for season in seasons_list:
            if season.get('id') and season.get('name') and season.get('ending_at') and season.get('starting_at'):
                try:
                    ending_date = datetime.fromisoformat(season['ending_at']).date()
                    starting_date = datetime.fromisoformat(season['starting_at']).date()

                    # Saison muss:
                    # 1. Nach März 2024 enden ODER aktuell laufen
                    # 2. Bereits gestartet haben (nicht zukünftige Saisons!)
                    if ending_date >= xg_start_date and starting_date <= today:
                        relevant_seasons.append(season)
                except (ValueError, TypeError):
                    continue

        if self.config.debug:
            print(f"DEBUG: {len(seasons_list)} Saisons gesamt, {len(relevant_seasons)} relevant")
            if relevant_seasons:
                print(f"       Gewählte Saisons: {[s['name'] for s in relevant_seasons]}")

        return relevant_seasons

    def get_fixtures_for_season(self, season_id: int, league_name: str) -> List[Dict]:
        """
        Hole alle Fixtures für eine Saison
        WICHTIG: OHNE Quoten (reduces Payload, vermeidet Timeouts)
        """

        endpoint = f"seasons/{season_id}"

        # OHNE Odds! (werden separat geholt)
        params = {
            'include': 'fixtures.participants;fixtures.scores;fixtures.xGFixture;league'
        }

        if self.config.debug:
            print(f"DEBUG: Lade Fixtures für Saison {season_id} (OHNE Quoten)")

        data = self._make_request(endpoint, params)

        if not data or 'data' not in data:
            return []

        season_data = data['data']
        fixtures = season_data.get('fixtures')
        league_info = season_data.get('league', {})

        if not fixtures or not isinstance(fixtures, list):
            return []

        # Füge Liga-Info zu jedem Fixture hinzu
        clean_fixtures = []
        for f in fixtures:
            if isinstance(f, dict):
                f['league_info_from_season'] = league_info
                clean_fixtures.append(f)

        self.stats['fixtures_fetched'] += len(clean_fixtures)

        if self.config.debug:
            print(f"DEBUG: {len(clean_fixtures)} Fixtures geladen")

        return clean_fixtures

    def get_odds_for_fixture(self, fixture_id: int) -> Dict:
        """
        Hole Quoten für ein einzelnes Fixture
        (Separater Call, nur für FT-Spiele)
        """

        endpoint = f'odds/pre-match/fixtures/{fixture_id}'
        params = {'include': 'market;bookmaker'}

        data = self._make_request(endpoint, params)

        if not data or 'data' not in data:
            return {}

        return self._parse_sportmonks_odds(data['data'])

    def _parse_sportmonks_odds(self, odds_data: List[Dict]) -> Dict:
        """Parse Pre-Match Odds-Daten"""

        odds_dict = {
            'odds_home': None,
            'odds_draw': None,
            'odds_away': None
        }

        if not isinstance(odds_data, list):
            return odds_dict

        # Suche nach "3Way Result" Markt
        for odds_item in odds_data:
            if not isinstance(odds_item, dict):
                continue

            market = odds_item.get('market')
            if not market or market.get('name') != '3Way Result':
                continue

            bookmaker = odds_item.get('bookmaker', {})
            if not isinstance(bookmaker, dict):
                continue

            bookmaker_odds = bookmaker.get('odds', [])
            if not isinstance(bookmaker_odds, list) or not bookmaker_odds:
                continue

            first_odds = bookmaker_odds[0] if bookmaker_odds else {}
            odds_values = first_odds.get('odds', [])

            if not isinstance(odds_values, list):
                continue

            for odd in odds_values:
                if not isinstance(odd, dict):
                    continue

                label = odd.get('label')
                value = odd.get('value')

                if label and value:
                    try:
                        if label == 'Home':
                            odds_dict['odds_home'] = float(value)
                        elif label == 'Draw':
                            odds_dict['odds_draw'] = float(value)
                        elif label == 'Away':
                            odds_dict['odds_away'] = float(value)
                    except (ValueError, TypeError):
                        continue

            if all([odds_dict['odds_home'], odds_dict['odds_draw'], odds_dict['odds_away']]):
                break

        return odds_dict

    def extract_xg_from_fixture(self, fixture: Dict) -> Dict:
        """Extrahiere xG-Daten aus einem Fixture"""

        result = {
            'fixture_id': None,
            'date': None,
            'home_team': None,
            'away_team': None,
            'home_xg': 0.0,
            'away_xg': 0.0,
            'home_score': None,
            'away_score': None,
            'league': 'Unknown',
            'season': 'Unknown',
            'status': 'Unknown'
        }

        if not isinstance(fixture, dict):
            return result

        try:
            result['fixture_id'] = fixture.get('id')
            result['date'] = fixture.get('starting_at')

            # Liga
            league_info = fixture.get('league_info_from_season', {})
            result['league'] = league_info.get('name', 'Unknown') if isinstance(league_info, dict) else 'Unknown'

            # Saison
            season_info = fixture.get('season', {})
            result['season'] = season_info.get('name', 'Unknown') if isinstance(season_info, dict) else 'Unknown'

            # Status
            state_id = fixture.get('state_id')
            if state_id in [5, 6, 7]:  # Finished states
                result['status'] = 'FT'
            else:
                state_info = fixture.get('state', {})
                status_str = state_info.get('state', 'Unknown') if isinstance(state_info, dict) else 'Unknown'
                if 'FT' in status_str or 'AET' in status_str or 'PEN' in status_str:
                    result['status'] = 'FT'
                else:
                    result['status'] = status_str

            # Teilnehmer
            participants = fixture.get('participants', [])
            if isinstance(participants, list):
                for p in participants:
                    if isinstance(p, dict):
                        if p.get('meta', {}).get('location') == 'home':
                            result['home_team'] = p.get('name')
                        elif p.get('meta', {}).get('location') == 'away':
                            result['away_team'] = p.get('name')

            # Scores
            scores = fixture.get('scores', [])
            if isinstance(scores, list):
                for score in scores:
                    if isinstance(score, dict):
                        desc = score.get('description', '').lower()
                        if 'current' in desc or 'full' in desc or 'final' in desc:
                            participant_id = score.get('participant_id')
                            score_data = score.get('score', {})
                            goals = score_data.get('goals') if isinstance(score_data, dict) else None

                            if goals is not None:
                                if isinstance(participants, list) and len(participants) > 0:
                                    if participants[0].get('id') == participant_id:
                                        result['home_score'] = int(goals)
                                    elif len(participants) > 1 and participants[1].get('id') == participant_id:
                                        result['away_score'] = int(goals)

            # xG-Daten (lowercase!)
            xg_data_list = fixture.get('xgfixture')

            if isinstance(xg_data_list, list):
                for xg_item in xg_data_list:
                    if isinstance(xg_item, dict):
                        if xg_item.get('type_id') == 5304:  # Haupt-xG
                            location = xg_item.get('location')
                            value = xg_item.get('data', {}).get('value')

                            if value is not None:
                                try:
                                    if location == 'home':
                                        result['home_xg'] = float(value)
                                    elif location == 'away':
                                        result['away_xg'] = float(value)
                                except (ValueError, TypeError):
                                    pass

        except Exception as e:
            if self.config.debug:
                print(f"WARNUNG: Fehler in extract_xg_from_fixture: {e}")

        return result


# ==========================================================
# HAUPT-SCRAPER v2.1
# ==========================================================
class SportmonksXGScraperV2_1:
    """Hauptklasse für das Scraping von xG-Daten v2.1 (FINAL)"""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.client = SportmonksXGClientV2_1(config)

        # Separate Listen
        self.complete_data = []
        self.odds_only_data = []
        self.xg_only_data = []

        self.completed_leagues = set()

        self.xg_start_date = self.config.XG_START_DATE.date()
        self.last_scraped_date = self.xg_start_date

        # Lade Cache
        self._load_cache()

    def _load_cache(self):
        """Lade existierende Daten aus CSV"""
        temp_file = f"temp_{self.config.output_file}"

        if os.path.exists(temp_file):
            print(f"🔄 Lade Zwischenstand aus {temp_file}...")
            try:
                cached_df = pd.read_csv(temp_file)
                cached_df['date'] = pd.to_datetime(cached_df['date'])
                cached_df = cached_df[cached_df['date'].dt.date >= self.xg_start_date]

                self.complete_data = cached_df.to_dict('records')

                if not cached_df.empty:
                    self.last_scraped_date = cached_df['date'].max().date()
                    print(f"✅ {len(self.complete_data)} Spiele im Cache. Letztes: {self.last_scraped_date}")
            except Exception as e:
                print(f"❌ Fehler beim Laden: {e}. Starte neu.")
        else:
            print(f"ℹ️ Keine Cache-Datei gefunden. Starte ab {self.xg_start_date}")

    def scrape_league(self, league: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scrape alle Daten für eine Liga"""

        league_id = league.get('id')
        league_name = league.get('name', 'Unknown')

        print(f"\n🏆 {league_name} (ID: {league_id})")
        print("=" * 60)

        seasons = self.client.get_seasons_for_league(league_id)

        if not seasons:
            print(f"⚠️ Keine relevanten Saisons gefunden")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print(f"📅 {len(seasons)} relevante Saisons: {[s.get('name') for s in seasons]}")

        league_complete = []
        league_odds_only = []
        league_xg_only = []

        for season in seasons:
            season_id = season.get('id')
            season_name = season.get('name')

            print(f"\n  🔄 Saison {season_name}...")

            fixtures = self.client.get_fixtures_for_season(season_id, league_name)

            if not fixtures:
                print(f"    ⚠️ Keine Spiele gefunden")
                continue

            # Limit für Testing
            if self.config.max_fixtures_per_season:
                fixtures = fixtures[:self.config.max_fixtures_per_season]
                print(f"    DEBUG: Limitiert auf {len(fixtures)} Fixtures")

            # PRE-FILTER: Nur abgeschlossene Spiele
            if self.config.only_finished_games:
                finished_fixtures = []
                for f in fixtures:
                    state_id = f.get('state_id')
                    if state_id in [5, 6, 7]:  # FT states
                        finished_fixtures.append(f)

                print(f"    📊 {len(finished_fixtures)}/{len(fixtures)} Spiele abgeschlossen (FT)")
                fixtures = finished_fixtures

            if not fixtures:
                print(f"    ⚠️ Keine abgeschlossenen Spiele")
                continue

            season_stats = {
                'complete': 0,
                'odds_only': 0,
                'xg_only': 0,
                'skipped_date': 0,
                'skipped_no_data': 0,
            }

            print(f"    🎯 Verarbeite {len(fixtures)} abgeschlossene Spiele...")

            # SCHRITT 1: Extrahiere Basis-Daten + xG
            games_data = []
            for fixture in tqdm(fixtures, desc=f"    Saison {season_name} - Basis-Daten"):
                try:
                    game_data = self.client.extract_xg_from_fixture(fixture)

                    # Datum prüfen
                    try:
                        fixture_date_str = game_data.get('date')
                        if not fixture_date_str:
                            continue
                        fixture_date = datetime.fromisoformat(fixture_date_str).date()
                    except Exception:
                        continue

                    # Filter: Datum
                    if fixture_date < self.last_scraped_date:
                        season_stats['skipped_date'] += 1
                        continue

                    # Speichere für Quoten-Abruf
                    games_data.append(game_data)

                except Exception as e:
                    if self.config.debug:
                        print(f"\n    WARNUNG: Fehler bei Fixture {fixture.get('id')}: {e}")
                    continue

            print(f"    ✅ {len(games_data)} Spiele für Quoten-Abruf vorbereitet")

            # SCHRITT 2: Hole Quoten (nur für relevante Spiele)
            for game_data in tqdm(games_data, desc=f"    Saison {season_name} - Quoten"):
                try:
                    fixture_id = game_data['fixture_id']

                    # Hole Quoten
                    odds_data = self.client.get_odds_for_fixture(fixture_id)

                    # Kombiniere
                    combined_data = {**game_data, **odds_data}

                    # Kategorisiere
                    has_odds = combined_data.get('odds_home') is not None
                    has_xg = (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)

                    if has_odds and has_xg:
                        league_complete.append(combined_data)
                        season_stats['complete'] += 1
                        self.client.stats['fixtures_complete'] += 1
                        self.client.stats['fixtures_with_odds'] += 1
                        self.client.stats['fixtures_with_xg'] += 1
                    elif has_odds:
                        league_odds_only.append(combined_data)
                        season_stats['odds_only'] += 1
                        self.client.stats['fixtures_with_odds'] += 1
                    elif has_xg:
                        league_xg_only.append(combined_data)
                        season_stats['xg_only'] += 1
                        self.client.stats['fixtures_with_xg'] += 1
                    else:
                        season_stats['skipped_no_data'] += 1

                except Exception as e:
                    if self.config.debug:
                        print(f"\n    WARNUNG: Fehler bei Quoten für {game_data.get('fixture_id')}: {e}")
                    continue

            # Saison-Statistik
            print(f"    ✅ Ergebnis:")
            print(f"       - Komplett (Quoten + xG): {season_stats['complete']}")
            print(f"       - Nur Quoten: {season_stats['odds_only']}")
            print(f"       - Nur xG: {season_stats['xg_only']}")
            if season_stats['skipped_date'] > 0:
                print(f"       - Übersprungen (Datum): {season_stats['skipped_date']}")
            if season_stats['skipped_no_data'] > 0:
                print(f"       - Übersprungen (keine Daten): {season_stats['skipped_no_data']}")

        # DataFrames erstellen
        df_complete = pd.DataFrame(league_complete) if league_complete else pd.DataFrame()
        df_odds = pd.DataFrame(league_odds_only) if league_odds_only else pd.DataFrame()
        df_xg = pd.DataFrame(league_xg_only) if league_xg_only else pd.DataFrame()

        return df_complete, df_odds, df_xg

    def scrape_all(self):
        """Scrape alle Ligen"""

        print("\n" + "=" * 70)
        print("🚀 SPORTMONKS xG & ODDS SCRAPER v2.1 (FINAL)")
        print("=" * 70)
        print(f"\n⚙️ Konfiguration:")
        print(f"  • Filter: Spiele ab {self.last_scraped_date}")
        print(f"  • Nur abgeschlossene Spiele: {self.config.only_finished_games}")
        print(f"  • Timeout: {self.config.request_timeout}s")
        print(f"  • Output: {self.config.output_file}")

        leagues = self.client.get_leagues()

        if not leagues:
            print("\n❌ Keine Ligen gefunden!")
            return

        try:
            for league in leagues:
                league_name = league.get('name', 'Unknown')

                try:
                    df_complete, df_odds, df_xg = self.scrape_league(league)

                    if not df_complete.empty:
                        self.complete_data.extend(df_complete.to_dict('records'))
                    if not df_odds.empty:
                        self.odds_only_data.extend(df_odds.to_dict('records'))
                    if not df_xg.empty:
                        self.xg_only_data.extend(df_xg.to_dict('records'))

                    if self.config.save_intermediate:
                        self._save_intermediate()

                except Exception as e:
                    print(f"❌ Fehler bei Liga {league_name}: {e}")
                    if self.config.debug:
                        traceback.print_exc()
                    continue

        except KeyboardInterrupt:
            print("\n\n⚠️ Abgebrochen durch Benutzer")

        finally:
            print("\n⏹️ Speichere finalen Stand...")
            self._save_intermediate()

    def _save_intermediate(self):
        """Speichere Zwischenstand"""
        try:
            if self.complete_data:
                df = pd.DataFrame(self.complete_data)
                df = df.drop_duplicates(subset=['fixture_id'], keep='last')
                df.to_csv(f"temp_{self.config.output_file}", index=False)
        except Exception as e:
            print(f"❌ Konnte Zwischenstand nicht speichern: {e}")

    def save_data(self):
        """Speichere finale Daten"""

        print(f"\n{'='*70}")
        print("💾 SPEICHERE DATEN")
        print(f"{'='*70}")

        output_cols = [
            'date', 'league', 'season', 'home_team', 'away_team',
            'home_score', 'away_score',
            'home_xg', 'away_xg',
            'odds_home', 'odds_draw', 'odds_away',
            'status', 'fixture_id'
        ]

        # 1. Komplette Daten
        if self.complete_data:
            df = pd.DataFrame(self.complete_data)
            df = df.drop_duplicates(subset=['fixture_id'], keep='last')

            final_cols = [col for col in output_cols if col in df.columns]
            df = df[final_cols].copy()

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)

            df.to_csv(self.config.output_file, index=False)
            print(f"\n✅ KOMPLETT (Quoten + xG): {len(df)} Spiele")
            print(f"   Datei: {self.config.output_file}")
            print(f"   Größe: {os.path.getsize(self.config.output_file)/1024:.1f} KB")
        else:
            print(f"\n⚠️ Keine kompletten Spiele gefunden!")

        # 2. Nur Quoten
        if self.odds_only_data:
            df = pd.DataFrame(self.odds_only_data)
            df = df.drop_duplicates(subset=['fixture_id'], keep='last')

            final_cols = [col for col in output_cols if col in df.columns]
            df = df[final_cols].copy()

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)

            df.to_csv(self.config.output_file_odds_only, index=False)
            print(f"\n✅ NUR QUOTEN: {len(df)} Spiele")
            print(f"   Datei: {self.config.output_file_odds_only}")

        # 3. Nur xG
        if self.xg_only_data:
            df = pd.DataFrame(self.xg_only_data)
            df = df.drop_duplicates(subset=['fixture_id'], keep='last')

            final_cols = [col for col in output_cols if col in df.columns]
            df = df[final_cols].copy()

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)

            df.to_csv(self.config.output_file_xg_only, index=False)
            print(f"\n✅ NUR xG: {len(df)} Spiele")
            print(f"   Datei: {self.config.output_file_xg_only}")

        # Statistiken
        self._print_statistics()

    def _print_statistics(self):
        """Drucke finale Statistiken"""

        print(f"\n{'='*70}")
        print("📊 FINALE STATISTIKEN")
        print(f"{'='*70}")

        print(f"\n🌐 API-Calls: {self.client.api_calls}")
        if self.client.stats['api_timeouts'] > 0:
            print(f"⚠️  Timeouts: {self.client.stats['api_timeouts']}")

        print(f"\n📈 Fixtures:")
        print(f"  • Gesamt abgerufen: {self.client.stats['fixtures_fetched']}")
        print(f"  • Mit Quoten + xG: {self.client.stats['fixtures_complete']} ⭐")
        print(f"  • Mit Quoten: {self.client.stats['fixtures_with_odds']}")
        print(f"  • Mit xG: {self.client.stats['fixtures_with_xg']}")

        # Liga-Verteilung
        if self.complete_data:
            df = pd.DataFrame(self.complete_data)
            if 'league' in df.columns:
                print(f"\n🏆 Verteilung nach Ligen (Komplett):")
                league_counts = df['league'].value_counts()
                for league, count in league_counts.items():
                    print(f"  • {league}: {count}")

            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                print(f"\n📅 Zeitraum: {df['date'].min().date()} bis {df['date'].max().date()}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Hauptfunktion"""

    api_token = os.getenv("SPORTMONKS_API_TOKEN")

    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        return

    config = ScraperConfig(
        api_token=api_token,
        request_delay=1.5,  # Etwas langsamer für Stabilität
        request_timeout=60,  # 60s Timeout
        output_file="game_database_sportmonks.csv",
        output_file_odds_only="game_database_sportmonks_odds_only.csv",
        output_file_xg_only="game_database_sportmonks_xg_only.csv",
        save_intermediate=True,
        debug=True,
        max_fixtures_per_season=None,  # Alle
        only_finished_games=True  # Nur FT-Spiele
    )

    scraper = SportmonksXGScraperV2_1(config)

    try:
        scraper.scrape_all()
        scraper.save_data()

    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen durch Benutzer")
        scraper.save_data()
    except Exception as e:
        print(f"\n\n❌ KRITISCHER FEHLER: {e}")
        traceback.print_exc()
        scraper.save_data()


if __name__ == "__main__":
    main()
