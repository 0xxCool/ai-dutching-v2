import json
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
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
    """Konfiguration für den Sportmonks Scraper (Final)"""
    
    # API Settings
    api_token: str = ""
    base_url: str = "https://api.sportmonks.com/v3/football"
    request_delay: float = 1.3  # Standard (Wird in main() gesetzt)
    
    # Daten-Einstellungen
    seasons_to_scrape: List[int] = None # Wird ignoriert, da wir nach Datum filtern
    
    # xG Settings
    include_xg: bool = True
    
    # Output
    output_file: str = "game_database_sportmonks.csv"
    save_intermediate: bool = True
    
    # Startdatum für xG-Daten (von Kim bestätigt)
    XG_START_DATE: datetime = datetime(2024, 3, 1)
    

# ==========================================================
# SPORTMONKS API CLIENT (FINAL)
# ==========================================================
class SportmonksXGClient:
    """Client für Sportmonks API (xG + Standard Odds)"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.api_calls = 0
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Zentrale Request-Funktion mit Retry-Logik"""
        if params is None:
            params = {}
        
        params['api_token'] = self.config.api_token
        url = f"{self.config.base_url}/{endpoint}"
        
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
                time.sleep(self.config.request_delay)
                
                return response.json()
                
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
                8, 82, 564, 384, 301, 72, 271, 2, 390, 501, 1489
            ]
            
            leagues = []
            initial_delay = 1.3 
            
            print(f"Lade Daten für {len(top_league_ids)} Ligen...")
            for league_id in tqdm(top_league_ids, desc="Lade Ligainformationen"):
                if leagues:
                    time.sleep(0.2)
                else:
                    time.sleep(initial_delay)

                data = self._make_request(f"leagues/{league_id}")
                if data and 'data' in data:
                    leagues.append(data['data'])
                else:
                    print(f"(WARNUNG: Konnte Liga-Info für ID {league_id} nicht laden)")
            
            print(f"✅ {len(leagues)} Ligainformationen erfolgreich geladen.")
            return leagues
    
    def get_seasons_for_league(self, league_id: int) -> List[Dict]:
            """
            Hole Saisons (NEUER, KORREKTER ANSATZ von Kim: /leagues/{id}?include=seasons)
            """
            
            # Kims neuer, korrekter Endpunkt
            endpoint = f"leagues/{league_id}"
            params = {'include': 'seasons'}
            
            print(f"DEBUG: Versuche Kims neuen Ansatz: /leagues/{league_id}?include=seasons")
            
            data = self._make_request(endpoint, params)

            if not data or 'data' not in data:
                print(f" 	 ⚠️ (DEBUG: Kein 'data'-Objekt für Liga {league_id} gefunden.)")
                return []

            # 'data' ist das Liga-Objekt, 'seasons' ist darin verschachtelt
            league_data = data['data']
            seasons_list = league_data.get('seasons') # Hole die "included" Saisons

            if not seasons_list or not isinstance(seasons_list, list):
                print(f" 	 ⚠️ (DEBUG: Liga {league_id} hat keinen 'seasons'-Include zurückgegeben.)")
                return []
            
            # Ab hier ist die Logik gleich: Filtere die Saisons nach Datum
            relevant_seasons = []
            xg_start_date = self.config.XG_START_DATE.date()

            for season in seasons_list:
                # Wir brauchen Saisons, die 2024 oder später enden
                if season.get('id') and season.get('name') and season.get('ending_at'):
                    try:
                        ending_date = datetime.fromisoformat(season['ending_at']).date()
                        if ending_date >= xg_start_date:
                            relevant_seasons.append(season)
                    except (ValueError, TypeError):
                        continue # Ignoriere Saisons mit ungültigem Datum
            
            print(f"DEBUG: API returned {len(seasons_list)} seasons, {len(relevant_seasons)} sind relevant (post-März-2024).")
            return relevant_seasons
    
    def get_fixtures_for_season(self, season_id: int, league_name: str) -> List[Dict]:
            """Hole alle Fixtures für eine Saison (via /seasons/{id} include)
               (FINAL: JETZT MIT KORREKTEM xGFixture INCLUDE)"""
            
            print(f" 	 (DEBUG: Versuche Ansatz: /seasons/{season_id}?include=fixtures...)")
            endpoint = f"seasons/{season_id}"
            
            # === FINALE KORREKTUR HIER: 'fixtures.xGFixture' hinzugefügt ===
            params = {
                'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;league;fixtures.xGFixture'
            }

            data = self._make_request(endpoint, params)

            if not data or 'data' not in data:
                print(f" 	 ⚠️ (DEBUG: Kein 'data'-Objekt für Saison {season_id} gefunden.)")
                return []

            season_data = data['data']
            fixtures = season_data.get('fixtures')
            league_info = season_data.get('league', {})
            
            if not fixtures or not isinstance(fixtures, list):
                print(f" 	 ⚠️ (DEBUG: Saison {season_id} hat keine 'fixtures' geladen.)")
                return []
            
            clean_fixtures = []
            for f in fixtures:
                if isinstance(f, dict):
                    f['league_info_from_season'] = league_info
                    clean_fixtures.append(f)
                
            if len(clean_fixtures) < len(fixtures):
                print(f" 	 (DEBUG: {len(fixtures) - len(clean_fixtures)} ungültige Einträge in fixtures-Liste gefiltert.)")

            print(f" 	 (DEBUG: {len(clean_fixtures)} Fixtures in einer Antwort erhalten.)")
            return clean_fixtures
    
    # === HIER IST DIE KORREKTE PLATZIERUNG ===
    
    def get_odds_for_fixture(self, fixture_id: int) -> Dict:
            """Hole Quoten für ein spezifisches historisches Spiel (FINAL: PRE-MATCH FEED)"""
            
            # === KORREKTER ENDPUNKT (PRE-MATCH, wie von Kim ZULETZT bestätigt) ===
            endpoint = f'odds/pre-match/fixtures/{fixture_id}'
            
            params = {
                'include': 'market;bookmaker' # Wir brauchen den Markt und den Bookmaker
            }
            
            data = self._make_request(endpoint, params)
            
            if not data or 'data' not in data:
                return {} # Leeres Dict zurückgeben, wenn keine Quoten
            
            return self._parse_sportmonks_odds(data['data'])
        
    def _parse_sportmonks_odds(self, odds_data: List[Dict]) -> Dict:
            """Parse die PRE-MATCH Odds-Daten und suche nach 3Way Result"""
            
            odds_dict = {
                'odds_home': None,
                'odds_draw': None,
                'odds_away': None
            }
            
            # Finde den "3Way Result" Markt
            for odds_item in odds_data:
                market = odds_item.get('market')
                if not market or market.get('name') != '3Way Result':
                    continue
                
                bookmaker_odds_list = odds_item.get('bookmaker', {}).get('odds', [])
                if not bookmaker_odds_list:
                    continue
                
                odds_values = bookmaker_odds_list[0].get('odds', [])
                if not odds_values:
                    continue

                home_odd = next((o['value'] for o in odds_values if o['label'] == 'Home'), 0)
                draw_odd = next((o['value'] for o in odds_values if o['label'] == 'Draw'), 0)
                away_odd = next((o['value'] for o in odds_values if o['label'] == 'Away'), 0)
                
                if all([home_odd, draw_odd, away_odd]):
                    odds_dict['odds_home'] = float(home_odd)
                    odds_dict['odds_draw'] = float(draw_odd)
                    odds_dict['odds_away'] = float(away_odd)
                    
                    return odds_dict 
            
            return odds_dict

    def extract_xg_from_fixture(self, fixture: Dict) -> Dict:
            """Extrahiere xG-Daten aus einem Fixture (FINAL: Liest 'xgfixture'-Liste korrekt)"""
            
            result = {
                'fixture_id': None, 'date': None, 'home_team': None, 'away_team': None,
                'home_xg': 0.0, 'away_xg': 0.0, 'home_score': None, 'away_score': None,
                'league': 'Unknown', 'season': 'Unknown', 'status': 'Unknown'
            }

            if not isinstance(fixture, dict):
                print("(DEBUG: extract_xg_from_fixture erhielt keinen dict)")
                return result

            try:
                result['fixture_id'] = fixture.get('id')
                result['date'] = fixture.get('starting_at')
                
                league_info = fixture.get('league_info_from_season', {})
                result['league'] = league_info.get('name', 'Unknown') if isinstance(league_info, dict) else 'InvalidLeagueData'
                
                season_info = fixture.get('season', {})
                result['season'] = season_info.get('name', 'Unknown') if isinstance(season_info, dict) else 'InvalidSeasonData'
                
                # Status-Logik (ist korrekt)
                state_id = fixture.get('state_id')
                if state_id in [5, 6, 7]:
                    result['status'] = 'FT' 
                else:
                    state_info = fixture.get('state', {})
                    status_str = state_info.get('state', 'Unknown') if isinstance(state_info, dict) else 'InvalidStateData'
                    if 'FT' in status_str or 'AET' in status_str or 'PEN' in status_str:
                        result['status'] = 'FT'
                    else:
                        result['status'] = status_str

                # Teilnehmer-Logik (ist korrekt)
                participants = fixture.get('participants', [])
                if isinstance(participants, list):
                    for p in participants:
                        if isinstance(p, dict) and p.get('meta', {}).get('location') == 'home':
                            result['home_team'] = p.get('name')
                        elif isinstance(p, dict) and p.get('meta', {}).get('location') == 'away':
                            result['away_team'] = p.get('name')

                # Scores-Logik (ist korrekt)
                scores = fixture.get('scores', [])
                if isinstance(scores, list):
                    for score in scores:
                        if isinstance(score, dict):
                            desc = score.get('description', '').lower()
                            # Historische Daten verwenden oft 'CURRENT' statt 'FULLTIME'
                            if 'current' in desc or 'full' in desc or 'final' in desc:
                                participant_id = score.get('participant_id')
                                score_data = score.get('score', {})
                                goals = score_data.get('goals') if isinstance(score_data, dict) else None
                                
                                if goals is not None:
                                    # Finde heraus, ob das 'home' oder 'away' ist
                                    if isinstance(participants, list) and len(participants) > 0:
                                        if participants[0].get('id') == participant_id: # Annahme: participants[0] ist home
                                            result['home_score'] = int(goals)
                                        elif len(participants) > 1 and participants[1].get('id') == participant_id: # Annahme: participants[1] ist away
                                            result['away_score'] = int(goals)

                # === KORRIGIERTE xG-LOGIK (BASIEREND AUF DEINEM JSON) ===
                xg_data_list = fixture.get('xgfixture') # Es ist eine LISTE
                
                if isinstance(xg_data_list, list):
                    for xg_item in xg_data_list:
                        if isinstance(xg_item, dict):
                            
                            # type_id 5304 scheint das Haupt-xG zu sein
                            if xg_item.get('type_id') == 5304: 
                                location = xg_item.get('location')
                                value = xg_item.get('data', {}).get('value')
                                
                                if value is not None:
                                    try:
                                        if location == 'home':
                                            result['home_xg'] = float(value)
                                        elif location == 'away':
                                            result['away_xg'] = float(value)
                                    except (ValueError, TypeError):
                                        pass # Behalte 0.0, wenn Wert ungültig ist
                # === ENDE DER KORREKTUR ===
            
            except Exception as e:
                print(f"\n(WARNUNG: Unerwarteter Fehler in extract_xg_from_fixture für Spiel {result.get('fixture_id', 'UNKNOWN')}: {e})")
                return result
                
            return result

# ==========================================================
# HAUPT-SCRAPER (FINAL)
# ==========================================================
class SportmonksXGScraper:
    """Hauptklasse für das Scraping von xG-Daten"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.client = SportmonksXGClient(config)
        self.all_data = [] 
        self.completed_leagues = set()
        self.temp_file_path = f"temp_{self.config.output_file}"
        
        # Definiere den Startpunkt
        self.xg_start_date = self.config.XG_START_DATE.date()
        self.last_scraped_date = self.xg_start_date

        if os.path.exists(self.temp_file_path):
            print(f"🔄 Lade Zwischenstand aus {self.temp_file_path}...")
            try:
                cached_df = pd.read_csv(self.temp_file_path)
                cached_df['date'] = pd.to_datetime(cached_df['date'])
                
                # Filtere alte Daten (pre-xG) aus dem Cache, falls vorhanden
                cached_df = cached_df[cached_df['date'].dt.date >= self.xg_start_date]
                
                self.all_data = cached_df.to_dict('records')
                
                if not cached_df.empty:
                    self.last_scraped_date = cached_df['date'].max().date()
                    print(f"✅ Letztes Spiel im Cache vom {self.last_scraped_date}. Scrape ab da weiter.")
                else:
                    print(f"ℹ️ Cache-Datei enthielt nur veraltete Daten (vor {self.xg_start_date}). Starte neu.")

            except Exception as e:
                print(f"❌ Fehler beim Laden der Cache-Datei: {e}. Starte neu ab {self.xg_start_date}.")
                self.last_scraped_date = self.xg_start_date
        else:
            print(f"ℹ️ Keine Cache-Datei gefunden. Starte Scrape ab {self.xg_start_date}.")


    def scrape_league(self, league: Dict) -> pd.DataFrame:
                """Scrape alle Daten für eine Liga (INKLUSIVE HISTORISCHER QUOTEN & DATUMS/XG-FILTER)"""
                league_id = league.get('id')
                league_name = league.get('name', 'Unknown')
                
                print(f"\n🏆 {league_name} (ID: {league_id})")
                print("=" * 60)
                
                seasons = self.client.get_seasons_for_league(league_id)
                
                if not seasons:
                    print(f"⚠️  Keine relevanten Saisons (post-März-2024) gefunden")
                    return pd.DataFrame()
                
                print(f"📅 {len(seasons)} relevante Saisons gefunden: {[s.get('name') for s in seasons]}")
                
                league_data = []
                
                for season in seasons:
                    season_id = season.get('id')
                    season_name = season.get('name')
                    
                    print(f"\n 	🔄 Saison {season_name}...")
                    
                    fixtures = self.client.get_fixtures_for_season(season_id, league_name)
                    
                    if not fixtures:
                        print(f" 	 	⚠️ Keine Spiele gefunden")
                        continue
                    
                    season_added_games_count = 0
                    
                    print(f" 	 	Filtere Spiele ab {self.last_scraped_date} und hole Quoten für {len(fixtures)} Spiele...")
                    
                    for fixture in tqdm(fixtures, desc=f" 	 Saison {season_name} Quoten"):
                        try:
                            game_data = self.client.extract_xg_from_fixture(fixture)
                            
                            try:
                                fixture_date_str = game_data.get('date')
                                if not fixture_date_str:
                                    continue
                                fixture_date = datetime.fromisoformat(fixture_date_str).date()
                            except Exception as e:
                                print(f"(DEBUG: Ungültiges Datumsformat für Spiel {game_data.get('fixture_id')}: {e})")
                                continue
                            
                            # === FILTER 1: DATUM (MUSS NACH MÄRZ 2024 & NACH LETZTEM SCRAPE SEIN) ===
                            if fixture_date < self.last_scraped_date:
                                continue
                                
                            # === FILTER 2: STATUS (MUSS ABGESCHLOSSEN SEIN) ===
                            if not (game_data['status'] in ['FT', 'AET', 'FT_PEN'] and 
                                    game_data['home_team'] and game_data['away_team']):
                                continue
                                
                            # === SCHRITT 3: HOLE QUOTEN (NUR FÜR RELEVANTE SPIELE) ===
                            odds_data = self.client.get_odds_for_fixture(fixture['id'])
                            
                            combined_data = {**game_data, **odds_data}
                            
                            # === FILTER 3: MUSS QUOTEN UND XG HABEN ===
                            if (combined_data.get('odds_home') and 
                                (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)):
                                
                                league_data.append(combined_data)
                                season_added_games_count += 1
                        
                        except Exception as e:
                            fixture_id = 'UNKNOWN_ID'
                            if isinstance(fixture, dict):
                                fixture_id = fixture.get('id', 'ID_NOT_FOUND')
                            print(f"\n 	 	 (WARNUNG: Überspringe Spiel {fixture_id} wegen Fehler: {e})")
                            continue
                    
                    print(f" 	 	✅ {season_added_games_count} abgeschlossene Spiele mit xG UND Quoten hinzugefügt")
                
                if league_data:
                    df = pd.DataFrame(league_data)
                    print(f"\n 	📊 Gesamt für {league_name}: {len(df)} Spiele")
                    return df
                
                return pd.DataFrame()
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape alle Ligen (mit Resume-Logik)"""
        print("\n" + "=" * 70)
        print("🚀 SPORTMONKS xG DATA SCRAPER (TARGETED: 2024-Heute mit Quoten)")
        print("=" * 70)
        print(f"\n⚙️  Konfiguration:")
        print(f"  • Filter: Nur Spiele ab {self.last_scraped_date} mit xG UND Quoten")
        print(f"  • Output: {self.config.output_file}")
        
        leagues = self.client.get_leagues()
        
        if not leagues:
            print("\n❌ Keine Ligen gefunden!")
            return pd.DataFrame(self.all_data)
        
        all_dataframes = []
        if self.all_data:
            all_dataframes.append(pd.DataFrame(self.all_data))
        
        try:
            for league in leagues:
                league_name = league.get('name', 'Unknown')
                
                # Die Resume-Logik (ganze Ligen überspringen) ist jetzt weniger wichtig,
                # da der Datumsfilter (self.last_scraped_date) das meiste überspringt.
                # Wir lassen sie aber zur Sicherheit drin.
                # if league_name in self.completed_leagues:
                #     print(f"\n⏭️  Überspringe {league_name} (bereits im Cache)")
                #     continue
                
                try:
                    df = self.scrape_league(league)
                    
                    if not df.empty:
                        all_dataframes.append(df)
                        # self.all_data wird jetzt nur mit NEUEN Daten gefüllt
                        self.all_data.extend(df.to_dict('records')) 
                        
                        if self.config.save_intermediate:
                            print(f" 💾 Speichere Zwischenstand...")
                            # Kombiniere ALTE (aus Cache) + NEUE (von diesem Lauf)
                            final_df_for_save = pd.DataFrame(self.all_data)
                            final_df_for_save = final_df_for_save.drop_duplicates(subset=['fixture_id'], keep='last')
                            
                            final_df_for_save.to_csv(self.temp_file_path, index=False)
                            self.completed_leagues.add(league_name)
                    
                except Exception as e:
                    print(f"❌ Schwerer Fehler bei Liga {league.get('name')}: {e}")
                    traceback.print_exc()
                    print("...fahre mit nächster Liga fort.")
                    continue
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Abgebrochen durch Benutzer")
        finally:
            print("\n⏹️  Speichere finalen Zwischenstand...")
            if self.all_data:
                try:
                    final_df = pd.DataFrame(self.all_data)
                    final_df = final_df.drop_duplicates(subset=['fixture_id'], keep='last')
                    final_df.to_csv(self.temp_file_path, index=False)
                    print(f"✅ Zwischenstand in {self.temp_file_path} gespeichert.")
                except Exception as e:
                    print(f"❌ Konnte Zwischenstand nicht speichern: {e}")
            else:
                print("Keine Daten zum Speichern.")

        if self.all_data:
            final_df = pd.DataFrame(self.all_data)
            final_df = final_df.drop_duplicates(subset=['fixture_id'], keep='last')
            if 'date' in final_df.columns:
                final_df['date'] = pd.to_datetime(final_df['date'])
                final_df = final_df.sort_values('date').reset_index(drop=True)
            return final_df
        
        return pd.DataFrame()

    def save_data(self, df: pd.DataFrame):
        """Speichimere Daten als CSV (MIT QUOTEN)"""
        if df.empty:
            print("\n❌ Keine neuen Daten zum Speichern gefunden!")
            if os.path.exists(self.config.output_file):
                 print(f"Behalte existierende Datei: {self.config.output_file}")
            return
        
        output_cols = [
            'date', 'league', 'season', 'home_team', 'away_team', 
            'home_score', 'away_score', 
            'home_xg', 'away_xg', 
            'odds_home', 'odds_draw', 'odds_away',
            'status', 'fixture_id'
        ]
        
        final_cols = [col for col in output_cols if col in df.columns]
        output_df = df[final_cols].copy()
        
        # === FINALE DATENBANK SPEICHERN ===
        # Wir überschreiben die alte Datei mit den neuen, relevanten Daten
        output_df.to_csv(self.config.output_file, index=False)
        
        print(f"\n{'='*70}")
        print("✅ SCRAPING ABGESCHLOSSEN")
        print(f"{'='*70}")
        print(f"\n📊 STATISTIKEN:")
        print(f"  • Gespeicherte Spiele (Gesamt): {len(output_df)}")
        print(f"  • API-Calls in diesem Lauf: {self.client.api_calls}")
        print(f"  • Datei: {self.config.output_file}")
        
        if 'league' in output_df.columns:
            print(f"\n📈 Verteilung nach Ligen:")
            print(output_df['league'].value_counts().to_string())
        
        if 'date' in output_df.columns and not output_df.empty:
            print(f"\n📅 Zeitraum: {output_df['date'].min()} bis {output_df['date'].max()}")
        
        if os.path.exists(self.config.output_file):
             print(f"\n💾 Dateigröße: {os.path.getsize(self.config.output_file) / 1024:.2f} KB")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    """Hauptfunktion"""
    
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    
    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        return
    
    config = ScraperConfig(
        api_token=api_token,
        request_delay=1.3,
        output_file="game_database_sportmonks.csv",
        save_intermediate=True,
        include_xg=True, 	 	 
    )
    
    scraper = SportmonksXGScraper(config)
    
    try:
        df = scraper.scrape_all()
        scraper.save_data(df)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen durch Benutzer")
    except Exception as e:
        print(f"\n\n❌ KRITISCHER FEHLER: {e}")
        traceback.print_exc()
        print("Versuche, Zwischenstand zu speichern...")
        scraper.save_data(pd.DataFrame(scraper.all_data))

if __name__ == "__main__":
    main()
