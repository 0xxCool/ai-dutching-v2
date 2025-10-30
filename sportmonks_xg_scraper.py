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
    """Konfiguration für den Sportmonks Scraper"""
    
    # API Settings
    api_token: str = ""
    base_url: str = "https://api.sportmonks.com/v3/football"
    max_workers: int = 5  # (Wird aktuell nicht genutzt, da sequenziell stabiler)
    request_delay: float = 0.3  # Standard, wird in main() überschrieben
    
    # Daten-Einstellungen
    seasons_to_scrape: List[int] = None  # Welche Saisons (z.B. [2023, 2024, 2025])
    include_current_season: bool = True
    include_previous_seasons: int = 2  # Wie viele vergangene Saisons
    
    # xG Settings (erfordert xG Add-on bei Sportmonks!)
    include_xg: bool = True  # Wenn False, verwende alternative Metriken
    
    # Output
    output_file: str = "game_database_sportmonks.csv"
    save_intermediate: bool = True  # Speichere nach jeder Liga
    
    # Performance
    batch_size: int = 100  # Fixtures pro Batch-Request
    use_cache: bool = True  # Cache bereits abgerufene Daten
    cache_file: str = "sportmonks_cache.json"
    
    def __post_init__(self):
        if self.seasons_to_scrape is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # Bestimme aktuelle Saison
            if current_month >= 8:  # August-Dezember
                current_season = current_year
            else:  # Januar-Juli
                current_season = current_year - 1
            
            # Erstelle Liste der zu scrapenden Saisons
            self.seasons_to_scrape = [
                current_season - i 
                for i in range(self.include_previous_seasons + 1)
            ]

# ==========================================================
# SPORTMONKS API CLIENT (Erweitert für xG)
# ==========================================================
class SportmonksXGClient:
    """Hochperformanter Client für Sportmonks API mit xG-Daten"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.api_calls = 0
        self.session = requests.Session()  # Wiederverwendbare Session für Performance
        
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
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                # WICHTIG: Delay NACH dem Request, um Server nicht zu überlasten
                time.sleep(self.config.request_delay)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ Fehler nach {max_retries} Versuchen: {e}")
                    return None
                time.sleep(1)
        
        return None
    
    def get_leagues(self) -> List[Dict]:
            """Hole alle verfügbaren Top-Ligen (OHNE PL/Championship wegen API-Bug im Trial)"""
            print("📋 Lade verfügbare Ligen...")
            
            # Top-Ligen (Sportmonks IDs)
            # IDs 8 (Premier League) und 2 (Championship) entfernt!
            top_league_ids = [
                # 8,      # Premier League - TEMPORÄR ENTFERNT
                82,     # Bundesliga
                564,    # La Liga
                384,    # Serie A
                301,    # Ligue 1
                72,     # Eredivisie
                271,    # Primeira Liga (Superliga)
                # 2,      # Championship - TEMPORÄR ENTFERNT
                390,    # Belgian First Division A
                501,    # Scottish Premiership
                1489,   # Brasileirão Série A
            ]
            
            leagues = []
            # Füge einen kleinen Delay hinzu, um Rate Limits sicher zu vermeiden
            initial_delay = 1.3 # Entspricht unserem Haupt-Request-Delay
            
            print(f"Lade Daten für {len(top_league_ids)} Ligen...")
            for league_id in tqdm(top_league_ids, desc="Lade Ligainformationen"):
                # Warte nur vor der ersten Anfrage länger
                if leagues:
                    time.sleep(0.2) # Kurzer Delay zwischen Liga-Abfragen
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
            """Hole Saisons für eine Liga (KORRIGIERT)"""
            
            # KORREKTE V3-SYNTAX:
            endpoint = 'seasons'
            params = {'filters': f'leagues:{league_id}'}
            data = self._make_request(endpoint, params)
            
            if not data or 'data' not in data:
                return []
            
            print(f"DEBUG: API returned {len(data['data'])} seasons for league {league_id}")

            # KORREKTER FILTER:
            # Wir nehmen ALLE Saisons, die der API-Plan liefert
            seasons = []
            for season in data['data']:
                if season.get('id') and season.get('name'):
                    seasons.append(season)
            
            return seasons
    
    def get_fixtures_for_season(self, season_id: int, league_name: str) -> List[Dict]:
            """
            Hole alle Fixtures für eine Saison (NEUER ANSATZ via /seasons/{id} include)
            Dieser Ansatz wurde vom Sportmonks-Support (Fred) vorgeschlagen,
            um den Paginierungs-Bug im /fixtures-Endpunkt zu umgehen.
            """
            
            print(f" 	 (DEBUG: Versuche neuen Ansatz: /seasons/{season_id}?include=fixtures...)")

            # KORREKTER ENDPUNKT (von Fred vorgeschlagen)
            # Wir rufen die Saison selbst auf, nicht den Fixtures-Endpunkt
            endpoint = f"seasons/{season_id}"
            
            # KORREKTE INCLUDES (verschachtelt, wie von Fred vorgeschlagen)
            # Wir holen die Fixtures UND deren Teilnehmer, Scores & Statistiken
            params = {
                'include': 'fixtures.participants;fixtures.scores;fixtures.statistics'
            }

            # Mache EINEN EINZIGEN Request für die Saison + alle ihre Fixtures
            # Der eingebaute request_delay=1.3s gilt hier
            data = self._make_request(endpoint, params)

            # Fehlerbehandlung
            if not data or 'data' not in data:
                print(f" 	 ⚠️ (DEBUG: Neuer Ansatz - Kein 'data'-Objekt für Saison {season_id} gefunden.)")
                return []

            # 'data' ist jetzt das Saison-Objekt
            season_data = data['data']

            # Die Fixtures sind jetzt ein verschachteltes Objekt IN der Saison-Antwort
            fixtures = season_data.get('fixtures')

            # Prüfen, ob Fixtures vorhanden und eine Liste sind
            if not fixtures or not isinstance(fixtures, list):
                print(f" 	 ⚠️ (DEBUG: Neuer Ansatz - Saison {season_id} hat keine 'fixtures' geladen.)")
                # Dies ist das erwartete Verhalten, wenn der Plan keine Spieldaten für diese Saison enthält
                return []
            
            # (Härtung) Filtere Strings oder kaputte Daten heraus
            clean_fixtures = [f for f in fixtures if isinstance(f, dict)]
            
            if len(clean_fixtures) < len(fixtures):
                print(f" 	 (DEBUG: {len(fixtures) - len(clean_fixtures)} ungültige Einträge in fixtures-Liste gefiltert.)")

            # Da wir eine einzige Antwort erhalten, ersetzen wir die tqdm-Schleife durch eine Log-Meldung
            print(f" 	 (DEBUG: Neuer Ansatz - {len(clean_fixtures)} Fixtures in einer Antwort erhalten.)")
            
            # Wir brauchen hier keine tqdm-Schleife mehr, da alle Daten auf einmal kommen.
            # Die Verarbeitung (Filterung) findet jetzt in `scrape_league` statt.
            return clean_fixtures
    
    # === DIESE FUNKTION MUSS INNERHALB DER "SportmonksXGClient"-KLASSE SEIN ===
    def extract_xg_from_fixture(self, fixture: Dict) -> Dict:
        """Extrahiere xG-Daten aus einem Fixture (GEHÄRTET, KORRIGIERT FÜR state_id)"""
        
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
            
            league_info = fixture.get('league', {})
            result['league'] = league_info.get('name', 'Unknown') if isinstance(league_info, dict) else 'InvalidLeagueData'
            
            season_info = fixture.get('season', {})
            result['season'] = season_info.get('name', 'Unknown') if isinstance(season_info, dict) else 'InvalidSeasonData'
            
            # === KORREKTUR FÜR STATUS (state_id) ===
            state_id = fixture.get('state_id')
            
            # 5 = Finished
            # 6 = Finished (After Extra Time)
            # 7 = Finished (After Penalties)
            
            if state_id in [5, 6, 7]:
                result['status'] = 'FT' 
            elif state_id == 1:
                result['status'] = 'NS'
            elif state_id == 2:
                result['status'] = 'LIVE'
            else:
                state_info = fixture.get('state', {})
                status_str = state_info.get('state', 'Unknown') if isinstance(state_info, dict) else 'InvalidStateData'
                if 'FT' in status_str or 'AET' in status_str or 'PEN' in status_str:
                     result['status'] = 'FT'
                else:
                     result['status'] = status_str

            # Extrahiere Team-Namen
            participants = fixture.get('participants', [])
            if isinstance(participants, list):
                for p in participants:
                     if isinstance(p, dict):
                         meta = p.get('meta', {})
                         if isinstance(meta, dict):
                             location = meta.get('location')
                             if location == 'home':
                                 result['home_team'] = p.get('name')
                             elif location == 'away':
                                 result['away_team'] = p.get('name')

                if not result['home_team'] and len(participants) >= 1 and isinstance(participants[0], dict):
                    result['home_team'] = participants[0].get('name')
                if not result['away_team'] and len(participants) >= 2 and isinstance(participants[1], dict):
                    result['away_team'] = participants[1].get('name')
            else:
                 if result['fixture_id']:
                    print(f"(DEBUG: fixture {result['fixture_id']} hat ungültige participants-Daten)")


            # Extrahiere Scores
            scores = fixture.get('scores', [])
            if isinstance(scores, list):
                for score in scores:
                     if isinstance(score, dict):
                         desc = score.get('description', '').lower()
                         if 'full' in desc or 'current' in desc or 'final' in desc:
                             participant_id = score.get('participant_id')
                             score_data = score.get('score', {})
                             goals = score_data.get('goals') if isinstance(score_data, dict) else None
                             
                             if goals is not None and isinstance(participants, list):
                                 for p in participants:
                                     if isinstance(p, dict) and p.get('id') == participant_id:
                                         meta = p.get('meta', {})
                                         if isinstance(meta, dict):
                                             location = meta.get('location')
                                             if location == 'home':
                                                 result['home_score'] = int(goals)
                                             elif location == 'away':
                                                 result['away_score'] = int(goals)
            else:
                if result['fixture_id']:
                    print(f"(DEBUG: fixture {result['fixture_id']} hat ungültige scores-Daten)")


            # Extrahiere xG aus Statistics
            statistics = fixture.get('statistics', [])
            if isinstance(statistics, list):
                for stat in statistics:
                     if isinstance(stat, dict):
                         stat_data = stat.get('data', [])
                         if isinstance(stat_data, list):
                             for data_point in stat_data:
                                 if isinstance(data_point, dict):
                                     type_id = data_point.get('type_id')
                                     
                                     if type_id == 52:  # xG
                                         participant_id = data_point.get('participant_id')
                                         xg_value = data_point.get('value')
                                         
                                         if xg_value is not None and isinstance(participants, list):
                                             for p in participants:
                                                 if isinstance(p, dict) and p.get('id') == participant_id:
                                                     meta = p.get('meta', {})
                                                     if isinstance(meta, dict):
                                                         location = meta.get('location')
                                                         try:
                                                             xg_float = float(xg_value)
                                                             if location == 'home':
                                                                 result['home_xg'] = xg_float
                                                             elif location == 'away':
                                                                 result['away_xg'] = xg_float
                                                         except (ValueError, TypeError):
                                                              if result['fixture_id']:
                                                                  print(f"(DEBUG: fixture {result['fixture_id']} hat ungültigen xG-Wert: {xg_value})")
            else:
                if result['fixture_id']:
                    print(f"(DEBUG: fixture {result['fixture_id']} hat ungültige statistics-Daten)")
        
        except Exception as e:
            print(f"\n(WARNUNG: Unerwarteter Fehler in extract_xg_from_fixture für Spiel {result.get('fixture_id', 'UNKNOWN')}: {e})")
            return result
            
        return result
    
# ==========================================================
# HAUPT-SCRAPER (ANGEPASST FÜR RESUME & STABILITÄT)
# ==========================================================
class SportmonksXGScraper:
    """Hauptklasse für das Scraping von xG-Daten"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.client = SportmonksXGClient(config)
        self.all_data = [] # Hält die Daten im Speicher
        self.completed_leagues = set()
        self.temp_file_path = f"temp_{self.config.output_file}"
        
        # === NEUE LOGIK: LADE CACHE ===
        if os.path.exists(self.temp_file_path):
            print(f"🔄 Lade Zwischenstand aus {self.temp_file_path}...")
            try:
                cached_df = pd.read_csv(self.temp_file_path)
                self.all_data = cached_df.to_dict('records')
                self.completed_leagues = set(cached_df['league'].unique())
                print(f"✅ {len(self.completed_leagues)} Ligen bereits geladen: {self.completed_leagues}")
            except pd.errors.EmptyDataError:
                print(f"⚠️ Temporäre Datei {self.temp_file_path} ist leer.")
            except Exception as e:
                print(f"❌ Fehler beim Laden der Cache-Datei: {e}")

    def scrape_league(self, league: Dict) -> pd.DataFrame:
            """Scrape alle Daten für eine Liga (GEHÄRTET: Fängt Fehler pro Spiel ab, xG-Filter ENTFERNT)"""
            league_id = league.get('id')
            league_name = league.get('name', 'Unknown')
            
            print(f"\n🏆 {league_name} (ID: {league_id})")
            print("=" * 60)
            
            seasons = self.client.get_seasons_for_league(league_id)
            
            if not seasons:
                print(f"⚠️  Keine Saisons gefunden")
                return pd.DataFrame()
            
            print(f"📅 {len(seasons)} Saisons gefunden: {[s.get('name') for s in seasons]}")
            
            league_data = []
            
            # Scrape jede Saison
            for season in seasons:
                season_id = season.get('id')
                season_name = season.get('name')
                
                print(f"\n 	🔄 Saison {season_name}...")
                
                # 1. Hole die (hoffentlich saubere) Liste der Spiele
                fixtures = self.client.get_fixtures_for_season(season_id, league_name)
                
                if not fixtures:
                    print(f" 	 	⚠️ Keine Spiele gefunden")
                    continue
                
                season_added_games_count = 0 # Umbenannt für Klarheit
                
                # 2. Verarbeite die Spiele (mit try...except)
                for fixture in fixtures:
                    try:
                        # === HIER IST DIE KORREKTUR: "self.client." wurde hinzugefügt ===
                        game_data = self.client.extract_xg_from_fixture(fixture)
                        
                        # === ANGEPASSTE BEDINGUNG (OHNE xG > 0) ===
                        # Speichere alle abgeschlossenen Spiele mit Teams
                        if (game_data['status'] in ['FT', 'AET', 'FT_PEN'] and 
                            game_data['home_team'] and game_data['away_team']):
                            league_data.append(game_data)
                            season_added_games_count += 1
                    
                    except Exception as e:
                        # Fange Fehler für EIN Spiel ab, ohne die Liga abzubrechen
                        fixture_id = 'UNKNOWN_ID'
                        if isinstance(fixture, dict):
                            fixture_id = fixture.get('id', 'ID_NOT_FOUND')
                        print(f"\n 	 	 (WARNUNG: Überspringe Spiel {fixture_id} wegen Fehler: {e})")
                        continue # Mache mit dem nächsten Spiel weiter
                
                # Passe die Log-Nachricht an
                print(f" 	 	✅ {season_added_games_count} abgeschlossene Spiele hinzugefügt (xG ignoriert)")
            
            if league_data:
                df = pd.DataFrame(league_data)
                print(f"\n 	📊 Gesamt für {league_name}: {len(df)} Spiele")
                return df
            
            return pd.DataFrame()
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape alle Ligen (mit Resume-Logik)"""
        print("\n" + "=" * 70)
        print("🚀 SPORTMONKS xG DATA SCRAPER")
        print("=" * 70)
        print(f"\n⚙️  Konfiguration:")
        print(f"  • Saisons: {self.config.seasons_to_scrape} (Ignoriert, nimmt alle vom API-Plan)")
        print(f"  • xG-Modus: {'Aktiviert' if self.config.include_xg else 'Proxy (Schüsse)'}")
        print(f"  • Parallele Workers: {self.config.max_workers} (Deaktiviert)")
        print(f"  • Output: {self.config.output_file}")
        
        # Hole Ligen
        leagues = self.client.get_leagues()
        
        if not leagues:
            print("\n❌ Keine Ligen gefunden!")
            return pd.DataFrame(self.all_data)
        
        # Temporäres DataFrame aus bereits geladenen Daten erstellen
        all_dataframes = []
        if self.all_data:
            all_dataframes.append(pd.DataFrame(self.all_data))
        
        try:
            # Scrape jede Liga
            for league in leagues:
                league_name = league.get('name', 'Unknown')
                
                # === NEUE LOGIK: ÜBERSPRINGE FERTIGE LIGEN ===
                if league_name in self.completed_leagues:
                    print(f"\n⏭️  Überspringe {league_name} (bereits im Cache)")
                    continue
                
                try:
                    df = self.scrape_league(league)
                    
                    if not df.empty:
                        all_dataframes.append(df)
                        self.all_data.extend(df.to_dict('records')) # Für den Fall eines Absturzes
                        
                        # Speichere Zwischenstand
                        if self.config.save_intermediate:
                            print(f" 💾 Speichere Zwischenstand ({len(self.all_data)} Spiele)...")
                            # Sichern durch Concat (stellt sicher, dass alle Spalten übereinstimmen)
                            temp_df = pd.concat(all_dataframes, ignore_index=True)
                            temp_df.to_csv(self.temp_file_path, index=False)
                            self.completed_leagues.add(league_name) # Markiere als fertig
                    
                except Exception as e:
                    print(f"❌ Schwerer Fehler bei Liga {league.get('name')}: {e}")
                    traceback.print_exc()
                    print("...fahre mit nächster Liga fort.")
                    continue
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Abgebrochen durch Benutzer")
            # Springt direkt zum 'finally'-Block

        finally:
            # === NEUE LOGIK: SPEICHERE IMMER BEI ABBRUCH ===
            print("\n⏹️  Speichere finalen Zwischenstand...")
            if self.all_data:
                try:
                    final_df = pd.DataFrame(self.all_data)
                    # Entferne Duplikate, falls vorhanden
                    final_df = final_df.drop_duplicates(subset=['fixture_id'], keep='last')
                    final_df.to_csv(self.temp_file_path, index=False)
                    print(f"✅ Zwischenstand in {self.temp_file_path} gespeichert.")
                except Exception as e:
                    print(f"❌ Konnte Zwischenstand nicht speichern: {e}")
            else:
                print("Keine Daten zum Speichern.")

        # Kombiniere alle Daten
        if self.all_data:
            final_df = pd.DataFrame(self.all_data)
            
            # Sortiere nach Datum
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values('date').reset_index(drop=True)
            
            # Entferne Duplikate
            final_df = final_df.drop_duplicates(subset=['fixture_id'], keep='first')
            
            return final_df
        
        return pd.DataFrame()

    def save_data(self, df: pd.DataFrame):
        """Speichere Daten als CSV"""
        if df.empty:
            print("\n❌ Keine Daten zum Speichern!")
            return
        
        # Wähle nur relevante Spalten für Output
        output_cols = [
            'date', 'home_team', 'away_team', 
            'home_xg', 'away_xg', 'league',
            'home_score', 'away_score', 'status', 'fixture_id'
        ]
        # Stelle sicher, dass alle Spalten existieren
        final_cols = [col for col in output_cols if col in df.columns]
        output_df = df[final_cols].copy()
        
        output_df.to_csv(self.config.output_file, index=False)
        
        print(f"\n{'='*70}")
        print("✅ SCRAPING ABGESCHLOSSEN")
        print(f"{'='*70}")
        print(f"\n📊 STATISTIKEN:")
        print(f"  • Gespeicherte Spiele: {len(output_df)}")
        print(f"  • API-Calls: {self.client.api_calls}")
        print(f"  • Datei: {self.config.output_file}")
        
        if 'league' in output_df.columns:
            print(f"\n📈 Verteilung nach Ligen:")
            print(output_df['league'].value_counts().to_string())
        
        if 'date' in output_df.columns:
            print(f"\n📅 Zeitraum: {output_df['date'].min()} bis {output_df['date'].max()}")
        
        if os.path.exists(self.config.output_file):
             print(f"\n💾 Dateigröße: {os.path.getsize(self.config.output_file) / 1024:.2f} KB")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    """Hauptfunktion"""
    
    # Lade API Token
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    
    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        print("\nBitte erstellen Sie eine .env Datei mit:")
        print("SPORTMONKS_API_TOKEN=your_token_here")
        return
    
    # Konfiguration
    config = ScraperConfig(
        api_token=api_token,
        include_previous_seasons=2,
        max_workers=3, 	 	 
        
        # === KORREKTUR RATE-LIMIT HIER ===
        # 3000 req/Stunde = 50 req/Minute = 1 req / 1.2s
        # Wir nehmen 1.3s für einen Sicherheitspuffer.
        request_delay=1.3,
        
        output_file="game_database_sportmonks.csv",
        save_intermediate=True,
        include_xg=True, 	 	 
    )
    
    # Starte Scraper
    scraper = SportmonksXGScraper(config)
    
    try:
        df = scraper.scrape_all()
        scraper.save_data(df)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen durch Benutzer")
        # Speichern wird jetzt im 'finally'-Block von scrape_all() gehandhabt
        
    except Exception as e:
        print(f"\n\n❌ KRITISCHER FEHLER: {e}")
        traceback.print_exc()
        # Auch hier speichern wir, was wir haben
        print("Versuche, Zwischenstand zu speichern...")
        scraper.save_data(pd.DataFrame(scraper.all_data))

if __name__ == "__main__":
    main()