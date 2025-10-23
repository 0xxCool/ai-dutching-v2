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
    max_workers: int = 5  # Parallele Requests (nicht zu hoch wegen Rate Limit)
    request_delay: float = 0.3  # Sekunden zwischen Requests
    
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
        
        # Top-Ligen (Sportmonks IDs)
        top_league_ids = [
            8,      # Premier League
            82,     # Bundesliga
            564,    # La Liga
            384,    # Serie A
            301,    # Ligue 1
            72,     # Eredivisie
            271,    # Primeira Liga
            2,      # Championship
            390,    # Belgian First Division A
            501,    # Scottish Premiership
            1489,   # Brasileirão Série A
        ]
        
        leagues = []
        for league_id in top_league_ids:
            data = self._make_request(f"leagues/{league_id}")
            if data and 'data' in data:
                leagues.append(data['data'])
        
        print(f"✅ {len(leagues)} Ligen geladen")
        return leagues
    
    def get_seasons_for_league(self, league_id: int) -> List[Dict]:
        """Hole Saisons für eine Liga"""
        params = {
            'filters': f'leagueId:{league_id}',
        }
        
        data = self._make_request('seasons', params)
        
        if not data or 'data' not in data:
            return []
        
        # Filtere nur gewünschte Saisons
        seasons = []
        for season in data['data']:
            # Extrahiere Jahr aus season name (z.B. "2024/2025" -> 2024)
            try:
                year = int(season.get('name', '').split('/')[0])
                if year in self.config.seasons_to_scrape:
                    seasons.append(season)
            except (ValueError, IndexError):
                continue
        
        return seasons
    
    def get_fixtures_for_season(self, season_id: int, league_name: str) -> List[Dict]:
        """Hole alle Fixtures für eine Saison mit xG-Daten"""
        
        # WICHTIG: xG-Daten sind in 'statistics' enthalten
        # Wenn xG Add-on nicht vorhanden, verwende Schüsse als Proxy
        includes = [
            'participants',  # Team-Namen
            'scores',        # Ergebnisse
            'statistics',    # Hier sind xG-Daten!
        ]
        
        params = {
            'filters': f'seasonId:{season_id}',
            'include': ';'.join(includes),
            'per_page': 100,  # Maximum pro Request
        }
        
        all_fixtures = []
        page = 1
        
        with tqdm(desc=f"  {league_name}", unit=" fixtures") as pbar:
            while True:
                params['page'] = page
                data = self._make_request('fixtures', params)
                
                if not data or 'data' not in data:
                    break
                
                fixtures = data['data']
                if not fixtures:
                    break
                
                all_fixtures.extend(fixtures)
                pbar.update(len(fixtures))
                
                # Prüfe ob es mehr Seiten gibt
                pagination = data.get('pagination', {})
                if not pagination.get('has_more', False):
                    break
                
                page += 1
        
        return all_fixtures
    
    def extract_xg_from_fixture(self, fixture: Dict) -> Dict:
        """Extrahiere xG-Daten aus einem Fixture"""
        
        # Basis-Informationen
        result = {
            'fixture_id': fixture.get('id'),
            'date': fixture.get('starting_at'),
            'home_team': None,
            'away_team': None,
            'home_xg': 0.0,
            'away_xg': 0.0,
            'home_score': None,
            'away_score': None,
            'league': fixture.get('league', {}).get('name', 'Unknown'),
            'season': fixture.get('season', {}).get('name', 'Unknown'),
            'status': fixture.get('state', {}).get('state', 'Unknown')
        }
        
        # Extrahiere Team-Namen
        participants = fixture.get('participants', [])
        if len(participants) >= 2:
            # participants[0] ist normalerweise Home, participants[1] ist Away
            # Aber prüfe sicherheitshalber 'meta.location'
            for p in participants:
                if p.get('meta', {}).get('location') == 'home':
                    result['home_team'] = p.get('name')
                elif p.get('meta', {}).get('location') == 'away':
                    result['away_team'] = p.get('name')
        
        # Fallback wenn meta nicht vorhanden
        if not result['home_team'] and len(participants) >= 1:
            result['home_team'] = participants[0].get('name')
        if not result['away_team'] and len(participants) >= 2:
            result['away_team'] = participants[1].get('name')
        
        # Extrahiere Scores
        scores = fixture.get('scores', [])
        for score in scores:
            desc = score.get('description', '').lower()
            if 'current' in desc or 'final' in desc:
                participant_id = score.get('participant_id')
                goals = score.get('score', {}).get('goals', 0)
                
                # Finde ob Home oder Away
                for p in participants:
                    if p.get('id') == participant_id:
                        if p.get('meta', {}).get('location') == 'home':
                            result['home_score'] = goals
                        else:
                            result['away_score'] = goals
        
        # Extrahiere xG aus Statistics
        statistics = fixture.get('statistics', [])
        
        for stat in statistics:
            # xG kann verschiedene type_ids haben, typischerweise:
            # type_id 52 = Expected Goals (xG)
            stat_data = stat.get('data', [])
            
            for data_point in stat_data:
                type_id = data_point.get('type_id')
                
                # Expected Goals
                if type_id == 52:  # xG
                    participant_id = data_point.get('participant_id')
                    xg_value = data_point.get('value', 0)
                    
                    # Bestimme ob Home oder Away
                    for p in participants:
                        if p.get('id') == participant_id:
                            if p.get('meta', {}).get('location') == 'home':
                                result['home_xg'] = float(xg_value) if xg_value else 0.0
                            else:
                                result['away_xg'] = float(xg_value) if xg_value else 0.0
        
        # Wenn keine xG-Daten, versuche Schüsse als Proxy (falls verfügbar)
        if result['home_xg'] == 0 and result['away_xg'] == 0:
            for stat in statistics:
                stat_data = stat.get('data', [])
                
                for data_point in stat_data:
                    type_id = data_point.get('type_id')
                    
                    # Total Shots = type_id 42
                    # Shots on Target = type_id 43
                    if type_id in [42, 43]:  # Verwende Schüsse als rough proxy
                        participant_id = data_point.get('participant_id')
                        shots = data_point.get('value', 0)
                        
                        # Sehr grobe xG-Schätzung: shots * 0.1 (10% Konversionsrate)
                        xg_estimate = float(shots) * 0.1 if shots else 0
                        
                        for p in participants:
                            if p.get('id') == participant_id:
                                if p.get('meta', {}).get('location') == 'home':
                                    if result['home_xg'] == 0:  # Nur wenn noch kein xG
                                        result['home_xg'] = xg_estimate
                                else:
                                    if result['away_xg'] == 0:
                                        result['away_xg'] = xg_estimate
        
        return result

# ==========================================================
# HAUPT-SCRAPER
# ==========================================================
class SportmonksXGScraper:
    """Hauptklasse für das Scraping von xG-Daten"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.client = SportmonksXGClient(config)
        self.all_data = []
    
    def scrape_league(self, league: Dict) -> pd.DataFrame:
        """Scrape alle Daten für eine Liga"""
        league_id = league.get('id')
        league_name = league.get('name', 'Unknown')
        
        print(f"\n🏆 {league_name} (ID: {league_id})")
        print("=" * 60)
        
        # Hole Saisons
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
            
            print(f"\n  🔄 Saison {season_name}...")
            
            fixtures = self.client.get_fixtures_for_season(season_id, league_name)
            
            if not fixtures:
                print(f"    ⚠️ Keine Spiele gefunden")
                continue
            
            # Extrahiere xG-Daten
            for fixture in fixtures:
                xg_data = self.client.extract_xg_from_fixture(fixture)
                
                # Nur abgeschlossene Spiele mit xG-Daten
                if (xg_data['status'] in ['FT', 'AET', 'FT_PEN'] and 
                    xg_data['home_team'] and xg_data['away_team'] and
                    (xg_data['home_xg'] > 0 or xg_data['away_xg'] > 0)):
                    league_data.append(xg_data)
            
            print(f"    ✅ {len([f for f in fixtures if self.client.extract_xg_from_fixture(f)['home_xg'] > 0])} Spiele mit xG-Daten")
        
        if league_data:
            df = pd.DataFrame(league_data)
            print(f"\n  📊 Gesamt für {league_name}: {len(df)} Spiele")
            return df
        
        return pd.DataFrame()
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape alle Ligen"""
        print("\n" + "=" * 70)
        print("🚀 SPORTMONKS xG DATA SCRAPER")
        print("=" * 70)
        print(f"\n⚙️  Konfiguration:")
        print(f"  • Saisons: {self.config.seasons_to_scrape}")
        print(f"  • xG-Modus: {'Aktiviert' if self.config.include_xg else 'Proxy (Schüsse)'}")
        print(f"  • Parallele Workers: {self.config.max_workers}")
        print(f"  • Output: {self.config.output_file}")
        
        # Hole Ligen
        leagues = self.client.get_leagues()
        
        if not leagues:
            print("\n❌ Keine Ligen gefunden!")
            return pd.DataFrame()
        
        all_dataframes = []
        
        # Scrape jede Liga (sequenziell für bessere Ausgabe)
        for league in leagues:
            try:
                df = self.scrape_league(league)
                
                if not df.empty:
                    all_dataframes.append(df)
                    
                    # Speichere Zwischenstand
                    if self.config.save_intermediate:
                        temp_df = pd.concat(all_dataframes, ignore_index=True)
                        temp_df.to_csv(f"temp_{self.config.output_file}", index=False)
                
            except Exception as e:
                print(f"❌ Fehler bei {league.get('name')}: {e}")
                continue
        
        # Kombiniere alle Daten
        if all_dataframes:
            final_df = pd.concat(all_dataframes, ignore_index=True)
            
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
        output_df = df[[
            'date', 'home_team', 'away_team', 
            'home_xg', 'away_xg', 'league'
        ]].copy()
        
        output_df.to_csv(self.config.output_file, index=False)
        
        print(f"\n{'='*70}")
        print("✅ SCRAPING ABGESCHLOSSEN")
        print(f"{'='*70}")
        print(f"\n📊 STATISTIKEN:")
        print(f"  • Gespeicherte Spiele: {len(output_df)}")
        print(f"  • API-Calls: {self.client.api_calls}")
        print(f"  • Datei: {self.config.output_file}")
        print(f"\n📈 Verteilung nach Ligen:")
        print(output_df['league'].value_counts().to_string())
        print(f"\n📅 Zeitraum: {output_df['date'].min()} bis {output_df['date'].max()}")
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
        include_previous_seasons=2,  # Letzte 3 Saisons (aktuelle + 2 vergangene)
        max_workers=3,               # Konservativ für Rate Limits
        request_delay=0.3,           # 300ms zwischen Requests
        output_file="game_database_sportmonks.csv",
        save_intermediate=True,
        include_xg=True,             # xG Add-on erforderlich!
    )
    
    # Starte Scraper
    scraper = SportmonksXGScraper(config)
    
    try:
        df = scraper.scrape_all()
        scraper.save_data(df)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen durch Benutzer")
        print(f"Zwischenstand in temp_{config.output_file} gespeichert")
        
    except Exception as e:
        print(f"\n\n❌ KRITISCHER FEHLER: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()