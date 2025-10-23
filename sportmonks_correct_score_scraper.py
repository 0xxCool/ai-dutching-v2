import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from dataclasses import dataclass

load_dotenv()

# ==========================================================
# KONFIGURATION
# ==========================================================
@dataclass
class CorrectScoreScraperConfig:
    """Konfiguration für Correct Score Scraper"""
    
    api_token: str = ""
    base_url: str = "https://api.sportmonks.com/v3/football"
    request_delay: float = 0.3
    
    # Daten-Einstellungen
    seasons_to_scrape: List[int] = None
    include_previous_seasons: int = 2  # 3 Saisons total
    
    # Output
    output_file: str = "correct_score_database.csv"
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.seasons_to_scrape is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if current_month >= 8:
                current_season = current_year
            else:
                current_season = current_year - 1
            
            self.seasons_to_scrape = [
                current_season - i 
                for i in range(self.include_previous_seasons + 1)
            ]

# ==========================================================
# CLIENT
# ==========================================================
class SportmonksCorrectScoreClient:
    """Client für Correct Score Daten"""
    
    def __init__(self, config: CorrectScoreScraperConfig):
        self.config = config
        self.api_calls = 0
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Request mit Retry-Logik"""
        if params is None:
            params = {}
        
        params['api_token'] = self.config.api_token
        url = f"{self.config.base_url}/{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=20)
                self.api_calls += 1
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Rate Limit - warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                time.sleep(self.config.request_delay)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"❌ Fehler: {e}")
                    return None
                time.sleep(1)
        
        return None
    
    def get_leagues(self) -> List[Dict]:
        """Hole Top-Ligen"""
        print("📋 Lade Ligen...")
        
        top_league_ids = [
            8, 82, 564, 384, 301, 72, 271, 2, 390, 501, 1489
        ]
        
        leagues = []
        for league_id in top_league_ids:
            data = self._make_request(f"leagues/{league_id}")
            if data and 'data' in data:
                leagues.append(data['data'])
        
        print(f"✅ {len(leagues)} Ligen geladen")
        return leagues
    
    def get_seasons_for_league(self, league_id: int) -> List[Dict]:
        """Hole Saisons"""
        params = {'filters': f'leagueId:{league_id}'}
        data = self._make_request('seasons', params)
        
        if not data or 'data' not in data:
            return []
        
        seasons = []
        for season in data['data']:
            try:
                year = int(season.get('name', '').split('/')[0])
                if year in self.config.seasons_to_scrape:
                    seasons.append(season)
            except (ValueError, IndexError):
                continue
        
        return seasons
    
    def get_fixtures_for_season(self, season_id: int, league_name: str) -> List[Dict]:
        """Hole Fixtures mit Scores und xG"""
        
        includes = ['participants', 'scores', 'statistics']
        
        params = {
            'filters': f'seasonId:{season_id}',
            'include': ';'.join(includes),
            'per_page': 100,
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
                
                pagination = data.get('pagination', {})
                if not pagination.get('has_more', False):
                    break
                
                page += 1
        
        return all_fixtures
    
    def extract_correct_score_data(self, fixture: Dict) -> Optional[Dict]:
        """Extrahiere Correct Score + xG Daten"""
        
        result = {
            'fixture_id': fixture.get('id'),
            'date': fixture.get('starting_at'),
            'home_team': None,
            'away_team': None,
            'home_score': None,
            'away_score': None,
            'correct_score': None,  # z.B. "2-1"
            'home_xg': 0.0,
            'away_xg': 0.0,
            'league': fixture.get('league', {}).get('name', 'Unknown'),
            'season': fixture.get('season', {}).get('name', 'Unknown'),
            'status': fixture.get('state', {}).get('state', 'Unknown')
        }
        
        # Team-Namen
        participants = fixture.get('participants', [])
        for p in participants:
            if p.get('meta', {}).get('location') == 'home':
                result['home_team'] = p.get('name')
            elif p.get('meta', {}).get('location') == 'away':
                result['away_team'] = p.get('name')
        
        if not result['home_team'] and len(participants) >= 1:
            result['home_team'] = participants[0].get('name')
        if not result['away_team'] and len(participants) >= 2:
            result['away_team'] = participants[1].get('name')
        
        # Scores (WICHTIG für Correct Score!)
        scores = fixture.get('scores', [])
        for score in scores:
            desc = score.get('description', '').lower()
            if 'current' in desc or 'final' in desc or 'fulltime' in desc:
                participant_id = score.get('participant_id')
                goals = score.get('score', {}).get('goals')
                
                if goals is not None:
                    for p in participants:
                        if p.get('id') == participant_id:
                            if p.get('meta', {}).get('location') == 'home':
                                result['home_score'] = int(goals)
                            else:
                                result['away_score'] = int(goals)
        
        # Erstelle Correct Score String
        if result['home_score'] is not None and result['away_score'] is not None:
            result['correct_score'] = f"{result['home_score']}-{result['away_score']}"
        
        # xG-Daten
        statistics = fixture.get('statistics', [])
        for stat in statistics:
            stat_data = stat.get('data', [])
            
            for data_point in stat_data:
                type_id = data_point.get('type_id')
                
                if type_id == 52:  # Expected Goals
                    participant_id = data_point.get('participant_id')
                    xg_value = data_point.get('value', 0)
                    
                    for p in participants:
                        if p.get('id') == participant_id:
                            if p.get('meta', {}).get('location') == 'home':
                                result['home_xg'] = float(xg_value) if xg_value else 0.0
                            else:
                                result['away_xg'] = float(xg_value) if xg_value else 0.0
        
        # Fallback: Schüsse als xG-Proxy
        if result['home_xg'] == 0 and result['away_xg'] == 0:
            for stat in statistics:
                stat_data = stat.get('data', [])
                
                for data_point in stat_data:
                    type_id = data_point.get('type_id')
                    
                    if type_id in [42, 43]:  # Shots
                        participant_id = data_point.get('participant_id')
                        shots = data_point.get('value', 0)
                        xg_estimate = float(shots) * 0.1 if shots else 0
                        
                        for p in participants:
                            if p.get('id') == participant_id:
                                if p.get('meta', {}).get('location') == 'home':
                                    if result['home_xg'] == 0:
                                        result['home_xg'] = xg_estimate
                                else:
                                    if result['away_xg'] == 0:
                                        result['away_xg'] = xg_estimate
        
        # Nur abgeschlossene Spiele mit gültigem Score zurückgeben
        if (result['status'] in ['FT', 'AET', 'FT_PEN'] and 
            result['correct_score'] and 
            result['home_team'] and 
            result['away_team']):
            return result
        
        return None

# ==========================================================
# HAUPT-SCRAPER
# ==========================================================
class CorrectScoreScraper:
    """Scraper für Correct Score Daten"""
    
    def __init__(self, config: CorrectScoreScraperConfig):
        self.config = config
        self.client = SportmonksCorrectScoreClient(config)
    
    def scrape_league(self, league: Dict) -> pd.DataFrame:
        """Scrape Liga"""
        league_id = league.get('id')
        league_name = league.get('name', 'Unknown')
        
        print(f"\n🏆 {league_name} (ID: {league_id})")
        print("=" * 60)
        
        seasons = self.client.get_seasons_for_league(league_id)
        
        if not seasons:
            print(f"⚠️  Keine Saisons gefunden")
            return pd.DataFrame()
        
        print(f"📅 {len(seasons)} Saisons: {[s.get('name') for s in seasons]}")
        
        league_data = []
        
        for season in seasons:
            season_id = season.get('id')
            season_name = season.get('name')
            
            print(f"\n  🔄 Saison {season_name}...")
            
            fixtures = self.client.get_fixtures_for_season(season_id, league_name)
            
            if not fixtures:
                print(f"    ⚠️ Keine Spiele")
                continue
            
            valid_count = 0
            for fixture in fixtures:
                data = self.client.extract_correct_score_data(fixture)
                if data:
                    league_data.append(data)
                    valid_count += 1
            
            print(f"    ✅ {valid_count} gültige Spiele")
        
        if league_data:
            df = pd.DataFrame(league_data)
            print(f"\n  📊 Gesamt: {len(df)} Spiele")
            return df
        
        return pd.DataFrame()
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape alle Ligen"""
        print("\n" + "=" * 70)
        print("⚽ CORRECT SCORE DATA SCRAPER")
        print("=" * 70)
        print(f"\n⚙️  Konfiguration:")
        print(f"  • Saisons: {self.config.seasons_to_scrape}")
        print(f"  • Output: {self.config.output_file}")
        
        leagues = self.client.get_leagues()
        
        if not leagues:
            print("\n❌ Keine Ligen!")
            return pd.DataFrame()
        
        all_dataframes = []
        
        for league in leagues:
            try:
                df = self.scrape_league(league)
                
                if not df.empty:
                    all_dataframes.append(df)
                    
                    if self.config.save_intermediate:
                        temp_df = pd.concat(all_dataframes, ignore_index=True)
                        temp_df.to_csv(f"temp_{self.config.output_file}", index=False)
                
            except Exception as e:
                print(f"❌ Fehler: {e}")
                continue
        
        if all_dataframes:
            final_df = pd.concat(all_dataframes, ignore_index=True)
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values('date').reset_index(drop=True)
            final_df = final_df.drop_duplicates(subset=['fixture_id'], keep='first')
            
            return final_df
        
        return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame):
        """Speichere Daten"""
        if df.empty:
            print("\n❌ Keine Daten!")
            return
        
        df.to_csv(self.config.output_file, index=False)
        
        print(f"\n{'='*70}")
        print("✅ SCRAPING ABGESCHLOSSEN")
        print(f"{'='*70}")
        print(f"\n📊 STATISTIKEN:")
        print(f"  • Spiele: {len(df)}")
        print(f"  • API-Calls: {self.client.api_calls}")
        print(f"  • Datei: {self.config.output_file}")
        
        print(f"\n📈 Verteilung nach Ligen:")
        print(df['league'].value_counts().to_string())
        
        print(f"\n⚽ Top Correct Scores:")
        print(df['correct_score'].value_counts().head(10).to_string())
        
        print(f"\n📅 Zeitraum: {df['date'].min()} bis {df['date'].max()}")
        print(f"\n💾 Größe: {os.path.getsize(self.config.output_file) / 1024:.2f} KB")

# ==========================================================
# MAIN
# ==========================================================
def main():
    """Hauptfunktion"""
    
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    
    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN fehlt in .env!")
        return
    
    config = CorrectScoreScraperConfig(
        api_token=api_token,
        include_previous_seasons=2,  # 3 Saisons
        output_file="correct_score_database.csv",
        save_intermediate=True,
    )
    
    scraper = CorrectScoreScraper(config)
    
    try:
        df = scraper.scrape_all()
        scraper.save_data(df)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Abgebrochen")
        
    except Exception as e:
        print(f"\n\n❌ FEHLER: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()