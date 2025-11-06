"""
📥 SPORTMONKS HYBRID SCRAPER V4
================================
Verbesserte Version mit robuster Fehlerbehandlung

Features:
- Lädt historische Match-Daten
- Erstellt Training-CSV
- Progress Tracking
- Automatic Retry
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    SPORTMONKS_API_KEY = os.getenv('SPORTMONKS_API_TOKEN')
    SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"
    
    # Leagues (Top 5)
    LEAGUE_IDS = [
        271,  # Premier League
        8,    # Bundesliga
        564,  # La Liga
        384,  # Serie A
        301   # Ligue 1
    ]
    
    # Time Range
    START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Output
    OUTPUT_FILE = "game_database_complete.csv"
    
    # API Settings
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds


# =============================================================================
# SCRAPER
# =============================================================================

class SportmonksScraper:
    """Scraped historische Daten von Sportmonks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.headers = {'Accept': 'application/json'}
        self.session = requests.Session()
    
    def fetch_matches(self, league_id: int, start_date: str, end_date: str) -> list:
        """Holt Matches für eine Liga"""
        
        url = f"{self.config.SPORTMONKS_BASE_URL}/fixtures/between/{start_date}/{end_date}/{league_id}"
        
        params = {
            'api_token': self.config.SPORTMONKS_API_KEY,
            'include': 'participants;scores;league'
        }
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                return data.get('data', [])
            
            except Exception as e:
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    print(f"❌ Fehler nach {self.config.MAX_RETRIES} Versuchen: {e}")
                    return []
        
        return []
    
    def parse_match(self, match: dict) -> dict:
        """Parsed ein Match zu DataFrame-Format"""
        
        try:
            participants = match.get('participants', [])
            if len(participants) < 2:
                return None
            
            home_team = participants[0]['name']
            away_team = participants[1]['name']
            
            scores = match.get('scores', [])
            home_score = 0
            away_score = 0
            
            # Find full-time score
            for score in scores:
                if score.get('description') == 'CURRENT' or score.get('type_id') == 1525:
                    home_score = score.get('score', {}).get('participant', 0)
                    away_score = score.get('score', {}).get('goals', 0)
            
            # Determine result
            if home_score > away_score:
                result = 'H'
            elif away_score > home_score:
                result = 'A'
            else:
                result = 'D'
            
            return {
                'date': match.get('starting_at', '')[:10],
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': home_score,
                'FTAG': away_score,
                'FTR': result,
                'league_id': match.get('league_id'),
                'match_id': match.get('id')
            }
        
        except Exception as e:
            return None
    
    def scrape_all_leagues(self) -> pd.DataFrame:
        """Scraped alle Ligen"""
        
        print("="*70)
        print("📥 SPORTMONKS HYBRID SCRAPER V4")
        print("="*70)
        print()
        
        all_matches = []
        
        for league_id in tqdm(self.config.LEAGUE_IDS, desc="Ligen"):
            matches = self.fetch_matches(league_id, self.config.START_DATE, self.config.END_DATE)
            
            for match in tqdm(matches, desc=f"Liga {league_id}", leave=False):
                parsed = self.parse_match(match)
                if parsed:
                    all_matches.append(parsed)
        
        df = pd.DataFrame(all_matches)
        
        if not df.empty:
            # Clean data
            df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    
    scraper = SportmonksScraper(config)
    df = scraper.scrape_all_leagues()
    
    if df.empty:
        print("❌ Keine Daten gescraped!")
        return
    
    # Save
    df.to_csv(config.OUTPUT_FILE, index=False)
    
    print("\n" + "="*70)
    print("✅ SCRAPING ABGESCHLOSSEN!")
    print("="*70)
    print(f"\n   Matches gescraped: {len(df)}")
    print(f"   Zeitraum: {df['date'].min()} bis {df['date'].max()}")
    print(f"   Ligen: {df['league_id'].nunique()}")
    print(f"\n   💾 Gespeichert: {config.OUTPUT_FILE}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
