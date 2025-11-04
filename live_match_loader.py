"""
LIVE MATCH DATA LOADER & ANALYZER
==================================
Lädt echte Daten aus der Datenbank und berechnet Betting Edges

Datenquellen:
- game_database_complete.csv (vom Hybrid Scraper)
- Sportmonks API (für Live-Updates)

Features:
- Live Match Detection
- Edge Calculation (Value Betting)
- Betting Recommendations
- Filter & Sort
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveMatchLoader:
    """Lädt und analysiert Live-Match-Daten"""
    
    def __init__(self, data_file: str = "game_database_complete.csv"):
        self.data_file = Path(data_file)
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Lade Daten aus CSV"""
        try:
            if self.data_file.exists():
                self.df = pd.read_csv(self.data_file)
                
                # Parse Datum
                self.df['date'] = pd.to_datetime(self.df['date'])
                
                logger.info(f"✅ Loaded {len(self.df)} matches from {self.data_file}")
            else:
                logger.warning(f"⚠️ Data file not found: {self.data_file}")
                self.df = pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def get_live_matches(self, 
                        time_window_hours: int = 2,
                        include_upcoming: bool = True) -> pd.DataFrame:
        """
        Hole Live & Upcoming Matches
        
        Args:
            time_window_hours: Zeitfenster für "Live" Spiele
            include_upcoming: Auch kommende Spiele inkludieren
        
        Returns:
            DataFrame mit Live/Upcoming Matches
        """
        if self.df.empty:
            logger.warning("⚠️ No data available")
            return pd.DataFrame()
        
        now = datetime.now()
        
        # Filter: Spiele von jetzt - time_window bis jetzt + time_window
        time_start = now - timedelta(hours=time_window_hours)
        time_end = now + timedelta(hours=time_window_hours if include_upcoming else 0)
        
        # Filter nach Zeitfenster
        mask = (self.df['date'] >= time_start) & (self.df['date'] <= time_end)
        
        df_live = self.df[mask].copy()
        
        if df_live.empty:
            logger.info(f"ℹ️ No live matches in window {time_start} to {time_end}")
            return df_live
        
        # Berechne Match-Status
        df_live['match_status'] = df_live['date'].apply(
            lambda x: self._get_match_status(x, now)
        )
        
        # Berechne Spielminute (geschätzt)
        df_live['minute'] = df_live['date'].apply(
            lambda x: self._estimate_minute(x, now)
        )
        
        logger.info(f"✅ Found {len(df_live)} live/upcoming matches")
        
        return df_live
    
    def _get_match_status(self, match_time: datetime, now: datetime) -> str:
        """Bestimme Match-Status"""
        diff_minutes = (now - match_time).total_seconds() / 60
        
        if diff_minutes < -10:
            return "SCHEDULED"
        elif diff_minutes < 0:
            return "PRE-MATCH"
        elif diff_minutes <= 45:
            return "FIRST_HALF"
        elif diff_minutes <= 60:
            return "HALF_TIME"
        elif diff_minutes <= 105:
            return "SECOND_HALF"
        elif diff_minutes <= 120:
            return "FINISHED"
        else:
            return "FINISHED"
    
    def _estimate_minute(self, match_time: datetime, now: datetime) -> str:
        """Schätze aktuelle Spielminute"""
        diff_minutes = int((now - match_time).total_seconds() / 60)
        
        if diff_minutes < 0:
            return f"-{abs(diff_minutes)}'"  # Noch nicht gestartet
        elif diff_minutes <= 45:
            return f"{diff_minutes}'"
        elif diff_minutes <= 60:
            return "HT"  # Halbzeit
        elif diff_minutes <= 105:
            minute = diff_minutes - 15  # 15min Pause
            return f"{min(minute, 90)}'"
        else:
            return "FT"  # Finished
    
    def calculate_betting_edge(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechne Betting Edge basierend auf xG vs Odds
        
        Value Betting Formula:
        1. Berechne "True Probability" aus xG
        2. Berechne "Implied Probability" aus Odds
        3. Edge = True Probability - Implied Probability
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. TRUE PROBABILITY aus xG (Poisson-Modell)
        df['prob_home_xg'] = df.apply(
            lambda row: self._calculate_win_probability(
                row['home_xg'], row['away_xg'], 'home'
            ) if pd.notna(row['home_xg']) and pd.notna(row['away_xg']) else None,
            axis=1
        )
        
        df['prob_draw_xg'] = df.apply(
            lambda row: self._calculate_win_probability(
                row['home_xg'], row['away_xg'], 'draw'
            ) if pd.notna(row['home_xg']) and pd.notna(row['away_xg']) else None,
            axis=1
        )
        
        df['prob_away_xg'] = df.apply(
            lambda row: self._calculate_win_probability(
                row['home_xg'], row['away_xg'], 'away'
            ) if pd.notna(row['home_xg']) and pd.notna(row['away_xg']) else None,
            axis=1
        )
        
        # 2. IMPLIED PROBABILITY aus Odds
        df['prob_home_odds'] = df['odds_home'].apply(
            lambda x: 1/x if pd.notna(x) and x > 0 else None
        )
        df['prob_draw_odds'] = df['odds_draw'].apply(
            lambda x: 1/x if pd.notna(x) and x > 0 else None
        )
        df['prob_away_odds'] = df['odds_away'].apply(
            lambda x: 1/x if pd.notna(x) and x > 0 else None
        )
        
        # 3. EDGE Berechnung
        df['edge_home'] = (df['prob_home_xg'] - df['prob_home_odds']) * 100
        df['edge_draw'] = (df['prob_draw_xg'] - df['prob_draw_odds']) * 100
        df['edge_away'] = (df['prob_away_xg'] - df['prob_away_odds']) * 100
        
        # 4. BESTE WETTE identifizieren
        df['best_bet'] = df.apply(self._identify_best_bet, axis=1)
        df['best_edge'] = df[['edge_home', 'edge_draw', 'edge_away']].max(axis=1)
        
        return df
    
    def _calculate_win_probability(self, home_xg: float, away_xg: float, 
                                   outcome: str) -> float:
        """
        Berechne Gewinnwahrscheinlichkeit mit Poisson-Verteilung
        
        Basierend auf xG (Expected Goals)
        """
        from scipy.stats import poisson
        
        # Maximum Tore die wir berechnen
        max_goals = 8
        
        prob = 0.0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                # Wahrscheinlichkeit für exakt i:j
                p_home_i = poisson.pmf(i, home_xg)
                p_away_j = poisson.pmf(j, away_xg)
                p_score = p_home_i * p_away_j
                
                # Addiere zur Gesamtwahrscheinlichkeit wenn Bedingung erfüllt
                if outcome == 'home' and i > j:
                    prob += p_score
                elif outcome == 'draw' and i == j:
                    prob += p_score
                elif outcome == 'away' and j > i:
                    prob += p_score
        
        return prob
    
    def _identify_best_bet(self, row: pd.Series) -> str:
        """Identifiziere beste Wette basierend auf Edge"""
        edges = {
            'home': row.get('edge_home', -100),
            'draw': row.get('edge_draw', -100),
            'away': row.get('edge_away', -100)
        }
        
        # Finde Maximum Edge
        best = max(edges.items(), key=lambda x: x[1] if pd.notna(x[1]) else -100)
        
        if best[1] > 5:  # Mindestens 5% Edge
            recommendations = {
                'home': f"BACK Home @ {row.get('odds_home', 0):.2f}",
                'draw': f"BACK Draw @ {row.get('odds_draw', 0):.2f}",
                'away': f"BACK Away @ {row.get('odds_away', 0):.2f}"
            }
            return recommendations[best[0]]
        else:
            return "No Value"
    
    def filter_matches(self, df: pd.DataFrame,
                      league: Optional[str] = None,
                      min_edge: float = 0.0,
                      market: Optional[str] = None) -> pd.DataFrame:
        """
        Filtere Matches nach Kriterien
        
        Args:
            df: DataFrame mit Matches
            league: Filter nach Liga (z.B. "Premier League")
            min_edge: Minimum Edge %
            market: Filter nach Market (z.B. "Match Winner")
        
        Returns:
            Gefiltertes DataFrame
        """
        if df.empty:
            return df
        
        df_filtered = df.copy()
        
        # Liga-Filter
        if league and league != "All":
            df_filtered = df_filtered[df_filtered['league'] == league]
        
        # Edge-Filter
        if min_edge > 0:
            df_filtered = df_filtered[df_filtered['best_edge'] >= min_edge]
        
        # Market-Filter (TODO: Erweitern für Over/Under, BTTS, etc.)
        # Aktuell nur Match Winner implementiert
        
        return df_filtered
    
    def format_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formatiere DataFrame für Dashboard-Anzeige
        
        Returns:
            DataFrame mit formatierten Spalten für UI
        """
        if df.empty:
            return pd.DataFrame({
                'Match': [],
                'League': [],
                'Time': [],
                'Score': [],
                'xG': [],
                'Best Odds': [],
                'Edge %': [],
                'Recommendation': []
            })
        
        df_display = pd.DataFrame({
            'Match': df['home_team'] + ' vs ' + df['away_team'],
            'League': df['league'],
            'Time': df['minute'],
            'Score': df['home_score'].fillna(0).astype(int).astype(str) + '-' + 
                     df['away_score'].fillna(0).astype(int).astype(str),
            'xG': df['home_xg'].round(1).astype(str) + ' - ' + 
                  df['away_xg'].round(1).astype(str),
            'Best Odds': df['odds_home'].round(2).astype(str) + ' | ' +
                        df['odds_draw'].round(2).astype(str) + ' | ' +
                        df['odds_away'].round(2).astype(str),
            'Edge %': df['best_edge'].round(1),
            'Recommendation': df['best_bet']
        })
        
        # Sortiere nach Edge (höchste zuerst)
        df_display = df_display.sort_values('Edge %', ascending=False)
        
        return df_display


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_live_matches(
    data_file: str = "game_database_complete.csv",
    time_window_hours: int = 2,
    min_edge: float = 0.0,
    league: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience Function: Lade & Analysiere Live Matches in einem Schritt
    
    Args:
        data_file: Path zur CSV-Datei
        time_window_hours: Zeitfenster für Live-Spiele
        min_edge: Minimum Betting Edge %
        league: Filter nach Liga
    
    Returns:
        Formatiertes DataFrame für Dashboard
    """
    loader = LiveMatchLoader(data_file)
    
    # Lade Live Matches
    df_live = loader.get_live_matches(time_window_hours=time_window_hours)
    
    if df_live.empty:
        logger.info("ℹ️ No live matches found")
        return pd.DataFrame()
    
    # Berechne Edges
    df_analyzed = loader.calculate_betting_edge(df_live)
    
    # Filter
    df_filtered = loader.filter_matches(
        df_analyzed,
        league=league,
        min_edge=min_edge
    )
    
    # Format für Display
    df_display = loader.format_for_display(df_filtered)
    
    return df_display


def get_available_leagues(data_file: str = "game_database_complete.csv") -> List[str]:
    """Hole verfügbare Ligen aus Datenbank"""
    try:
        df = pd.read_csv(data_file)
        leagues = ["All"] + sorted(df['league'].unique().tolist())
        return leagues
    except Exception as e:
        logger.error(f"Error loading leagues: {e}")
        return ["All"]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🎯 LIVE MATCH LOADER - TEST")
    print("=" * 70)
    
    # Test 1: Load Live Matches
    print("\n1️⃣ Loading Live Matches...")
    df = load_live_matches(
        time_window_hours=24,  # Großes Fenster für Test
        min_edge=5.0
    )
    
    if not df.empty:
        print(f"\n✅ Found {len(df)} matches with edge >= 5%")
        print("\nTop 5 Matches:")
        print(df.head().to_string())
    else:
        print("\n⚠️ No matches found (normal if no live games)")
    
    # Test 2: Get Leagues
    print("\n2️⃣ Available Leagues:")
    leagues = get_available_leagues()
    for league in leagues:
        print(f"   - {league}")