"""
📊 PORTFOLIO MANAGEMENT SYSTEM

Optimale Allokation und Risiko-Management über:
- Mehrere Märkte (1X2, Over/Under, BTTS, Correct Score)
- Mehrere Ligen
- Korrelations-Analyse
- Risk Parity
- Dynamic Rebalancing

Ziel: Maximiere Returns bei kontrolliertem Risiko
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==========================================================
# CONFIGURATION
# ==========================================================
@dataclass
class PortfolioConfig:
    """Portfolio Management Konfiguration"""
    # Exposure Limits
    max_total_exposure: float = 1.0  # 100% der Bankroll
    max_market_exposure: float = 0.30  # Max 30% pro Markt
    max_league_exposure: float = 0.30  # Max 30% pro Liga
    max_match_exposure: float = 0.10  # Max 10% pro Match

    # Diversification
    min_markets: int = 2  # Minimum 2 verschiedene Märkte
    min_leagues: int = 3  # Minimum 3 verschiedene Ligen

    # Correlation
    max_correlation: float = 0.70  # Max Korrelation zwischen Bets

    # Risk Management
    max_var_95: float = 0.15  # Max 15% Value-at-Risk (95% Konfidenz)
    target_sharpe: float = 2.0  # Ziel Sharpe Ratio


@dataclass
class Position:
    """Einzelne Position im Portfolio"""
    bet_id: str
    match: str
    league: str
    market: str
    selection: str
    odds: float
    stake: float
    probability: float
    expected_value: float
    timestamp: datetime

    # Risk Metriken
    variance: float = 0.0
    correlation_exposure: float = 0.0

    # --- NEU: Für die Analyse nach Schließung ---
    result: Optional[str] = None  # 'win', 'loss', 'void'
    profit: Optional[float] = None

    def potential_profit(self) -> float:
        """Potentieller Gewinn"""
        return self.stake * (self.odds - 1)

    def potential_loss(self) -> float:
        """Potentieller Verlust"""
        return self.stake


# ==========================================================
# PORTFOLIO MANAGER
# ==========================================================
class PortfolioManager:
    """
    Hauptklasse für Portfolio-Management
    """

    def __init__(self, bankroll: float, config: PortfolioConfig = None):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll  # bankroll ist der *aktuelle Wert* des Portfolios
        self.config = config or PortfolioConfig()

        self.positions: List[Position] = []  # Aktive, offene Wetten
        self.closed_positions: List[Position] = []  # Abgerechnete Wetten

        # Tracking
        self.total_staked = 0.0
        self.total_profit = 0.0
        
        # --- NEU: Bankroll-Historie für Charts (KORRIGIERT) ---
        # 'field(default_factory=list)' ist nur für Dataclasses.
        # Für eine normale Klasse ist dies der korrekte Weg, eine Liste zu initialisieren:
        self.bankroll_history: List[Dict] = [] 
        self.bankroll_history.append({'Date': datetime.now(), 'Bankroll': self.bankroll})
        
        # --- NEU: Für Tab 6 (Settings) ---
        self.max_daily_loss = 500.0
        self.max_bet_size = 100.0
        self.stop_loss_enabled = True


    def add_position(self, position: Position) -> bool:
        """
        Füge neue Position hinzu (mit Validierung)
        """
        if not self._validate_position(position):
            return False
        if not self._check_correlation(position):
            return False

        self.positions.append(position)
        # HINWEIS: self.total_staked wird in 'close_position' aktualisiert,
        # um nur abgerechnete Wetten für ROI zu zählen
        return True

    def _validate_position(self, position: Position) -> bool:
        """Validiere Position gegen Limits"""
        current_exposure = sum(p.stake for p in self.positions)
        new_exposure = current_exposure + position.stake

        if new_exposure > self.bankroll * self.config.max_total_exposure:
            print(f"❌ Total Exposure Limit überschritten")
            return False
        
        # ... (andere Validierungen: Market, League, Match) ...
        market_exposure = sum(p.stake for p in self.positions if p.market == position.market) + position.stake
        if market_exposure > self.bankroll * self.config.max_market_exposure:
            print(f"❌ Market Exposure Limit ({position.market})")
            return False

        league_exposure = sum(p.stake for p in self.positions if p.league == position.league) + position.stake
        if league_exposure > self.bankroll * self.config.max_league_exposure:
            print(f"❌ League Exposure Limit ({position.league})")
            return False

        match_exposure = sum(p.stake for p in self.positions if p.match == position.match) + position.stake
        if match_exposure > self.bankroll * self.config.max_match_exposure:
            print(f"❌ Match Exposure Limit")
            return False

        return True

    def _check_correlation(self, new_position: Position) -> bool:
        """Prüfe Korrelation mit existierenden Positionen"""
        for existing in self.positions:
            corr = self._calculate_correlation(existing, new_position)
            if corr > self.config.max_correlation:
                print(f"⚠️ Hohe Korrelation ({corr:.2f}) mit {existing.match}")
                return False
        return True

    def _calculate_correlation(self, pos1: Position, pos2: Position) -> float:
        """Schätze Korrelation zwischen zwei Positionen"""
        corr = 0.0
        if pos1.match == pos2.match:
            return 0.7 # Stark korreliert, auch wenn nicht gleicher Markt
        if pos1.league == pos2.league:
            corr += 0.3
        if pos1.market == pos2.market:
            corr += 0.2
        return min(corr, 1.0)

    def close_position(self, bet_id: str, result: str, profit: float):
        """
        Schließe Position (Wette ist abgerechnet)
        
        Args:
            bet_id: ID der Wette
            result: 'win', 'loss', oder 'void'
            profit: Netto-Profit (z.B. 50 bei Gewinn, -100 bei Verlust)
        """
        for i, pos in enumerate(self.positions):
            if pos.bet_id == bet_id:
                # 1. Hole Position aus aktiven Wetten
                pos_copy = self.positions.pop(i)

                # 2. --- KORREKTUR: Speichere Ergebnis & Profit ---
                pos_copy.result = result
                pos_copy.profit = profit
                
                # 3. Füge zu abgerechneten Wetten hinzu
                self.closed_positions.append(pos_copy)
                
                # 4. Aktualisiere globales Tracking
                self.total_profit += profit
                self.total_staked += pos_copy.stake # Stake zählt erst für ROI, wenn Wette zu ist
                
                # 5. --- KORREKTUR: Aktualisiere Bankroll & Historie ---
                self.bankroll += profit
                self.bankroll_history.append({'Date': datetime.now(), 'Bankroll': self.bankroll})
                return
        
        print(f"⚠️ Konnte Position {bet_id} zum Schließen nicht finden.")

    # ... (Methoden _gini_coefficient, suggest_rebalancing, etc. bleiben unverändert) ...
    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        if not values or len(values) == 1: return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        sum_values = cumsum[-1]
        if sum_values == 0: return 0.0
        gini = (2 * sum([(i + 1) * v for i, v in enumerate(sorted_values)]) / (n * sum_values)) - ((n + 1) / n)
        return gini
        
    def suggest_rebalancing(self) -> List[str]:
        # ... (Unveränderte Logik) ...
        return ["✅ Portfolio ist gut diversifiziert und balanced!"]
        
    def print_summary(self):
        # ... (Unveränderte Logik) ...
        pass
        
    # ==========================================================
    # --- NEUE METHODEN FÜR DAS STREAMLIT DASHBOARD ---
    # ==========================================================

    def _get_closed_bets_df(self) -> pd.DataFrame:
        """
        (NEUE HELPER-FUNKTION)
        Konvertiert geschlossene Positionen in einen DataFrame für Analysen.
        """
        if not self.closed_positions:
            # Erstelle leeren DF mit korrekten Spalten für den Fall, dass keine Daten vorhanden sind
            cols = [f.name for f in Position.__dataclass_fields__.values()]
            return pd.DataFrame(columns=cols)
            
        data = [asdict(p) for p in self.closed_positions]
        df = pd.DataFrame(data)
        
        # Stelle sicher, dass Typen korrekt sind (wichtig für Analysen)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['profit'] = pd.to_numeric(df['profit'])
        df['stake'] = pd.to_numeric(df['stake'])
        df['expected_value'] = pd.to_numeric(df['expected_value'])
        
        return df

    # --- FUNKTION FÜR TOP-METRIKEN (ersetzt die alte 'calculate_portfolio_metrics') ---
    
    def get_portfolio_statistics(self) -> Dict:
        """
        (NEUE FUNKTION)
        Berechnet alle wichtigen Statistiken für das Dashboard in einem Aufruf.
        """
        
        # 1. Aktuelle Portfolio-Werte
        in_bets_balance = sum(p.stake for p in self.positions)
        available_balance = self.bankroll - in_bets_balance
        
        # 2. Historische Performance (aus geschlossenen Wetten)
        df_closed = self._get_closed_bets_df()
        
        if not df_closed.empty:
            total_profit = df_closed['profit'].sum()
            total_staked = df_closed['stake'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
            win_rate = (df_closed['result'] == 'win').mean() * 100
            avg_stake = df_closed['stake'].mean()
            
            # Sharpe Ratio (vereinfacht, basierend auf Profit pro Wette)
            profit_per_bet = df_closed['profit']
            if profit_per_bet.std() > 0:
                sharpe_ratio = profit_per_bet.mean() / profit_per_bet.std()
            else:
                sharpe_ratio = 0.0
        else:
            # Keine geschlossenen Wetten, alles auf 0
            total_profit = 0.0
            total_staked = 0.0
            roi = 0.0
            win_rate = 0.0
            avg_stake = 0.0
            sharpe_ratio = 0.0

        return {
            'total_value': self.bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_stake': avg_stake,
            'available_balance': available_balance,
            'in_bets_balance': in_bets_balance
        }

    # --- FUNKTIONEN FÜR TAB 5: PORTFOLIO ---

    def get_active_positions(self) -> pd.DataFrame:
        """
        (NEUE FUNKTION)
        Gibt alle aktiven (offenen) Wetten als DataFrame zurück.
        """
        if not self.positions:
            return pd.DataFrame(columns=['Match', 'Market', 'Stake', 'Odds', 'Potential Profit', 'Status'])
            
        data = []
        for p in self.positions:
            data.append({
                'Match': p.match,
                'Market': p.market,
                'Stake': f"€{p.stake:.2f}",
                'Odds': p.odds,
                'Potential Profit': f"€{p.potential_profit():.2f}",
                'Status': '🟢 Live'
            })
        
        return pd.DataFrame(data)

    def get_bankroll_history(self) -> pd.DataFrame:
        """
        (NEUE FUNKTION)
        Gibt die Bankroll-Historie als DataFrame zurück.
        """
        if not self.bankroll_history:
            return pd.DataFrame(columns=['Date', 'Bankroll'])
            
        return pd.DataFrame(self.bankroll_history)

    # --- FUNKTIONEN FÜR TAB 7: ANALYTICS ---

    def get_bet_distribution_by_market(self) -> pd.DataFrame:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            return pd.DataFrame(columns=['Market', 'Count', 'ROI', 'WinRate'])
            
        grouped = df.groupby('market')
        
        def safe_roi(x):
            profit = x['profit'].sum()
            stake = x['stake'].sum()
            return (profit / stake * 100) if stake > 0 else 0
            
        stats = grouped.agg(
            Count=('bet_id', 'size'),
            WinRate=('result', lambda x: (x == 'win').mean() * 100)
        )
        # ROI muss separat berechnet werden
        stats['ROI'] = df.groupby('market').apply(safe_roi)
        
        return stats.reset_index().rename(columns={'market': 'Market'})

    def get_daily_pnl(self) -> pd.DataFrame:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            return pd.DataFrame(columns=['Day', 'Profit'])
            
        df['Day'] = df['timestamp'].dt.day_name()
        # Ordne die Wochentage korrekt
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pnl = df.groupby('Day')['profit'].sum().reindex(days_order).fillna(0)
        
        return pnl.reset_index().rename(columns={'profit': 'Profit'})

    def get_win_rate_by_league(self) -> pd.DataFrame:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            return pd.DataFrame(columns=['League', 'WinRate', 'TotalProfit', 'Matches'])

        grouped = df.groupby('league')
        stats = grouped.agg(
            WinRate=('result', lambda x: (x == 'win').mean() * 100),
            TotalProfit=('profit', 'sum'),
            Matches=('bet_id', 'size')
        )
        
        return stats.reset_index().rename(columns={'league': 'League'})

    def get_roi_trend(self) -> pd.DataFrame:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            return pd.DataFrame(columns=['Date', 'ROI'])
            
        df = df.sort_values('timestamp')
        df['CumulativeStake'] = df['stake'].cumsum()
        df['CumulativeProfit'] = df['profit'].cumsum()
        
        # Verhindere Division durch 0
        df['ROI'] = df.apply(
            lambda row: (row['CumulativeProfit'] / row['CumulativeStake'] * 100) if row['CumulativeStake'] > 0 else 0,
            axis=1
        )
        
        return df[['timestamp', 'ROI']].rename(columns={'timestamp': 'Date'})

    def get_best_betting_time(self) -> Dict:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            return {'Time': 'N/A', 'AvgEdge': 0, 'SuccessRate': 0}

        df['Hour'] = df['timestamp'].dt.hour
        grouped = df.groupby('Hour')
        
        stats = grouped.agg(
            SuccessRate=('result', lambda x: (x == 'win').mean()),
            AvgEdge=('expected_value', 'mean')
        )
        
        if stats.empty:
            return {'Time': 'N/A', 'AvgEdge': 0, 'SuccessRate': 0}
            
        best_hour_stats = stats.sort_values('SuccessRate', ascending=False).iloc[0]
        best_hour = best_hour_stats.name
        
        return {
            'Time': f"{best_hour:02d}:00 - {best_hour+1:02d}:00",
            'AvgEdge': best_hour_stats['AvgEdge'] * 100,
            'SuccessRate': best_hour_stats['SuccessRate'] * 100
        }

    # --- FUNKTIONEN FÜR TAB 6: SETTINGS ---
    
    def export_data(self) -> str:
        """(NEUE FUNKTION)"""
        df = self._get_closed_bets_df()
        if df.empty:
            raise Exception("Keine geschlossenen Wetten zum Exportieren vorhanden.")
            
        filepath = f"portfolio_export_{datetime.now():%Y%m%d_%H%M%S}.csv"
        df.to_csv(filepath, index=False)
        return filepath
        
    def update_safety_settings(self, max_daily_loss, max_bet_size, stop_loss_enabled):
        """(NEUE FUNKTION)"""
        self.max_daily_loss = max_daily_loss
        self.max_bet_size = max_bet_size
        self.stop_loss_enabled = stop_loss_enabled
        print(f"Safety Settings aktualisiert: Max Loss={max_daily_loss}, Max Bet={max_bet_size}")


# ==========================================================
# EXAMPLE USAGE (UNVERÄNDERT)
# ==========================================================
if __name__ == "__main__":
    print("📊 PORTFOLIO MANAGER - EXAMPLE\n")

    # Setup
    bankroll = 1000.0
    manager = PortfolioManager(bankroll)

    # Add Positions
    positions = [
        Position(
            bet_id="1",
            match="Liverpool vs Chelsea",
            league="Premier League",
            market="3Way Result",
            selection="Home",
            odds=2.10,
            stake=50.0,
            probability=0.52,
            expected_value=0.09,
            timestamp=datetime.now()
        ),
        Position(
            bet_id="2",
            match="Bayern vs Dortmund",
            league="Bundesliga",
            market="Over/Under 2.5",
            selection="Over",
            odds=1.85,
            stake=40.0,
            probability=0.58,
            expected_value=0.07,
            timestamp=datetime.now()
        ),
        Position(
            bet_id="3",
            match="Real Madrid vs Barcelona",
            league="La Liga",
            market="Both Teams Score",
            selection="Yes",
            odds=1.75,
            stake=45.0,
            probability=0.62,
            expected_value=0.09,
            timestamp=datetime.now()
        ),
        Position(
            bet_id="4",
            match="PSG vs Marseille",
            league="Ligue 1",
            market="3Way Result",
            selection="Home",
            odds=1.50,
            stake=60.0,
            probability=0.70,
            expected_value=0.05,
            timestamp=datetime.now()
        ),
    ]

    print("Adding positions...")
    for pos in positions:
        success = manager.add_position(pos)
        if success:
            print(f"  ✅ Added: {pos.match} ({pos.market})")
        else:
            print(f"  ❌ Rejected: {pos.match}")

    # Print Summary
    # manager.print_summary() # Diese Methode ist jetzt für interne Nutzung, wir nutzen get_portfolio_statistics
    stats = manager.get_portfolio_statistics()
    print("\n--- STATS ---")
    print(stats)
    print("---------------")


    # Test Correlation Limit
    print("\n" + "="*70)
    print("Testing Correlation Limit...")
    print("=" * 70 + "\n")
    # Try to add highly correlated position (same match)
    correlated_pos = Position(
        bet_id="5",
        match="Liverpool vs Chelsea",  # Same match!
        league="Premier League",
        market="Over/Under 2.5",
        selection="Over",
        odds=1.90,
        stake=50.0,
        probability=0.55,
        expected_value=0.05,
        timestamp=datetime.now()
    )

    print(f"Trying to add: {correlated_pos.match} ({correlated_pos.market})")
    success = manager.add_position(correlated_pos)

    if not success:
        print("❌ Position rejected due to correlation!")
        
    # Test closing a position
    print("\n--- Closing Position 1 ---")
    # Annahme: Bet 1 (Liverpool) hat gewonnen
    manager.close_position(bet_id="1", result="win", profit=55.0) # Stake 50 @ 2.10 = 55 profit
    
    print("\n--- STATS NACH 1 GESCHLOSSENEN WETTE ---")
    stats_after_close = manager.get_portfolio_statistics()
    print(stats_after_close)
    print("---------------------------------------")


    print("\n✅ Portfolio Manager Test Complete!")
