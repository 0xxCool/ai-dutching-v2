"""
Backtesting Framework für Wett-Strategien

Features:
- Historische Simulation
- P&L Tracking
- Performance-Metriken (Sharpe, Max Drawdown, ROI)
- Visualisierung
- Kelly-Criterion Backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Konfiguration für Backtest"""
    initial_bankroll: float = 1000.0
    kelly_cap: float = 0.25
    min_odds: float = 1.1
    max_odds: float = 100.0
    min_edge: float = -0.05  # Minimum Expected Value

    # Risk Management
    max_stake_percent: float = 0.10  # Max 10% pro Wette
    stop_loss_percent: float = 0.50  # Stop bei 50% Verlust
    take_profit_percent: float = 3.0  # Take Profit bei 300%

    # Tracking
    track_daily_stats: bool = True
    save_all_bets: bool = True


@dataclass
class Bet:
    """Einzelne Wette"""
    date: datetime
    match: str
    market: str
    selection: str
    odds: float
    probability: float
    stake: float
    result: Optional[str] = None  # 'win', 'loss', 'void'
    profit: Optional[float] = None
    bankroll_before: Optional[float] = None
    bankroll_after: Optional[float] = None


@dataclass
class BacktestResult:
    """Ergebnis eines Backtests"""
    # P&L
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    void_bets: int = 0

    total_staked: float = 0.0
    total_profit: float = 0.0
    final_bankroll: float = 0.0

    # Performance Metriken
    roi: float = 0.0  # Return on Investment
    win_rate: float = 0.0
    avg_odds: float = 0.0
    avg_stake: float = 0.0

    # Risk Metriken
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0

    # Bankroll-Entwicklung
    bankroll_history: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # Alle Wetten
    bets: List[Bet] = field(default_factory=list)


class AdaptiveKelly:
    """
    Adaptive Kelly-Criterion mit Bankroll-Management

    Features:
    - Dynamische Kelly-Anpassung basierend auf Performance
    - Drawdown-Protection
    - Confidence-Weighting
    """

    def __init__(
        self,
        base_kelly_cap: float = 0.25,
        min_kelly_cap: float = 0.05,
        max_kelly_cap: float = 0.35
    ):
        self.base_kelly_cap = base_kelly_cap
        self.min_kelly_cap = min_kelly_cap
        self.max_kelly_cap = max_kelly_cap

    def calculate_stake(
        self,
        bankroll: float,
        odds: float,
        probability: float,
        confidence: float = 1.0,
        current_drawdown: float = 0.0
    ) -> float:
        """
        Berechne optimalen Stake mit Adaptive Kelly

        Args:
            bankroll: Aktuelle Bankroll
            odds: Quote
            probability: Geschätzte Gewinnwahrscheinlichkeit
            confidence: Confidence Score (0-1)
            current_drawdown: Aktueller Drawdown (0-1)

        Returns:
            Empfohlener Stake
        """
        # Basis Kelly-Fraction
        edge = probability * odds - 1
        kelly_fraction = edge / (odds - 1)

        # Adaptive Kelly-Cap basierend auf Drawdown
        if current_drawdown > 0.30:  # Großer Drawdown
            kelly_cap = self.min_kelly_cap
        elif current_drawdown > 0.15:  # Moderater Drawdown
            kelly_cap = self.base_kelly_cap * 0.5
        else:  # Normal
            kelly_cap = self.base_kelly_cap

        # Confidence-Weighting
        kelly_fraction *= confidence

        # Cap
        kelly_fraction = min(kelly_fraction, kelly_cap)
        kelly_fraction = max(kelly_fraction, 0)

        # Berechne Stake
        stake = bankroll * kelly_fraction

        return stake


class Backtester:
    """
    Hauptklasse für Backtesting
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.adaptive_kelly = AdaptiveKelly(
            base_kelly_cap=self.config.kelly_cap
        )

    def _simulate_bet_result(
        self,
        odds: float,
        actual_outcome: str,
        predicted_outcome: str
    ) -> str:
        """
        Simuliere Wett-Ergebnis

        Args:
            odds: Quote
            actual_outcome: Tatsächliches Ergebnis ('Home', 'Draw', 'Away')
            predicted_outcome: Vorhergesagtes Ergebnis

        Returns:
            'win', 'loss', or 'void'
        """
        if actual_outcome == predicted_outcome:
            return 'win'
        else:
            return 'loss'

    def _calculate_drawdown(
        self,
        bankroll_history: List[float]
    ) -> Tuple[float, float]:
        """
        Berechne Maximum Drawdown

        Returns:
            (absolute_drawdown, percent_drawdown)
        """
        if len(bankroll_history) < 2:
            return 0.0, 0.0

        peak = bankroll_history[0]
        max_dd = 0.0

        for value in bankroll_history:
            if value > peak:
                peak = value

            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        max_dd_percent = (max_dd / peak * 100) if peak > 0 else 0

        return max_dd, max_dd_percent

    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Berechne Sharpe Ratio

        Args:
            returns: Liste von Returns
            risk_free_rate: Risikofreier Zinssatz (annual)

        Returns:
            Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std()

        # Annualisiert (angenommen täglich)
        sharpe_annual = sharpe * np.sqrt(252)

        return sharpe_annual

    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        prediction_func: Callable[[pd.Series], Dict]
    ) -> BacktestResult:
        """
        Führe Backtest aus

        Args:
            historical_data: DataFrame mit Spalten:
                - date, home_team, away_team, home_score, away_score
                - home_xg, away_xg (optional)
                - odds_home, odds_draw, odds_away (optional)

            prediction_func: Funktion die für jede Zeile Predictions zurückgibt
                Signature: func(row) -> Dict mit:
                    {
                        'market': str,
                        'selection': str,
                        'probability': float,
                        'confidence': float,
                        'odds': float (falls nicht in row)
                    }

        Returns:
            BacktestResult mit allen Metriken
        """
        result = BacktestResult()
        result.final_bankroll = self.config.initial_bankroll
        result.bankroll_history.append(self.config.initial_bankroll)

        bankroll = self.config.initial_bankroll
        peak_bankroll = bankroll

        print(f"\n🔄 Starte Backtest...")
        print(f"{'='*60}")
        print(f"  Initial Bankroll: €{bankroll:.2f}")
        print(f"  Matches: {len(historical_data)}")
        print(f"{'='*60}\n")

        for idx, row in historical_data.iterrows():
            # Prüfe Stop-Loss / Take-Profit
            if bankroll <= self.config.initial_bankroll * self.config.stop_loss_percent:
                print(f"\n⛔ STOP LOSS erreicht bei €{bankroll:.2f}")
                break

            if bankroll >= self.config.initial_bankroll * self.config.take_profit_percent:
                print(f"\n🎯 TAKE PROFIT erreicht bei €{bankroll:.2f}")
                break

            # Hole Prediction
            try:
                pred = prediction_func(row)
            except Exception as e:
                print(f"⚠️ Prediction Error: {e}")
                continue

            if not pred or 'probability' not in pred:
                continue

            # Extrahiere Daten
            odds = pred.get('odds', row.get('odds_home', 0))
            probability = pred['probability']
            selection = pred.get('selection', 'Unknown')
            market = pred.get('market', '3Way Result')
            confidence = pred.get('confidence', 1.0)

            # Validierung
            if odds < self.config.min_odds or odds > self.config.max_odds:
                continue

            # Berechne Expected Value
            ev = probability * odds - 1

            if ev < self.config.min_edge:
                continue

            # Berechne Drawdown
            current_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0

            # Berechne Stake (Adaptive Kelly)
            stake = self.adaptive_kelly.calculate_stake(
                bankroll=bankroll,
                odds=odds,
                probability=probability,
                confidence=confidence,
                current_drawdown=current_dd
            )

            # Max Stake Limit
            max_stake = bankroll * self.config.max_stake_percent
            stake = min(stake, max_stake)

            if stake <= 0:
                continue

            # Update Bankroll
            bankroll -= stake

            # Simuliere Ergebnis
            actual_home = row.get('home_score', 0)
            actual_away = row.get('away_score', 0)

            if actual_home > actual_away:
                actual_outcome = 'Home'
            elif actual_home < actual_away:
                actual_outcome = 'Away'
            else:
                actual_outcome = 'Draw'

            bet_result = self._simulate_bet_result(odds, actual_outcome, selection)

            # Berechne Profit/Loss
            if bet_result == 'win':
                profit = stake * (odds - 1)
                bankroll += stake + profit  # Stake zurück + Gewinn
                result.winning_bets += 1
            elif bet_result == 'loss':
                profit = -stake
                result.losing_bets += 1
            else:  # void
                bankroll += stake  # Stake zurück
                profit = 0
                result.void_bets += 1

            # Update Peak
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll

            # Erstelle Bet-Object
            bet = Bet(
                date=row['date'],
                match=f"{row['home_team']} vs {row['away_team']}",
                market=market,
                selection=selection,
                odds=odds,
                probability=probability,
                stake=stake,
                result=bet_result,
                profit=profit,
                bankroll_before=bankroll + stake if bet_result == 'win' else bankroll + stake + abs(profit),
                bankroll_after=bankroll
            )

            result.bets.append(bet)
            result.total_staked += stake
            result.total_profit += profit

            result.bankroll_history.append(bankroll)

        # Finale Berechnungen
        result.total_bets = len(result.bets)
        result.final_bankroll = bankroll
        result.roi = (result.total_profit / result.total_staked * 100) if result.total_staked > 0 else 0
        result.win_rate = (result.winning_bets / result.total_bets * 100) if result.total_bets > 0 else 0

        if result.total_bets > 0:
            result.avg_odds = np.mean([b.odds for b in result.bets])
            result.avg_stake = result.total_staked / result.total_bets

        # Risk Metriken
        result.max_drawdown, result.max_drawdown_percent = self._calculate_drawdown(
            result.bankroll_history
        )

        # Daily Returns
        if len(result.bankroll_history) > 1:
            returns = [
                (result.bankroll_history[i] - result.bankroll_history[i-1]) / result.bankroll_history[i-1]
                for i in range(1, len(result.bankroll_history))
            ]
            result.daily_returns = returns
            result.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            result.volatility = np.std(returns) * 100 if returns else 0

        return result

    def print_results(self, result: BacktestResult):
        """Ausgabe Backtest-Ergebnisse"""
        print(f"\n{'='*70}")
        print("📊 BACKTEST ERGEBNISSE")
        print(f"{'='*70}")

        print(f"\n💰 P&L:")
        print(f"  Initial Bankroll:    €{self.config.initial_bankroll:.2f}")
        print(f"  Final Bankroll:      €{result.final_bankroll:.2f}")
        print(f"  Total Profit:        €{result.total_profit:.2f}")
        print(f"  ROI:                 {result.roi:.2f}%")

        print(f"\n📈 Wett-Statistiken:")
        print(f"  Total Bets:          {result.total_bets}")
        print(f"  Winning Bets:        {result.winning_bets} ({result.win_rate:.1f}%)")
        print(f"  Losing Bets:         {result.losing_bets}")
        print(f"  Total Staked:        €{result.total_staked:.2f}")
        print(f"  Avg Stake:           €{result.avg_stake:.2f}")
        print(f"  Avg Odds:            {result.avg_odds:.2f}")

        print(f"\n⚠️  Risk-Metriken:")
        print(f"  Max Drawdown:        €{result.max_drawdown:.2f} ({result.max_drawdown_percent:.1f}%)")
        print(f"  Sharpe Ratio:        {result.sharpe_ratio:.2f}")
        print(f"  Volatility:          {result.volatility:.2f}%")

        if result.total_bets > 0:
            print(f"\n🎯 Top 5 Wetten (Profit):")
            top_bets = sorted(result.bets, key=lambda x: x.profit or 0, reverse=True)[:5]
            for i, bet in enumerate(top_bets, 1):
                print(f"  {i}. {bet.match} @ {bet.odds:.2f} → €{bet.profit:.2f}")

        print(f"\n{'='*70}\n")

    def save_results(self, result: BacktestResult, filename: str):
        """Speichere Ergebnisse als CSV"""
        if not result.bets:
            print("⚠️ Keine Wetten zum Speichern")
            return

        # Konvertiere Bets zu DataFrame
        bets_data = []
        for bet in result.bets:
            bets_data.append({
                'Date': bet.date,
                'Match': bet.match,
                'Market': bet.market,
                'Selection': bet.selection,
                'Odds': bet.odds,
                'Probability': bet.probability,
                'Stake': bet.stake,
                'Result': bet.result,
                'Profit': bet.profit,
                'Bankroll': bet.bankroll_after
            })

        df = pd.DataFrame(bets_data)
        df.to_csv(filename, index=False)

        print(f"💾 Ergebnisse gespeichert: {filename}")


# ==========================================================
# BEISPIEL
# ==========================================================
if __name__ == "__main__":
    print("🧪 Backtesting Framework Test\n")

    # Erstelle Mock Historical Data
    n_matches = 100

    historical_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n_matches),
        'home_team': ['Team A'] * 50 + ['Team B'] * 50,
        'away_team': ['Team C'] * 50 + ['Team D'] * 50,
        'home_score': np.random.randint(0, 4, n_matches),
        'away_score': np.random.randint(0, 4, n_matches),
        'home_xg': np.random.uniform(0.5, 3.0, n_matches),
        'away_xg': np.random.uniform(0.5, 3.0, n_matches),
        'odds_home': np.random.uniform(1.5, 5.0, n_matches),
        'odds_draw': np.random.uniform(2.5, 4.5, n_matches),
        'odds_away': np.random.uniform(1.5, 5.0, n_matches)
    })

    # Einfache Prediction Function (Random für Demo)
    def simple_prediction(row):
        """Simple Mock Prediction"""
        # Verwende xG als Proxy für Wahrscheinlichkeit
        home_xg = row['home_xg']
        away_xg = row['away_xg']

        total_xg = home_xg + away_xg
        prob_home = home_xg / total_xg if total_xg > 0 else 0.33

        return {
            'market': '3Way Result',
            'selection': 'Home',
            'probability': prob_home,
            'confidence': 0.7,
            'odds': row['odds_home']
        }

    # Konfiguration
    config = BacktestConfig(
        initial_bankroll=1000.0,
        kelly_cap=0.25,
        min_edge=-0.05
    )

    # Run Backtest
    backtester = Backtester(config)
    result = backtester.run_backtest(historical_data, simple_prediction)

    # Print Results
    backtester.print_results(result)

    # Save Results
    backtester.save_results(result, 'backtest_results.csv')

    print("✅ Backtest abgeschlossen!")
