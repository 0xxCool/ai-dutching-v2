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
from dataclasses import dataclass, field
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

    Features:
    - Exposure Monitoring
    - Diversification Enforcement
    - Correlation Analysis
    - Risk Metrics (VaR, CVaR, Sharpe)
    - Rebalancing Recommendations
    """

    def __init__(self, bankroll: float, config: PortfolioConfig = None):
        self.bankroll = bankroll
        self.config = config or PortfolioConfig()

        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []

        # Tracking
        self.total_staked = 0.0
        self.total_profit = 0.0

    def add_position(self, position: Position) -> bool:
        """
        Füge neue Position hinzu (mit Validierung)

        Returns:
            True wenn Position hinzugefügt, False wenn abgelehnt
        """
        # Validierung
        if not self._validate_position(position):
            return False

        # Correlation Check
        if not self._check_correlation(position):
            return False

        # Füge hinzu
        self.positions.append(position)
        self.total_staked += position.stake

        return True

    def _validate_position(self, position: Position) -> bool:
        """Validiere Position gegen Limits"""

        # Check 1: Total Exposure
        current_exposure = sum(p.stake for p in self.positions)
        new_exposure = current_exposure + position.stake

        if new_exposure > self.bankroll * self.config.max_total_exposure:
            print(f"❌ Total Exposure Limit überschritten: {new_exposure:.2f} > {self.bankroll * self.config.max_total_exposure:.2f}")
            return False

        # Check 2: Market Exposure
        market_exposure = sum(p.stake for p in self.positions if p.market == position.market)
        market_exposure += position.stake

        if market_exposure > self.bankroll * self.config.max_market_exposure:
            print(f"❌ Market Exposure Limit ({position.market}): {market_exposure:.2f} > {self.bankroll * self.config.max_market_exposure:.2f}")
            return False

        # Check 3: League Exposure
        league_exposure = sum(p.stake for p in self.positions if p.league == position.league)
        league_exposure += position.stake

        if league_exposure > self.bankroll * self.config.max_league_exposure:
            print(f"❌ League Exposure Limit ({position.league}): {league_exposure:.2f} > {self.bankroll * self.config.max_league_exposure:.2f}")
            return False

        # Check 4: Match Exposure (mehrere Wetten auf gleiches Match)
        match_exposure = sum(p.stake for p in self.positions if p.match == position.match)
        match_exposure += position.stake

        if match_exposure > self.bankroll * self.config.max_match_exposure:
            print(f"❌ Match Exposure Limit: {match_exposure:.2f} > {self.bankroll * self.config.max_match_exposure:.2f}")
            return False

        return True

    def _check_correlation(self, new_position: Position) -> bool:
        """
        Prüfe Korrelation mit existierenden Positionen

        Returns:
            True wenn Korrelation OK
        """
        for existing in self.positions:
            corr = self._calculate_correlation(existing, new_position)

            if corr > self.config.max_correlation:
                print(f"⚠️ Hohe Korrelation ({corr:.2f}) mit {existing.match}")
                return False

        return True

    def _calculate_correlation(self, pos1: Position, pos2: Position) -> float:
        """
        Schätze Korrelation zwischen zwei Positionen

        Faktoren:
        - Gleiches Match = 1.0 (perfekte Korrelation)
        - Gleiche Liga = 0.3
        - Gleicher Markt = 0.2
        """
        corr = 0.0

        # Gleiches Match
        if pos1.match == pos2.match:
            # Gleiche Selection = 1.0
            if pos1.selection == pos2.selection:
                return 1.0
            # Unterschiedliche Selection = 0.7 (stark korreliert)
            else:
                return 0.7

        # Gleiche Liga
        if pos1.league == pos2.league:
            corr += 0.3

        # Gleicher Markt
        if pos1.market == pos2.market:
            corr += 0.2

        return min(corr, 1.0)

    def calculate_portfolio_metrics(self) -> Dict:
        """
        Berechne Portfolio-Metriken

        Returns:
            Dict mit Metriken
        """
        if not self.positions:
            return {
                'total_exposure': 0.0,
                'exposure_pct': 0.0,
                'num_positions': 0,
                'expected_value': 0.0,
                'portfolio_var_95': 0.0,
                'portfolio_sharpe': 0.0,
                'diversification_score': 0.0
            }

        # Basis-Metriken
        total_exposure = sum(p.stake for p in self.positions)
        exposure_pct = (total_exposure / self.bankroll * 100)
        num_positions = len(self.positions)

        # Expected Value
        expected_value = sum(p.expected_value * p.stake for p in self.positions)

        # Value-at-Risk (95%)
        # Vereinfachte Berechnung: Summe der potentiellen Verluste
        potential_losses = [p.potential_loss() for p in self.positions]
        portfolio_var_95 = np.percentile(potential_losses, 95) if potential_losses else 0

        # Sharpe Ratio (geschätzt)
        if total_exposure > 0:
            expected_return = expected_value / total_exposure
            # Variance basierend auf Odds
            returns_variance = np.var([p.probability * (p.odds - 1) for p in self.positions])
            portfolio_sharpe = expected_return / np.sqrt(returns_variance) if returns_variance > 0 else 0
        else:
            portfolio_sharpe = 0

        # Diversification Score (0-1, höher ist besser)
        diversification = self._calculate_diversification_score()

        return {
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct,
            'num_positions': num_positions,
            'expected_value': expected_value,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_sharpe': portfolio_sharpe,
            'diversification_score': diversification
        }

    def _calculate_diversification_score(self) -> float:
        """
        Berechne Diversification Score (0-1)

        Faktoren:
        - Anzahl verschiedener Märkte
        - Anzahl verschiedener Ligen
        - Gleichmäßigkeit der Allokation
        """
        if not self.positions:
            return 0.0

        score = 0.0

        # Faktor 1: Anzahl Märkte
        markets = set(p.market for p in self.positions)
        market_score = min(len(markets) / 4.0, 1.0)  # Ideal: 4+ Märkte
        score += market_score * 0.3

        # Faktor 2: Anzahl Ligen
        leagues = set(p.league for p in self.positions)
        league_score = min(len(leagues) / 5.0, 1.0)  # Ideal: 5+ Ligen
        score += league_score * 0.3

        # Faktor 3: Gleichmäßigkeit (Gini-Koeffizient)
        stakes = [p.stake for p in self.positions]
        gini = self._gini_coefficient(stakes)
        uniformity_score = 1 - gini  # Niedriger Gini = besser
        score += uniformity_score * 0.4

        return score

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Berechne Gini-Koeffizient (0=perfekte Gleichheit, 1=perfekte Ungleichheit)"""
        if not values or len(values) == 1:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        sum_values = cumsum[-1]

        if sum_values == 0:
            return 0.0

        gini = (2 * sum([(i + 1) * v for i, v in enumerate(sorted_values)]) / (n * sum_values)) - ((n + 1) / n)

        return gini

    def get_exposure_breakdown(self) -> Dict:
        """Detaillierte Exposure-Aufschlüsselung"""
        if not self.positions:
            return {}

        # By Market
        market_exposure = {}
        for p in self.positions:
            market_exposure[p.market] = market_exposure.get(p.market, 0) + p.stake

        # By League
        league_exposure = {}
        for p in self.positions:
            league_exposure[p.league] = league_exposure.get(p.league, 0) + p.stake

        # By Match
        match_exposure = {}
        for p in self.positions:
            match_exposure[p.match] = match_exposure.get(p.match, 0) + p.stake

        return {
            'by_market': market_exposure,
            'by_league': league_exposure,
            'by_match': match_exposure
        }

    def suggest_rebalancing(self) -> List[str]:
        """
        Schlage Rebalancing-Aktionen vor

        Returns:
            Liste von Empfehlungen
        """
        recommendations = []

        # Exposure Check
        metrics = self.calculate_portfolio_metrics()

        if metrics['exposure_pct'] > 90:
            recommendations.append("🔴 Warnung: Portfolio >90% exposed. Reduziere neue Positionen.")

        # Diversification Check
        if metrics['diversification_score'] < 0.5:
            recommendations.append("⚠️ Niedrige Diversifikation. Füge Positionen in anderen Märkten/Ligen hinzu.")

        # Sharpe Check
        if metrics['portfolio_sharpe'] < self.config.target_sharpe:
            recommendations.append(f"📉 Sharpe Ratio ({metrics['portfolio_sharpe']:.2f}) unter Ziel ({self.config.target_sharpe}). Optimiere Stake-Sizing.")

        # Market Concentration
        exposure_breakdown = self.get_exposure_breakdown()

        for market, exposure in exposure_breakdown['by_market'].items():
            limit = self.bankroll * self.config.max_market_exposure

            if exposure > limit * 0.9:  # 90% vom Limit
                recommendations.append(f"⚠️ Market '{market}' nahe am Limit ({exposure:.2f} / {limit:.2f})")

        # League Concentration
        for league, exposure in exposure_breakdown['by_league'].items():
            limit = self.bankroll * self.config.max_league_exposure

            if exposure > limit * 0.9:
                recommendations.append(f"⚠️ Liga '{league}' nahe am Limit ({exposure:.2f} / {limit:.2f})")

        if not recommendations:
            recommendations.append("✅ Portfolio ist gut diversifiziert und balanced!")

        return recommendations

    def close_position(self, bet_id: str, profit: float):
        """Schließe Position"""
        for i, pos in enumerate(self.positions):
            if pos.bet_id == bet_id:
                pos_copy = self.positions.pop(i)
                self.closed_positions.append(pos_copy)
                self.total_profit += profit
                break

    def print_summary(self):
        """Ausgabe Portfolio-Zusammenfassung"""
        print(f"\n{'='*70}")
        print("📊 PORTFOLIO SUMMARY")
        print(f"{'='*70}")

        print(f"\n💰 Bankroll: €{self.bankroll:.2f}")

        metrics = self.calculate_portfolio_metrics()

        print(f"\n📈 Positions:")
        print(f"  Active: {metrics['num_positions']}")
        print(f"  Total Exposure: €{metrics['total_exposure']:.2f} ({metrics['exposure_pct']:.1f}%)")
        print(f"  Expected Value: €{metrics['expected_value']:.2f}")

        print(f"\n⚠️  Risk Metrics:")
        print(f"  VaR (95%): €{metrics['portfolio_var_95']:.2f}")
        print(f"  Sharpe Ratio: {metrics['portfolio_sharpe']:.2f}")
        print(f"  Diversification: {metrics['diversification_score']:.2%}")

        # Exposure Breakdown
        breakdown = self.get_exposure_breakdown()

        print(f"\n📊 Exposure by Market:")
        for market, exposure in sorted(breakdown['by_market'].items(), key=lambda x: x[1], reverse=True):
            pct = exposure / self.bankroll * 100
            print(f"  {market}: €{exposure:.2f} ({pct:.1f}%)")

        print(f"\n🌍 Exposure by League:")
        for league, exposure in sorted(breakdown['by_league'].items(), key=lambda x: x[1], reverse=True):
            pct = exposure / self.bankroll * 100
            print(f"  {league}: €{exposure:.2f} ({pct:.1f}%)")

        # Recommendations
        print(f"\n💡 Recommendations:")
        recs = self.suggest_rebalancing()
        for rec in recs:
            print(f"  {rec}")

        print(f"\n{'='*70}\n")


# ==========================================================
# EXAMPLE USAGE
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
    manager.print_summary()

    # Test Correlation Limit
    print("\n" + "="*70)
    print("Testing Correlation Limit...")
    print("="*70 + "\n")

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

    print("\n✅ Portfolio Manager Test Complete!")
