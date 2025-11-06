"""
Output Formatter für Sportmonks Dutching System
================================================
Verbessert die Lesbarkeit und Übersichtlichkeit der Ausgaben
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json

class OutputFormatter:
    """
    Formatiert die Ausgaben des Dutching Systems übersichtlich
    """
    
    @staticmethod
    def format_currency(value: float, currency: str = "€") -> str:
        """Formatiert Währungsbeträge"""
        if pd.isna(value):
            return "-"
        return f"{currency}{value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2, show_sign: bool = True) -> str:
        """Formatiert Prozentwerte"""
        if pd.isna(value):
            return "-"
        sign = "+" if show_sign and value > 0 else ""
        return f"{sign}{value:.{decimals}f}%"
    
    @staticmethod
    def format_odds(value: float, decimals: int = 2) -> str:
        """Formatiert Odds"""
        if pd.isna(value):
            return "-"
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_probability(value: float, as_percentage: bool = True) -> str:
        """Formatiert Wahrscheinlichkeiten"""
        if pd.isna(value):
            return "-"
        if as_percentage:
            return f"{value * 100:.1f}%"
        return f"{value:.4f}"
    
    @staticmethod
    def color_code_value(value: float, thresholds: Dict[str, float] = None) -> str:
        """
        Gibt einen Farbcode basierend auf Schwellenwerten zurück
        
        Args:
            value: Der zu bewertende Wert
            thresholds: Dict mit 'good', 'neutral', 'bad' Schwellenwerten
        
        Returns:
            String mit Farb-Emoji
        """
        if pd.isna(value):
            return "⚪"
        
        if thresholds is None:
            thresholds = {'good': 5, 'bad': -5}
        
        if value >= thresholds.get('good', 5):
            return "🟢"  # Grün für gute Werte
        elif value <= thresholds.get('bad', -5):
            return "🔴"  # Rot für schlechte Werte
        else:
            return "🟡"  # Gelb für neutrale Werte
    
    @staticmethod
    def create_match_summary(row: pd.Series) -> str:
        """
        Erstellt eine lesbare Match-Zusammenfassung
        
        Args:
            row: Pandas Series mit Match-Daten
        
        Returns:
            Formatierter String
        """
        summary_parts = []
        
        # Match Info
        match_info = f"🆚 {row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}"
        summary_parts.append(match_info)
        
        # League und Zeit
        if 'league' in row:
            summary_parts.append(f"🏆 {row['league']}")
        if 'datetime' in row:
            try:
                dt = pd.to_datetime(row['datetime'])
                summary_parts.append(f"📅 {dt.strftime('%d.%m.%Y %H:%M')}")
            except:
                summary_parts.append(f"📅 {row['datetime']}")
        
        # Market
        if 'market' in row:
            summary_parts.append(f"🎯 {row['market']}")
        
        # Odds
        if 'odds' in row:
            summary_parts.append(f"📊 Odds: {OutputFormatter.format_odds(row['odds'])}")
        
        # Expected Value
        if 'expected_value' in row:
            ev = row['expected_value']
            color = OutputFormatter.color_code_value(ev)
            summary_parts.append(f"{color} EV: {OutputFormatter.format_percentage(ev)}")
        
        # Stake
        if 'stake' in row:
            summary_parts.append(f"💰 Einsatz: {OutputFormatter.format_currency(row['stake'])}")
        
        # Potential Profit
        if 'potential_profit' in row:
            summary_parts.append(f"💵 Potentieller Gewinn: {OutputFormatter.format_currency(row['potential_profit'])}")
        
        return " | ".join(summary_parts)
    
    @staticmethod
    def format_results_dataframe(df: pd.DataFrame, 
                                 sort_by: str = 'expected_value',
                                 ascending: bool = False,
                                 top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Formatiert ein Results-DataFrame für bessere Lesbarkeit
        
        Args:
            df: Input DataFrame
            sort_by: Spalte zum Sortieren
            ascending: Sortierreihenfolge
            top_n: Anzahl der Top-Einträge (None = alle)
        
        Returns:
            Formatierter DataFrame
        """
        if df.empty:
            return df
        
        # Kopiere DataFrame
        formatted_df = df.copy()
        
        # Sortieren
        if sort_by in formatted_df.columns:
            formatted_df = formatted_df.sort_values(by=sort_by, ascending=ascending)
        
        # Top N auswählen
        if top_n:
            formatted_df = formatted_df.head(top_n)
        
        # Formatiere Spalten
        display_df = formatted_df.copy()
        
        # Currency Columns
        currency_cols = ['stake', 'potential_profit', 'expected_profit', 'bankroll']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(OutputFormatter.format_currency)
        
        # Percentage Columns
        percentage_cols = ['expected_value', 'probability', 'kelly_fraction', 'roi']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: OutputFormatter.format_percentage(x) if pd.notna(x) else '-')
        
        # Odds Columns
        odds_cols = ['odds', 'fair_odds']
        for col in odds_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: OutputFormatter.format_odds(x) if pd.notna(x) else '-')
        
        # Datetime Columns
        if 'datetime' in display_df.columns:
            display_df['datetime'] = pd.to_datetime(display_df['datetime']).dt.strftime('%d.%m.%Y %H:%M')
        
        # Füge Quality-Indicator hinzu
        if 'expected_value' in formatted_df.columns:
            display_df.insert(0, '📊', formatted_df['expected_value'].apply(OutputFormatter.color_code_value))
        
        return display_df
    
    @staticmethod
    def create_summary_stats(df: pd.DataFrame) -> Dict[str, str]:
        """
        Erstellt Zusammenfassungsstatistiken
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dict mit formatierten Statistiken
        """
        if df.empty:
            return {
                'total_bets': '0',
                'total_stake': '-',
                'avg_ev': '-',
                'best_ev': '-',
                'total_potential_profit': '-'
            }
        
        stats = {}
        
        # Anzahl Wetten
        stats['total_bets'] = str(len(df))
        
        # Gesamteinsatz
        if 'stake' in df.columns:
            stats['total_stake'] = OutputFormatter.format_currency(df['stake'].sum())
            stats['avg_stake'] = OutputFormatter.format_currency(df['stake'].mean())
        
        # Expected Value
        if 'expected_value' in df.columns:
            stats['avg_ev'] = OutputFormatter.format_percentage(df['expected_value'].mean())
            stats['best_ev'] = OutputFormatter.format_percentage(df['expected_value'].max())
            stats['worst_ev'] = OutputFormatter.format_percentage(df['expected_value'].min())
        
        # Potential Profit
        if 'potential_profit' in df.columns:
            stats['total_potential_profit'] = OutputFormatter.format_currency(df['potential_profit'].sum())
            stats['avg_potential_profit'] = OutputFormatter.format_currency(df['potential_profit'].mean())
        
        # Odds Range
        if 'odds' in df.columns:
            stats['min_odds'] = OutputFormatter.format_odds(df['odds'].min())
            stats['max_odds'] = OutputFormatter.format_odds(df['odds'].max())
            stats['avg_odds'] = OutputFormatter.format_odds(df['odds'].mean())
        
        return stats
    
    @staticmethod
    def print_summary(df: pd.DataFrame, title: str = "Dutching Results Summary"):
        """
        Druckt eine formatierte Zusammenfassung in die Konsole
        
        Args:
            df: Results DataFrame
            title: Titel der Zusammenfassung
        """
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
        
        if df.empty:
            print("\n  ⚠️  Keine Ergebnisse verfügbar\n")
            return
        
        stats = OutputFormatter.create_summary_stats(df)
        
        print(f"\n  📊 Gesamt Wetten: {stats['total_bets']}")
        
        if 'total_stake' in stats:
            print(f"  💰 Gesamteinsatz: {stats['total_stake']} (Ø {stats.get('avg_stake', '-')})")
        
        if 'avg_ev' in stats:
            print(f"  📈 Expected Value:")
            print(f"     • Durchschnitt: {stats['avg_ev']}")
            print(f"     • Best: {stats['best_ev']}")
            print(f"     • Worst: {stats['worst_ev']}")
        
        if 'total_potential_profit' in stats:
            print(f"  💵 Potentieller Gewinn: {stats['total_potential_profit']} (Ø {stats.get('avg_potential_profit', '-')})")
        
        if 'min_odds' in stats:
            print(f"  🎲 Odds Range: {stats['min_odds']} - {stats['max_odds']} (Ø {stats['avg_odds']})")
        
        print("\n" + "=" * 80)
        
        # Top 5 Bets
        if len(df) > 0 and 'expected_value' in df.columns:
            print("\n  ⭐ Top 5 Value Bets:\n")
            top_5 = df.nlargest(5, 'expected_value')
            
            for idx, row in top_5.iterrows():
                print(f"  {idx + 1}. {OutputFormatter.create_match_summary(row)}")
            
            print("\n" + "=" * 80 + "\n")
    
    @staticmethod
    def save_formatted_report(df: pd.DataFrame, 
                             filename: str = None,
                             format: str = 'csv') -> str:
        """
        Speichert einen formatierten Report
        
        Args:
            df: Results DataFrame
            filename: Ausgabedateiname (None = auto-generate)
            format: 'csv', 'excel', oder 'json'
        
        Returns:
            Pfad zur gespeicherten Datei
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dutching_report_{timestamp}"
        
        # Formatiere DataFrame
        formatted_df = OutputFormatter.format_results_dataframe(df)
        
        # Speichere basierend auf Format
        if format == 'csv':
            filepath = f"{filename}.csv"
            formatted_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        elif format == 'excel':
            filepath = f"{filename}.xlsx"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Hauptdaten
                formatted_df.to_excel(writer, sheet_name='Results', index=False)
                
                # Statistiken
                stats = OutputFormatter.create_summary_stats(df)
                stats_df = pd.DataFrame([stats]).T
                stats_df.columns = ['Value']
                stats_df.to_excel(writer, sheet_name='Summary')
        
        elif format == 'json':
            filepath = f"{filename}.json"
            # Konvertiere zu JSON-freundlichem Format
            json_data = {
                'summary': OutputFormatter.create_summary_stats(df),
                'results': df.to_dict(orient='records')
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unbekanntes Format: {format}")
        
        print(f"\n✅ Report gespeichert: {filepath}\n")
        return filepath


# =============================================================================
# BEISPIEL-NUTZUNG
# =============================================================================

def demo_formatter():
    """Demonstriert die Nutzung des OutputFormatters"""
    
    # Beispiel-Daten
    sample_data = {
        'datetime': ['2025-11-05 20:00', '2025-11-05 20:30', '2025-11-06 15:00'],
        'league': ['Premier League', 'La Liga', 'Bundesliga'],
        'home_team': ['Arsenal', 'Real Madrid', 'Bayern Munich'],
        'away_team': ['Liverpool', 'Barcelona', 'Dortmund'],
        'market': ['1X2 - Home', '1X2 - Home', '1X2 - Away'],
        'odds': [2.10, 1.85, 2.45],
        'expected_value': [8.5, -2.3, 12.7],
        'stake': [45.20, 32.50, 58.00],
        'potential_profit': [49.72, 27.63, 84.10],
        'probability': [0.52, 0.57, 0.43]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 1. Formatierte Tabelle ausgeben
    print("\n=== 1. FORMATIERTE TABELLE ===\n")
    formatted = OutputFormatter.format_results_dataframe(df, sort_by='expected_value')
    print(formatted.to_string(index=False))
    
    # 2. Summary ausgeben
    print("\n=== 2. ZUSAMMENFASSUNG ===")
    OutputFormatter.print_summary(df, "Demo Dutching Results")
    
    # 3. Statistiken als Dict
    print("\n=== 3. STATISTIKEN ===\n")
    stats = OutputFormatter.create_summary_stats(df)
    for key, value in stats.items():
        print(f"  {key:25s}: {value}")
    
    # 4. Report speichern
    print("\n=== 4. REPORTS SPEICHERN ===")
    OutputFormatter.save_formatted_report(df, 'demo_report', format='csv')
    OutputFormatter.save_formatted_report(df, 'demo_report', format='json')


if __name__ == "__main__":
    print("🎨 Output Formatter - Demo")
    demo_formatter()