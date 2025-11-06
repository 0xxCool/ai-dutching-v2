"""
Integration des OutputFormatters in das Sportmonks Dutching System
===================================================================

Dieses Modul zeigt, wie du den OutputFormatter verwendest, um
die Ausgaben deines Dutching Systems übersichtlicher zu machen.
"""

import sys
from pathlib import Path
import pandas as pd

# Füge den Pfad zum OutputFormatter hinzu
sys.path.append(str(Path.cwd()))

from output_formatter import OutputFormatter

# =============================================================================
# INTEGRATION IN SPORTMONKS SYSTEM
# =============================================================================

def enhance_dutching_output(results_file: str = 'sportmonks_results.csv'):
    """
    Nimmt die Ergebnisse aus dem Sportmonks System und
    erstellt übersichtliche Ausgaben
    
    Args:
        results_file: Pfad zur CSV-Datei mit den Dutching-Ergebnissen
    """
    
    print("\n" + "=" * 80)
    print("  🔄 LADE DUTCHING-ERGEBNISSE")
    print("=" * 80 + "\n")
    
    # Lade Ergebnisse
    try:
        df = pd.read_csv(results_file)
        print(f"✅ {len(df)} Wetten geladen aus: {results_file}\n")
    except FileNotFoundError:
        print(f"❌ Datei nicht gefunden: {results_file}")
        print("   Bitte stelle sicher, dass das Dutching System ausgeführt wurde.\n")
        return
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}\n")
        return
    
    # 1. ÜBERSICHTLICHE ZUSAMMENFASSUNG
    print("\n" + "=" * 80)
    print("  📊 SCHRITT 1: ZUSAMMENFASSUNG AUSGEBEN")
    print("=" * 80)
    
    OutputFormatter.print_summary(df, "Sportmonks Dutching Results")
    
    # 2. TOP VALUE BETS IM DETAIL
    print("\n" + "=" * 80)
    print("  ⭐ SCHRITT 2: TOP 10 VALUE BETS (DETAILLIERT)")
    print("=" * 80 + "\n")
    
    if 'expected_value' in df.columns:
        top_10 = df.nlargest(10, 'expected_value')
        
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            print(f"\n  #{i}")
            print(f"  {'-' * 75}")
            print(f"  {OutputFormatter.create_match_summary(row)}")
    
    # 3. FORMATIERTE TABELLE (NUR TOP 20)
    print("\n" + "=" * 80)
    print("  📋 SCHRITT 3: FORMATIERTE TABELLE (TOP 20 nach EV)")
    print("=" * 80 + "\n")
    
    formatted_top = OutputFormatter.format_results_dataframe(
        df, 
        sort_by='expected_value',
        ascending=False,
        top_n=20
    )
    
    # Wähle die wichtigsten Spalten für die Anzeige
    display_columns = ['📊', 'datetime', 'home_team', 'away_team', 
                      'market', 'odds', 'expected_value', 'stake']
    
    available_columns = [col for col in display_columns if col in formatted_top.columns]
    
    print(formatted_top[available_columns].to_string(index=False))
    
    # 4. REPORTS SPEICHERN
    print("\n" + "=" * 80)
    print("  💾 SCHRITT 4: REPORTS SPEICHERN")
    print("=" * 80 + "\n")
    
    # CSV Report
    csv_file = OutputFormatter.save_formatted_report(
        df, 
        filename='reports/formatted_dutching_report',
        format='csv'
    )
    
    # JSON Report
    json_file = OutputFormatter.save_formatted_report(
        df,
        filename='reports/formatted_dutching_report',
        format='json'
    )
    
    # Excel Report (optional, falls openpyxl installiert ist)
    try:
        excel_file = OutputFormatter.save_formatted_report(
            df,
            filename='reports/formatted_dutching_report',
            format='excel'
        )
    except ImportError:
        print("⚠️  Excel-Export nicht verfügbar (openpyxl nicht installiert)")
    
    # 5. FILTER-BEISPIELE
    print("\n" + "=" * 80)
    print("  🔍 SCHRITT 5: GEFILTERTE ANSICHTEN")
    print("=" * 80 + "\n")
    
    # Nur positive EV
    if 'expected_value' in df.columns:
        positive_ev = df[df['expected_value'] > 0]
        print(f"  ✅ Wetten mit positivem EV: {len(positive_ev)}")
        
        if len(positive_ev) > 0:
            print(f"  💰 Gesamteinsatz: {OutputFormatter.format_currency(positive_ev['stake'].sum())}")
            print(f"  📈 Durchschnitts-EV: {OutputFormatter.format_percentage(positive_ev['expected_value'].mean())}")
    
    # Nur hohe Odds (> 2.5)
    if 'odds' in df.columns:
        high_odds = df[df['odds'] > 2.5]
        print(f"\n  🎲 Wetten mit Odds > 2.5: {len(high_odds)}")
    
    # Nach Liga filtern
    if 'league' in df.columns:
        print(f"\n  🏆 Verfügbare Ligen:")
        for league in df['league'].unique():
            league_bets = df[df['league'] == league]
            print(f"     • {league}: {len(league_bets)} Wetten")
    
    print("\n" + "=" * 80)
    print("  ✅ FERTIG!")
    print("=" * 80 + "\n")


# =============================================================================
# INTEGRATION IN DEIN SPORTMONKS_DUTCHING_SYSTEM.PY
# =============================================================================

def add_to_sportmonks_system():
    """
    Zeigt, wie du den OutputFormatter in dein bestehendes
    sportmonks_dutching_system.py integrierst
    """
    
    integration_code = '''
# ============================================================================= 
# AM ENDE DEINER sportmonks_dutching_system.py DATEI HINZUFÜGEN:
# =============================================================================

from output_formatter import OutputFormatter

def main():
    """Hauptfunktion mit verbesserter Ausgabe"""
    
    # ... dein bestehender Code ...
    
    # Am Ende, nachdem results.csv gespeichert wurde:
    
    print("\\n" + "="*80)
    print("  📊 ERSTELLE ÜBERSICHTLICHE AUSGABE")
    print("="*80 + "\\n")
    
    # Lade die gerade gespeicherten Results
    results_df = pd.read_csv(config.OUTPUT_FILE)
    
    # 1. Drucke Zusammenfassung in die Konsole
    OutputFormatter.print_summary(results_df, "Dutching Results Summary")
    
    # 2. Erstelle formatierte Reports
    OutputFormatter.save_formatted_report(
        results_df,
        filename='reports/dutching_formatted',
        format='csv'
    )
    
    OutputFormatter.save_formatted_report(
        results_df,
        filename='reports/dutching_formatted',
        format='json'
    )
    
    # 3. Zeige Top 5 Wetten im Detail
    if 'expected_value' in results_df.columns:
        print("\\n⭐ TOP 5 VALUE BETS:\\n")
        top_5 = results_df.nlargest(5, 'expected_value')
        
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {OutputFormatter.create_match_summary(row)}")
        
    print("\\n" + "="*80 + "\\n")

if __name__ == "__main__":
    main()
'''
    
    print("\n" + "=" * 80)
    print("  📝 INTEGRATION IN SPORTMONKS SYSTEM")
    print("=" * 80 + "\n")
    print(integration_code)
    print("\n" + "=" * 80 + "\n")


# =============================================================================
# STREAMLIT DASHBOARD INTEGRATION
# =============================================================================

def streamlit_integration_example():
    """
    Zeigt, wie du den OutputFormatter in Streamlit verwendest
    """
    
    streamlit_code = '''
# ============================================================================= 
# IN DEINEM DASHBOARD.PY (für Tab "Live Matches & Dutching"):
# =============================================================================

import streamlit as st
from output_formatter import OutputFormatter

# ... in deinem Dashboard Code ...

with tab1:  # Dutching Tab
    st.markdown("### ⚽ Dutching System Results")
    
    # Lade Results
    dutching_file = Path("sportmonks_results.csv")
    
    if dutching_file.exists():
        df = pd.read_csv(dutching_file)
        
        # Erstelle Tabs für verschiedene Ansichten
        view_tab1, view_tab2, view_tab3 = st.tabs([
            "📊 Übersicht",
            "🏆 Top Bets",
            "📈 Statistiken"
        ])
        
        with view_tab1:
            # ÜBERSICHT MIT FILTERN
            st.markdown("#### Alle Wetten (gefiltert)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_ev = st.slider("Min EV %", -20, 50, 0)
            with col2:
                max_odds = st.slider("Max Odds", 1.5, 20.0, 20.0)
            with col3:
                selected_league = st.selectbox(
                    "Liga", 
                    ["Alle"] + list(df['league'].unique())
                )
            
            # Filter anwenden
            filtered = df.copy()
            if 'expected_value' in filtered.columns:
                filtered = filtered[filtered['expected_value'] >= min_ev]
            if 'odds' in filtered.columns:
                filtered = filtered[filtered['odds'] <= max_odds]
            if selected_league != "Alle":
                filtered = filtered[filtered['league'] == selected_league]
            
            # Formatierte Tabelle anzeigen
            formatted = OutputFormatter.format_results_dataframe(
                filtered,
                sort_by='expected_value',
                ascending=False
            )
            
            st.dataframe(formatted, use_container_width=True, height=500)
            
            # Download-Button
            csv = filtered.to_csv(index=False)
            st.download_button(
                "📥 CSV Download",
                csv,
                "dutching_filtered.csv",
                "text/csv"
            )
        
        with view_tab2:
            # TOP BETS MIT MATCH-KARTEN
            st.markdown("#### Top 10 Value Bets")
            
            if 'expected_value' in df.columns:
                top_10 = df.nlargest(10, 'expected_value')
                
                for idx, row in top_10.iterrows():
                    with st.expander(
                        f"{row['home_team']} vs {row['away_team']} - "
                        f"EV: {row['expected_value']:+.2f}%"
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Expected Value",
                                OutputFormatter.format_percentage(
                                    row['expected_value']
                                )
                            )
                        
                        with col2:
                            st.metric(
                                "Stake",
                                OutputFormatter.format_currency(
                                    row['stake']
                                )
                            )
                        
                        with col3:
                            st.metric(
                                "Potential Profit",
                                OutputFormatter.format_currency(
                                    row['potential_profit']
                                )
                            )
                        
                        st.markdown(f"**Details:** {OutputFormatter.create_match_summary(row)}")
        
        with view_tab3:
            # STATISTIKEN
            st.markdown("#### Statistik-Übersicht")
            
            stats = OutputFormatter.create_summary_stats(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Gesamt Wetten", stats['total_bets'])
            
            with col2:
                st.metric("Gesamteinsatz", stats.get('total_stake', '-'))
            
            with col3:
                st.metric("Durchschnitts-EV", stats.get('avg_ev', '-'))
            
            with col4:
                st.metric("Best EV", stats.get('best_ev', '-'))
            
            # Histogramme
            if 'expected_value' in df.columns:
                import plotly.express as px
                
                fig = px.histogram(
                    df,
                    x='expected_value',
                    nbins=30,
                    title="Expected Value Verteilung",
                    labels={'expected_value': 'Expected Value (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 Noch keine Dutching-Ergebnisse verfügbar")
'''
    
    print("\n" + "=" * 80)
    print("  🎨 STREAMLIT INTEGRATION")
    print("=" * 80 + "\n")
    print(streamlit_code)
    print("\n" + "=" * 80 + "\n")


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def main():
    """Hauptfunktion für Integrations-Beispiele"""
    
    print("\n" + "🎨" * 40)
    print("\n  OUTPUT FORMATTER - INTEGRATION GUIDE")
    print("\n" + "🎨" * 40 + "\n")
    
    while True:
        print("\nWas möchtest du sehen?\n")
        print("  1️⃣  Dutching-Ergebnisse formatiert anzeigen")
        print("  2️⃣  Integration in sportmonks_dutching_system.py")
        print("  3️⃣  Integration in Streamlit Dashboard")
        print("  4️⃣  Demo mit Beispieldaten")
        print("  0️⃣  Beenden\n")
        
        choice = input("Deine Wahl: ").strip()
        
        if choice == "1":
            results_file = input("\nPfad zur Results-CSV (Enter für default): ").strip()
            if not results_file:
                results_file = 'sportmonks_results.csv'
            enhance_dutching_output(results_file)
        
        elif choice == "2":
            add_to_sportmonks_system()
        
        elif choice == "3":
            streamlit_integration_example()
        
        elif choice == "4":
            from output_formatter import demo_formatter
            demo_formatter()
        
        elif choice == "0":
            print("\n👋 Auf Wiedersehen!\n")
            break
        
        else:
            print("\n❌ Ungültige Auswahl\n")


if __name__ == "__main__":
    main()