#!/usr/bin/env python3
"""
QUICK TEST: Sportmonks Scraper v2.0
====================================
Testet den v2.0 Scraper mit nur 10 Fixtures pro Saison
für schnelles Feedback.

Verwendung:
    python test_scraper_v2.py
"""

import os
from dotenv import load_dotenv
from sportmonks_xg_scraper_v2 import ScraperConfig, SportmonksXGScraperV2

def main():
    """Test-Hauptfunktion"""

    load_dotenv()
    api_token = os.getenv("SPORTMONKS_API_TOKEN")

    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        return

    print("\n" + "="*70)
    print("🧪 QUICK TEST - SCRAPER v2.0")
    print("="*70)
    print("\n⚙️ Test-Modus:")
    print("  • Max 10 Fixtures pro Saison")
    print("  • Debug-Output aktiviert")
    print("  • Test dauert ~2-3 Minuten\n")

    # Test-Konfiguration
    config = ScraperConfig(
        api_token=api_token,
        request_delay=1.0,  # Etwas schneller für Test
        output_file="test_game_database_sportmonks.csv",
        output_file_odds_only="test_game_database_sportmonks_odds_only.csv",
        output_file_xg_only="test_game_database_sportmonks_xg_only.csv",
        save_intermediate=False,  # Kein Cache für Test
        debug=True,
        max_fixtures_per_season=10  # NUR 10 Fixtures pro Saison!
    )

    scraper = SportmonksXGScraperV2(config)

    try:
        print("🚀 Starte Test-Scraping...\n")
        scraper.scrape_all()
        scraper.save_data()

        print("\n" + "="*70)
        print("✅ TEST ABGESCHLOSSEN")
        print("="*70)

        print("\n📁 Erstellte Test-Dateien:")
        for filename in [config.output_file, config.output_file_odds_only, config.output_file_xg_only]:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  • {filename} ({size} bytes)")

                # Zeige erste Zeile
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"    → {len(lines)-1} Spiele")
            else:
                print(f"  • {filename} (nicht erstellt)")

        print("\n💡 Nächste Schritte:")
        print("  1. Prüfe Test-Dateien")
        print("  2. Wenn OK: Führe vollen Scrape aus:")
        print("     python sportmonks_xg_scraper_v2.py")

    except KeyboardInterrupt:
        print("\n\n⚠️ Test abgebrochen")
    except Exception as e:
        print(f"\n\n❌ TEST FEHLER: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
