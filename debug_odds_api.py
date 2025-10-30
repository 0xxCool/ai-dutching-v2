#!/usr/bin/env python3
"""
DEBUG-SKRIPT: SPORTMONKS ODDS API TESTER
=========================================
Testet verschiedene Endpunkte, um herauszufinden,
welcher Endpunkt historische Quoten zurückgibt.

Verwendung:
    python debug_odds_api.py
"""

import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime

def test_odds_endpoints():
    """Teste verschiedene Odds-Endpunkte"""

    load_dotenv()
    api_token = os.getenv("SPORTMONKS_API_TOKEN")

    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        print("Bitte erstelle eine .env-Datei mit deinem API-Token:")
        print("  cp .env.example .env")
        print("  # Dann API-Token eintragen")
        return

    print("\n" + "="*70)
    print("🔍 SPORTMONKS ODDS API - ENDPOINT TESTER")
    print("="*70)

    # Test-Fixture (ein abgeschlossenes Premier League Spiel)
    # Du kannst diese ID durch eine aus deinem Scraper-Output ersetzen
    test_fixtures = [
        18535258,  # Beispiel-ID (anpassen!)
        18535259,  # Beispiel-ID (anpassen!)
    ]

    print(f"\n⚙️  API-Token: {api_token[:10]}...{api_token[-10:]}")
    print(f"⚙️  Test-Fixtures: {test_fixtures}")

    # Verschiedene Endpunkte testen
    endpoints = [
        ("Pre-Match (current)", "odds/pre-match/fixtures/{fixture_id}", {'include': 'market;bookmaker'}),
        ("Pre-Match (alt)", "odds/pre-match/fixtures/{fixture_id}", {'include': 'bookmaker;market'}),
        ("Fixture with Odds", "fixtures/{fixture_id}", {'include': 'odds;odds.bookmaker;odds.market'}),
        ("Fixture Odds Direct", "fixtures/{fixture_id}/odds", {'include': 'bookmaker;market'}),
        ("In-Play Odds", "odds/inplay/fixtures/{fixture_id}", {'include': 'market;bookmaker'}),
        ("All Odds", "odds/fixtures/{fixture_id}", {'include': 'market;bookmaker'}),
        ("Fixture with Markets", "fixtures/{fixture_id}", {'include': 'markets'}),
    ]

    results = []

    for fixture_id in test_fixtures:
        print(f"\n{'='*70}")
        print(f"🏆 TESTE FIXTURE ID: {fixture_id}")
        print(f"{'='*70}")

        for name, endpoint_template, params in endpoints:
            endpoint = endpoint_template.format(fixture_id=fixture_id)
            url = f"https://api.sportmonks.com/v3/football/{endpoint}"

            # API-Token hinzufügen
            params['api_token'] = api_token

            print(f"\n📡 {name}")
            print(f"   URL: {endpoint}")
            print(f"   Params: {params}")

            try:
                response = requests.get(url, params=params, timeout=10)

                print(f"   Status: {response.status_code}", end="")

                if response.status_code == 200:
                    data = response.json()

                    # Analysiere Antwort
                    has_data = 'data' in data and data['data']

                    if has_data:
                        # Prüfe, ob Odds vorhanden sind
                        data_content = data['data']

                        # Verschiedene mögliche Strukturen prüfen
                        has_odds = False
                        odds_location = None

                        if isinstance(data_content, list):
                            if len(data_content) > 0:
                                has_odds = True
                                odds_location = "list"
                        elif isinstance(data_content, dict):
                            # Suche nach Odds in verschiedenen Feldern
                            if 'odds' in data_content:
                                has_odds = True
                                odds_location = "data.odds"
                            elif 'markets' in data_content:
                                has_odds = True
                                odds_location = "data.markets"

                        if has_odds:
                            print(f" ✅ DATEN GEFUNDEN! (Location: {odds_location})")

                            # Speichere Sample
                            sample_file = f"odds_sample_{name.replace(' ', '_')}_{fixture_id}.json"
                            with open(sample_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            print(f"   📄 Sample gespeichert: {sample_file}")

                            results.append({
                                'fixture_id': fixture_id,
                                'name': name,
                                'endpoint': endpoint,
                                'status': 'SUCCESS',
                                'odds_location': odds_location
                            })
                        else:
                            print(f" ⚠️  Keine Odds gefunden (data vorhanden, aber leer)")
                            results.append({
                                'fixture_id': fixture_id,
                                'name': name,
                                'endpoint': endpoint,
                                'status': 'NO_ODDS'
                            })
                    else:
                        print(f" ❌ Leere Antwort")
                        results.append({
                            'fixture_id': fixture_id,
                            'name': name,
                            'endpoint': endpoint,
                            'status': 'EMPTY'
                        })

                elif response.status_code == 404:
                    print(f" ❌ Endpunkt nicht gefunden (404)")
                    results.append({
                        'fixture_id': fixture_id,
                        'name': name,
                        'endpoint': endpoint,
                        'status': 'NOT_FOUND'
                    })
                elif response.status_code == 429:
                    print(f" ⚠️  Rate Limit erreicht (429)")
                    results.append({
                        'fixture_id': fixture_id,
                        'name': name,
                        'endpoint': endpoint,
                        'status': 'RATE_LIMIT'
                    })
                else:
                    print(f" ❌ Fehler")
                    results.append({
                        'fixture_id': fixture_id,
                        'name': name,
                        'endpoint': endpoint,
                        'status': f'ERROR_{response.status_code}'
                    })

            except Exception as e:
                print(f" ❌ Exception: {e}")
                results.append({
                    'fixture_id': fixture_id,
                    'name': name,
                    'endpoint': endpoint,
                    'status': f'EXCEPTION: {e}'
                })

    # Zusammenfassung
    print(f"\n\n{'='*70}")
    print("📊 ZUSAMMENFASSUNG")
    print(f"{'='*70}")

    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']

    print(f"\n✅ Erfolgreiche Endpunkte: {len(successful)}/{len(results)}")
    if successful:
        print("\n🎯 FUNKTIONIERT:")
        for r in successful:
            print(f"  - {r['name']}: {r['endpoint']}")
            print(f"    Odds gefunden in: {r['odds_location']}")

    print(f"\n❌ Fehlgeschlagene Endpunkte: {len(failed)}/{len(results)}")
    if failed:
        print("\n❌ FEHLGESCHLAGEN:")
        for r in failed:
            print(f"  - {r['name']}: {r['endpoint']} ({r['status']})")

    # Empfehlung
    print(f"\n{'='*70}")
    print("💡 EMPFEHLUNG")
    print(f"{'='*70}\n")

    if successful:
        best = successful[0]
        print(f"✅ Verwende diesen Endpunkt in sportmonks_xg_scraper.py:")
        print(f"\n   endpoint = '{best['endpoint'].replace(str(best['fixture_id']), '{fixture_id}')}'")
        print(f"\n   Odds-Location: {best['odds_location']}")
        print(f"\n   Sample-Datei: odds_sample_{best['name'].replace(' ', '_')}_{best['fixture_id']}.json")
    else:
        print("❌ Kein funktionierender Endpunkt gefunden!")
        print("\n🔍 Mögliche Gründe:")
        print("  1. API-Plan unterstützt keine historischen Quoten")
        print("  2. Test-Fixture-IDs sind ungültig")
        print("  3. API-Token hat keine Berechtigung")
        print("\n📞 Nächste Schritte:")
        print("  1. Sportmonks Support kontaktieren")
        print("  2. API-Dokumentation prüfen: https://docs.sportmonks.com/")
        print("  3. Alternative Datenquellen evaluieren")

    # Speichere Zusammenfassung
    with open('odds_api_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2)

    print(f"\n💾 Vollständige Ergebnisse gespeichert in: odds_api_test_results.json")

if __name__ == "__main__":
    test_odds_endpoints()
