#!/usr/bin/env python3
"""
DEBUG-SKRIPT: SPORTMONKS xG DATA INSPECTOR
===========================================
Untersucht die Struktur der xG-Daten in Fixture-Antworten,
um herauszufinden, wo und wie die xG-Daten gespeichert sind.

Verwendung:
    python debug_xg_data.py
"""

import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Dict, Any

def search_dict_for_xg(data: Any, path: str = "root") -> list:
    """Rekursiv nach xG-relevanten Feldern suchen"""
    results = []

    xg_keywords = ['xg', 'expected', 'expectation', 'type_id']

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}"

            # Prüfe, ob Key xG-relevant ist
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in xg_keywords):
                results.append({
                    'path': current_path,
                    'key': key,
                    'type': type(value).__name__,
                    'value': value if not isinstance(value, (dict, list)) else f"{type(value).__name__} ({len(value)} items)"
                })

            # Rekursiv weitergehen
            if isinstance(value, (dict, list)):
                results.extend(search_dict_for_xg(value, current_path))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            current_path = f"{path}[{idx}]"
            if isinstance(item, (dict, list)):
                results.extend(search_dict_for_xg(item, current_path))

    return results

def test_xg_data():
    """Teste xG-Daten-Abruf"""

    load_dotenv()
    api_token = os.getenv("SPORTMONKS_API_TOKEN")

    if not api_token:
        print("❌ FEHLER: SPORTMONKS_API_TOKEN nicht in .env gefunden!")
        print("Bitte erstelle eine .env-Datei mit deinem API-Token:")
        print("  cp .env.example .env")
        print("  # Dann API-Token eintragen")
        return

    print("\n" + "="*70)
    print("🔍 SPORTMONKS xG DATA - STRUCTURE INSPECTOR")
    print("="*70)

    # Test-Fixture (ein abgeschlossenes Premier League Spiel)
    test_fixtures = [
        18535258,  # Beispiel-ID (anpassen!)
        18535259,  # Beispiel-ID (anpassen!)
    ]

    print(f"\n⚙️  API-Token: {api_token[:10]}...{api_token[-10:]}")
    print(f"⚙️  Test-Fixtures: {test_fixtures}")

    # Verschiedene Include-Kombinationen testen
    include_options = [
        # Aktuelle Version
        "fixtures.xGFixture;participants;scores;statistics;league",

        # Alternative Schreibweisen
        "xG;xGFixture;xg;expectedGoals;statistics",
        "xG;xg;expectedGoals",
        "statistics;statistics.xG",
        "xGFixture",
        "xg",
        "expectedGoals",

        # Kombinationen
        "statistics;xG;xg;expectedGoals",
        "all",  # Falls unterstützt
    ]

    all_results = []

    for fixture_id in test_fixtures:
        print(f"\n{'='*70}")
        print(f"🏆 TESTE FIXTURE ID: {fixture_id}")
        print(f"{'='*70}")

        # Hole Season-Daten (wie im aktuellen Scraper)
        season_endpoint = f"seasons/23614"  # Premier League 2024/25
        season_url = f"https://api.sportmonks.com/v3/football/{season_endpoint}"
        season_params = {
            'api_token': api_token,
            'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;fixtures.xGFixture;league'
        }

        print(f"\n📡 Teste Season-Endpunkt (aktueller Scraper-Ansatz)")
        print(f"   URL: {season_endpoint}")

        try:
            response = requests.get(season_url, params=season_params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                season_data = data.get('data', {})
                fixtures = season_data.get('fixtures', [])

                # Finde Test-Fixture
                test_fixture = next((f for f in fixtures if f.get('id') == fixture_id), None)

                if test_fixture:
                    print(f"   ✅ Fixture gefunden in Season-Antwort")

                    # Speichere Fixture
                    with open(f'fixture_from_season_{fixture_id}.json', 'w') as f:
                        json.dump(test_fixture, f, indent=2)
                    print(f"   📄 Gespeichert: fixture_from_season_{fixture_id}.json")

                    # Suche nach xG
                    xg_fields = search_dict_for_xg(test_fixture)
                    if xg_fields:
                        print(f"   🎯 {len(xg_fields)} xG-relevante Felder gefunden:")
                        for field in xg_fields[:5]:  # Zeige erste 5
                            print(f"      - {field['path']}: {field['value']}")
                    else:
                        print(f"   ❌ Keine xG-Felder gefunden")
                else:
                    print(f"   ⚠️  Fixture {fixture_id} nicht in Season gefunden")
            else:
                print(f"   ❌ Fehler: Status {response.status_code}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        # Jetzt teste direkte Fixture-Abfrage mit verschiedenen Includes
        for include_str in include_options:
            endpoint = f"fixtures/{fixture_id}"
            url = f"https://api.sportmonks.com/v3/football/{endpoint}"
            params = {
                'api_token': api_token,
                'include': include_str
            }

            print(f"\n📡 Include: {include_str}")

            try:
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    fixture_data = data.get('data', {})

                    # Suche nach xG-Feldern
                    xg_fields = search_dict_for_xg(fixture_data)

                    if xg_fields:
                        print(f"   ✅ {len(xg_fields)} xG-Felder gefunden")

                        # Speichere Sample
                        sample_file = f"xg_sample_{include_str.replace(';', '_')[:30]}_{fixture_id}.json"
                        with open(sample_file, 'w') as f:
                            json.dump(fixture_data, f, indent=2)
                        print(f"   📄 Sample: {sample_file}")

                        # Zeige Felder
                        for field in xg_fields[:3]:
                            print(f"      - {field['path']}")

                        all_results.append({
                            'fixture_id': fixture_id,
                            'include': include_str,
                            'status': 'SUCCESS',
                            'xg_fields_count': len(xg_fields),
                            'xg_fields': xg_fields
                        })
                    else:
                        print(f"   ⚠️  Keine xG-Felder gefunden")
                        all_results.append({
                            'fixture_id': fixture_id,
                            'include': include_str,
                            'status': 'NO_XG'
                        })
                else:
                    print(f"   ❌ Status: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Exception: {e}")

    # Zusammenfassung
    print(f"\n\n{'='*70}")
    print("📊 ZUSAMMENFASSUNG")
    print(f"{'='*70}")

    successful = [r for r in all_results if r['status'] == 'SUCCESS']

    if successful:
        print(f"\n✅ {len(successful)} erfolgreiche Include-Kombinationen gefunden\n")

        # Beste Option finden (meiste xG-Felder)
        best = max(successful, key=lambda r: r['xg_fields_count'])

        print("🎯 BESTE OPTION:")
        print(f"   Include: {best['include']}")
        print(f"   xG-Felder gefunden: {best['xg_fields_count']}")
        print("\n   Gefundene Felder:")
        for field in best['xg_fields'][:10]:
            print(f"      - {field['path']}: {field['value']}")

        # Analyse der Struktur
        print(f"\n{'='*70}")
        print("🔧 CODE-ANPASSUNG")
        print(f"{'='*70}\n")

        # Finde xG-Werte
        xg_values = [f for f in best['xg_fields'] if 'value' in f['path'].lower()]
        if xg_values:
            print("✅ xG-Werte gefunden in:")
            for xg_val in xg_values[:5]:
                print(f"   - {xg_val['path']}")

            print("\n💡 Empfohlene Code-Änderung in extract_xg_from_fixture():")
            print(f"\n   # Statt:")
            print(f"   xg_data_list = fixture.get('xgfixture')")
            print(f"\n   # Verwende:")
            first_xg = xg_values[0]['path'].replace('root.data.', '').split('.')[0]
            print(f"   xg_data_list = fixture.get('{first_xg}')")
    else:
        print("\n❌ Keine xG-Daten gefunden!")
        print("\n🔍 Mögliche Gründe:")
        print("  1. API-Plan unterstützt kein xG-Add-on")
        print("  2. Test-Fixture-IDs haben keine xG-Daten")
        print("  3. xG-Daten sind nur für aktuelle Spiele verfügbar")
        print("\n📞 Nächste Schritte:")
        print("  1. Sportmonks Support kontaktieren")
        print("  2. API-Plan prüfen (xG Add-on aktiviert?)")
        print("  3. Dokumentation: https://docs.sportmonks.com/")

    # Speichere Ergebnisse
    with open('xg_data_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(all_results),
            'successful': len(successful),
            'results': all_results
        }, f, indent=2)

    print(f"\n💾 Vollständige Ergebnisse gespeichert in: xg_data_test_results.json")

if __name__ == "__main__":
    test_xg_data()
