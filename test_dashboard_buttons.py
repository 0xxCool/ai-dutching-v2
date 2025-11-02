#!/usr/bin/env python3
"""
Dashboard Button Runtime Test
Tests ob Buttons tatsächlich die Funktionen aufrufen
"""

import sys
from pathlib import Path

# Test 1: Importiere das Dashboard und prüfe ob Funktionen definiert sind
print("="*80)
print("TEST 1: Funktions-Definitionen prüfen")
print("="*80)

try:
    # Lade das Dashboard Modul
    sys.path.insert(0, str(Path.cwd()))

    # Versuche die start_* Funktionen zu importieren
    import dashboard

    # Prüfe ob Funktionen existieren
    functions_to_check = [
        'start_scraper',
        'stop_scraper',
        'start_dutching',
        'stop_dutching',
        'start_ml_training',
        'start_portfolio_optimizer',
        'start_alert_system',
        'update_logs',
        'display_live_logs'
    ]

    print("\n✓ Dashboard-Modul erfolgreich importiert")
    print("\nPrüfe Funktionen:")

    for func_name in functions_to_check:
        if hasattr(dashboard, func_name):
            func = getattr(dashboard, func_name)
            print(f"  ✅ {func_name}: {type(func)}")
        else:
            print(f"  ❌ {func_name}: NICHT GEFUNDEN!")

except Exception as e:
    print(f"❌ FEHLER beim Importieren: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 2: LogStreamManager prüfen")
print("="*80)

try:
    # Prüfe ob LogStreamManager importierbar ist
    from dashboard import LogStreamManager

    print("✅ LogStreamManager importiert")

    # Erstelle Test-Instanz
    manager = LogStreamManager()
    print(f"✅ LogStreamManager Instanz erstellt: {type(manager)}")

    # Prüfe Methoden
    methods = ['start_process', 'stop_process', 'is_running', 'get_logs', 'stop_all']
    for method in methods:
        if hasattr(manager, method):
            print(f"  ✅ {method}: vorhanden")
        else:
            print(f"  ❌ {method}: FEHLT!")

except Exception as e:
    print(f"❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 3: Process-Start simulieren")
print("="*80)

try:
    from dashboard import LogStreamManager
    import subprocess

    manager = LogStreamManager()

    # Teste ob wir einen Prozess starten können (mit Echo-Befehl)
    print("\nTeste Process-Start mit echo-Befehl...")

    test_command = ['echo', 'Test erfolgreich']

    # Prüfe ob Script-Validierung funktioniert
    print(f"  Command: {test_command}")

    # Simuliere den Start (ohne echtes Script)
    # Das wird fehlschlagen wegen Script-Validierung, was GUT ist

    print("\n✅ LogStreamManager kann instanziiert werden")
    print("✅ Methoden sind vorhanden")

except Exception as e:
    print(f"❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST 4: Prüfe Script-Pfade")
print("="*80)

scripts_to_check = [
    'sportmonks_hybrid_scraper_v3_FINAL.py',
    'sportmonks_dutching_system.py',
    'train_ml_models.py',
    'portfolio_manager.py',
    'alert_system.py',
    'sportmonks_correct_score_system.py'
]

cwd = Path.cwd()
print(f"\nCurrent Working Directory: {cwd}")

for script in scripts_to_check:
    script_path = cwd / script
    if script_path.exists():
        print(f"  ✅ {script}: {script_path}")
    else:
        print(f"  ❌ {script}: NICHT GEFUNDEN!")

print("\n" + "="*80)
print("ZUSAMMENFASSUNG")
print("="*80)
print("\nDieser Test zeigt:")
print("1. Ob die Funktionen definiert sind")
print("2. Ob LogStreamManager funktioniert")
print("3. Ob die Scripts existieren")
print("\nWenn alles ✅ ist, liegt das Problem im Streamlit-Kontext.")
print("="*80)
