#!/usr/bin/env python3
"""
Test ob init_session_state() fehlerfrei läuft
"""

import sys
from pathlib import Path

print("="*80)
print("SESSION STATE INITIALIZATION TEST")
print("="*80)

# Mocke Streamlit für den Test
class MockSessionState:
    def __init__(self):
        self._state = {}

    def __contains__(self, key):
        return key in self._state

    def __getattr__(self, key):
        if key == '_state':
            return object.__getattribute__(self, '_state')
        return self._state.get(key)

    def __setattr__(self, key, value):
        if key == '_state':
            object.__setattr__(self, key, value)
        else:
            self._state[key] = value

    def get(self, key, default=None):
        return self._state.get(key, default)

# Mock Streamlit module
class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

    def error(self, msg):
        print(f"[ST ERROR] {msg}")

# Teste die Komponenten einzeln
print("\n1. Teste LogStreamManager Import:")
try:
    sys.path.insert(0, str(Path.cwd()))

    # Importiere die Komponenten die init_session_state() braucht
    from unified_config import get_config
    from sportmonks_dutching_system import Config as DutchingConfig, SportmonksClient
    from portfolio_manager import PortfolioManager
    from alert_system import AlertManager, AlertConfig
    from api_cache_system import FileCache as APICache, CacheConfig
    from continuous_training_system import ModelRegistry

    print("  ✅ Alle Imports erfolgreich")

except Exception as e:
    print(f"  ❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Teste Config Loading:")
try:
    config = get_config()
    print(f"  ✅ Config geladen: {type(config)}")
    print(f"  ✅ API Token: {config.api.api_token[:10]}..." if config.api.api_token else "  ⚠️  API Token leer")
except Exception as e:
    print(f"  ❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Teste Component Initialization:")
try:
    # DutchingConfig
    dutching_config = DutchingConfig()
    print(f"  ✅ DutchingConfig: {type(dutching_config)}")

    # SportmonksClient - DIES KÖNNTE FEHLSCHLAGEN wenn API Token fehlt
    try:
        api_token = config.api.api_token
        if not api_token:
            print("  ⚠️  API Token ist leer - SportmonksClient könnte fehlschlagen")
            api_token = "dummy_token_for_test"

        client = SportmonksClient(
            api_token=api_token,
            config=dutching_config
        )
        print(f"  ✅ SportmonksClient: {type(client)}")
    except Exception as e:
        print(f"  ❌ SportmonksClient FEHLER: {e}")
        import traceback
        traceback.print_exc()

    # PortfolioManager
    portfolio_mgr = PortfolioManager(bankroll=10000.0)
    print(f"  ✅ PortfolioManager: {type(portfolio_mgr)}")

    # AlertManager
    alert_config = AlertConfig()
    alert_mgr = AlertManager(alert_config)
    print(f"  ✅ AlertManager: {type(alert_mgr)}")

    # APICache
    cache_config = CacheConfig()
    api_cache = APICache(cache_config)
    print(f"  ✅ APICache: {type(api_cache)}")

    # ModelRegistry
    model_registry = ModelRegistry()
    print(f"  ✅ ModelRegistry: {type(model_registry)}")

except Exception as e:
    print(f"  ❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ERGEBNIS")
print("="*80)
print("\nWenn alle Komponenten ✅ sind, sollte init_session_state() funktionieren.")
print("Wenn es Fehler gibt, werden diese beim Dashboard-Start auftreten.")
print("="*80)
