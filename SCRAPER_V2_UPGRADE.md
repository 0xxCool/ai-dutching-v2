# 🚀 SPORTMONKS SCRAPER v2.0 - UPGRADE GUIDE

**Datum:** 2025-10-30
**Version:** 2.0 (basierend auf Debug-Ergebnissen)

---

## 📊 ÄNDERUNGEN GEGENÜBER v1.0

### ✅ HAUPTVERBESSERUNGEN

#### 1. **Optimierte API-Nutzung** (⚡ 380+ API-Calls gespart!)

**v1.0:**
```python
# Fixtures holen
fixtures = get_fixtures_for_season(season_id)  # 1 API-Call

# Dann für JEDES Fixture einzeln Quoten holen
for fixture in fixtures:  # 380 API-Calls!
    odds = get_odds_for_fixture(fixture['id'])
```

**v2.0:**
```python
# Fixtures UND Quoten in EINEM Call holen!
fixtures = get_fixtures_for_season(season_id)  # 1 API-Call (mit include=odds)

# Quoten direkt aus Fixture extrahieren
for fixture in fixtures:  # 0 zusätzliche API-Calls!
    odds = extract_odds_from_fixture(fixture)
```

**Ergebnis:** **~95% weniger API-Calls** für Quoten!

#### 2. **Separate Datenqualitäts-Kategorien**

**v1.0:** Speichert nur Spiele mit Quoten UND xG
```python
if has_odds and has_xg:
    save(game)  # Alles andere wird verworfen!
```

**v2.0:** Speichert ALLE Daten, kategorisiert nach Qualität
```python
if has_odds and has_xg:
    save_complete(game)           # → game_database_sportmonks.csv
elif has_odds:
    save_odds_only(game)          # → game_database_sportmonks_odds_only.csv
elif has_xg:
    save_xg_only(game)            # → game_database_sportmonks_xg_only.csv
```

**Ergebnis:** Keine Daten gehen verloren!

#### 3. **Detailliertes Debugging**

**v1.0:** Zeigt nur Fortschrittsbalken
```
Saison 2024/2025 Quoten: 100%|███| 380/380 [15:48<00:00]
✅ 0 Spiele gespeichert
```

**v2.0:** Zeigt genau, was passiert
```
Saison 2024/2025
    Verarbeite 380 Spiele...
    ✅ Ergebnis:
       - Komplett (Quoten + xG): 125
       - Nur Quoten: 180
       - Nur xG: 45
       - Übersprungen (Datum): 10
       - Übersprungen (Status): 15
       - Übersprungen (keine Daten): 5
```

**Ergebnis:** Du siehst EXAKT, warum Spiele übersprungen werden!

#### 4. **Robustere Fehlerbehandlung**

**v1.0:** Bei Fehler stoppt der Scraper
```python
odds = get_odds_for_fixture(fixture_id)  # Fehler → Crash
```

**v2.0:** Fehler werden protokolliert, aber Scraping läuft weiter
```python
try:
    odds = extract_odds_from_fixture(fixture)
except Exception as e:
    if debug:
        print(f"WARNUNG: {e}")
    # Weiter mit nächstem Fixture
```

#### 5. **Umfassende Statistiken**

**v2.0 Endstatistik:**
```
📊 FINALE STATISTIKEN
========================================

🌐 API-Calls: 142

📈 Fixtures:
  • Gesamt abgerufen: 2,340
  • Mit Quoten + xG: 845
  • Mit Quoten: 1,234
  • Mit xG: 923

⏭️ Übersprungen:
  • Datum zu alt: 67
  • Status nicht FT: 189
  • Keine Daten: 45

🏆 Verteilung nach Ligen (Komplett):
  Premier League       125
  La Liga              98
  Bundesliga           87
  ...

📅 Zeitraum: 2024-03-01 bis 2025-10-30
```

---

## 🔧 TECHNISCHE ÄNDERUNGEN

### Code-Änderungen im Detail

#### Änderung 1: Include-Parameter erweitert

```python
# v1.0 (sportmonks_xg_scraper.py:164)
params = {
    'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;league;fixtures.xGFixture'
}

# v2.0
params = {
    'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;fixtures.xGFixture;fixtures.odds;league'
    #                                                                       ^^^^^^^^^^^^^^^^ NEU!
}
```

#### Änderung 2: Neue Methode extract_odds_from_fixture()

```python
# v1.0: Separater API-Call
def get_odds_for_fixture(self, fixture_id: int) -> Dict:
    endpoint = f'odds/pre-match/fixtures/{fixture_id}'
    data = self._make_request(endpoint, params)
    return self._parse_sportmonks_odds(data['data'])

# v2.0: Extrahiere direkt aus Fixture
def extract_odds_from_fixture(self, fixture: Dict) -> Dict:
    odds_dict = {'odds_home': None, 'odds_draw': None, 'odds_away': None}

    odds_list = fixture.get('odds', [])
    for odds_item in odds_list:
        market = odds_item.get('market')
        if market and market.get('name') == '3Way Result':
            # ... extrahiere Quoten ...

    return odds_dict
```

#### Änderung 3: Separate Speicherung

```python
# v2.0: Drei separate Listen
self.complete_data = []      # Quoten + xG
self.odds_only_data = []     # Nur Quoten
self.xg_only_data = []       # Nur xG

# v2.0: Kategorisierung
has_odds = combined_data.get('odds_home') is not None
has_xg = (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)

if has_odds and has_xg:
    self.complete_data.append(combined_data)
elif has_odds:
    self.odds_only_data.append(combined_data)
elif has_xg:
    self.xg_only_data.append(combined_data)
```

#### Änderung 4: Statistik-Tracking

```python
# v2.0: Detaillierte Statistiken
self.stats = {
    'fixtures_fetched': 0,
    'fixtures_with_odds': 0,
    'fixtures_with_xg': 0,
    'fixtures_complete': 0,
    'fixtures_skipped_date': 0,
    'fixtures_skipped_status': 0,
    'fixtures_skipped_no_data': 0,
}
```

---

## 📥 VERWENDUNG

### Installation

Keine zusätzlichen Dependencies erforderlich! Verwendet dieselben Pakete wie v1.0.

### Schnellstart

```bash
# 1. .env erstellen (falls noch nicht vorhanden)
cp .env.example .env
# API-Token eintragen

# 2. v2.0 ausführen
python sportmonks_xg_scraper_v2.py
```

### Konfiguration

```python
# In main():
config = ScraperConfig(
    api_token=api_token,
    request_delay=1.3,
    debug=True,                    # Zeigt detailliertes Debug-Output
    max_fixtures_per_season=None   # None = alle, oder z.B. 10 für Testing
)
```

**Für Testing:**
```python
max_fixtures_per_season=10  # Nur erste 10 Fixtures pro Saison
```

**Für Production:**
```python
max_fixtures_per_season=None  # Alle Fixtures
debug=False                   # Weniger Output
```

---

## 📁 OUTPUT-DATEIEN

### v1.0
- `game_database_sportmonks.csv` (oft leer wegen Filter)
- `temp_game_database_sportmonks.csv` (Cache)

### v2.0
- **`game_database_sportmonks.csv`** - Komplett (Quoten + xG) ⭐
- **`game_database_sportmonks_odds_only.csv`** - Nur Quoten
- **`game_database_sportmonks_xg_only.csv`** - Nur xG
- `temp_game_database_sportmonks.csv` - Cache

**Empfehlung für ML-Training:**
- **Hauptdatenbank:** `game_database_sportmonks.csv` (vollständige Features)
- **Fallback:** `game_database_sportmonks_odds_only.csv` (ohne xG-Features trainieren)

---

## 🔄 MIGRATION VON v1.0 ZU v2.0

### Option A: Direkter Ersatz (Empfohlen)

```bash
# 1. Backup erstellen
mv sportmonks_xg_scraper.py sportmonks_xg_scraper_v1_backup.py

# 2. v2.0 als Standard setzen
mv sportmonks_xg_scraper_v2.py sportmonks_xg_scraper.py

# 3. Cache löschen (für sauberen Neustart)
rm temp_game_database_sportmonks.csv

# 4. Scraper ausführen
python sportmonks_xg_scraper.py
```

### Option B: Parallelbetrieb

```bash
# Behalte v1.0 und verwende v2.0 parallel
python sportmonks_xg_scraper_v2.py  # v2.0 ausführen
```

### Option C: Schrittweise Migration

```bash
# 1. Teste v2.0 mit wenigen Fixtures
# Editiere sportmonks_xg_scraper_v2.py:
#   max_fixtures_per_season=10

python sportmonks_xg_scraper_v2.py

# 2. Prüfe Output
ls -lh game_database_sportmonks*.csv

# 3. Wenn OK, voller Scrape
# Editiere sportmonks_xg_scraper_v2.py:
#   max_fixtures_per_season=None

python sportmonks_xg_scraper_v2.py
```

---

## 🐛 TROUBLESHOOTING

### Problem: Immer noch 0 Spiele mit "Komplett"

**Mögliche Ursache:** Historische Spiele haben keine Quoten in der API

**Lösung:**
1. Prüfe die anderen Dateien:
   ```bash
   wc -l game_database_sportmonks_odds_only.csv
   wc -l game_database_sportmonks_xg_only.csv
   ```

2. Wenn `odds_only` oder `xg_only` Daten haben:
   - **Quoten fehlen:** Historische Pre-Match Odds nicht verfügbar
   - **xG fehlt:** xG-Add-on nicht aktiviert oder nur für aktuelle Spiele

3. Kontaktiere Sportmonks Support:
   ```
   "I'm using your API to fetch historical match data.
    I can see odds/xG for future fixtures, but not for finished matches.
    Does my plan support historical pre-match odds and xG data?"
   ```

### Problem: Zu wenig Daten

**Debug:**
```python
# Setze in sportmonks_xg_scraper_v2.py:
max_fixtures_per_season=10  # Teste mit wenigen
debug=True                  # Zeige Details
```

Dann prüfe Output:
- Wie viele Fixtures haben `status='FT'`?
- Wie viele davon haben Quoten?
- Wie viele davon haben xG?

### Problem: API-Rate-Limit

**Symptom:**
```
⚠️ Rate Limit - warte 2s...
```

**Lösung:**
```python
# Erhöhe Delay
request_delay=2.0  # Statt 1.3
```

---

## 📊 ERWARTETE ERGEBNISSE

### Realistische Erwartungen

**Für abgeschlossene Spiele (status='FT'):**

| Datentyp | Verfügbarkeit | Warum? |
|----------|---------------|--------|
| **Basis-Daten** (Teams, Score) | ✅ 100% | Immer verfügbar |
| **xG-Daten** | ✅ ~80-95% | Mit xG-Add-on |
| **Pre-Match Odds** | ⚠️ 0-30% | Oft nur für aktive/zukünftige Spiele |

**Für zukünftige Spiele (status='NS'):**

| Datentyp | Verfügbarkeit | Warum? |
|----------|---------------|--------|
| **Basis-Daten** | ✅ 100% | Immer verfügbar |
| **Pre-Match Odds** | ✅ ~90-100% | Aktiv verfügbar |
| **xG-Daten** | ❌ 0% | Noch nicht gespielt |

**Fazit:** Wenn du **historische Daten für ML-Training** brauchst:
- **xG:** Sollte verfügbar sein (mit Add-on)
- **Quoten:** Möglicherweise NICHT verfügbar für alte Spiele

**Alternative:** Verwende zusätzliche Datenquelle für historische Quoten:
- **Football-Data.co.uk** (kostenlos, CSV-Download)
- **Odds API** (kostenpflichtig, aber historische Daten)
- **Betfair API** (historische Exchange-Quoten)

---

## 🎯 NEXT STEPS

### Für sofortigen Einsatz:

1. **Führe v2.0 aus:**
   ```bash
   python sportmonks_xg_scraper_v2.py
   ```

2. **Analysiere Output:**
   ```bash
   # Zeige Anzahl Spiele in jeder Datei
   wc -l game_database_sportmonks*.csv

   # Zeige erste 5 Zeilen (komplett)
   head -5 game_database_sportmonks.csv

   # Zeige Statistik
   # (wird automatisch am Ende des Scrapes angezeigt)
   ```

3. **Verwende für ML-Training:**
   ```python
   import pandas as pd

   # Lade vollständige Daten
   df_complete = pd.read_csv('game_database_sportmonks.csv')
   print(f"Vollständige Daten: {len(df_complete)} Spiele")

   # Falls zu wenig: Kombiniere mit Odds-Only
   df_odds = pd.read_csv('game_database_sportmonks_odds_only.csv')
   print(f"Mit Quoten (ohne xG): {len(df_odds)} Spiele")

   # Kombiniere (setze xG=0 für Odds-Only Spiele)
   df_combined = pd.concat([df_complete, df_odds], ignore_index=True)
   df_combined['home_xg'] = df_combined['home_xg'].fillna(0)
   df_combined['away_xg'] = df_combined['away_xg'].fillna(0)

   print(f"Gesamt für Training: {len(df_combined)} Spiele")
   ```

### Für langfristige Verbesserung:

1. **Historische Quoten aus alternativen Quellen:**
   - Implementiere Football-Data.co.uk Scraper
   - Merge mit Sportmonks xG-Daten

2. **Automatisierung:**
   - Cronjob für tägliches Scraping
   - Inkrementelles Update (nur neue Spiele)

3. **Monitoring:**
   - Alert bei 0 neuen Spielen
   - API-Call-Tracking

---

## 📞 SUPPORT

**Bei Fragen:**
- **Technische Probleme:** Siehe `DEBUG_ANLEITUNG.md`
- **API-Probleme:** Sportmonks Support (support@sportmonks.com)
- **Code-Fragen:** Siehe Kommentare in `sportmonks_xg_scraper_v2.py`

---

**Viel Erfolg mit v2.0! 🚀**

**Changelog:**
- **v2.0** (2025-10-30): Initiale v2.0 basierend auf Debug-Ergebnissen
- **v1.0** (ursprünglich): Basis-Version
