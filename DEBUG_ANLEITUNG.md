# 🔧 DEBUG-ANLEITUNG: Sportmonks Scraper Problem

## 📋 Schnellstart

### Schritt 1: .env-Datei erstellen

```bash
cp .env.example .env
```

Öffne `.env` und trage deinen Sportmonks API-Token ein:

```bash
SPORTMONKS_API_TOKEN=dein_api_token_hier
```

### Schritt 2: Debug-Skripte ausführbar machen

```bash
chmod +x debug_odds_api.py
chmod +x debug_xg_data.py
```

### Schritt 3: Odds-API testen

```bash
python debug_odds_api.py
```

**Was macht das Skript?**
- Testet 7 verschiedene API-Endpunkte für Quoten
- Zeigt, welche Endpunkte Daten zurückgeben
- Speichert funktionierende Antworten als JSON-Samples

**Erwartete Ausgabe:**
```
🔍 SPORTMONKS ODDS API - ENDPOINT TESTER
========================================

🏆 TESTE FIXTURE ID: 18535258
========================================

📡 Pre-Match (current)
   URL: odds/pre-match/fixtures/18535258
   Status: 200 ✅ DATEN GEFUNDEN! (Location: list)
   📄 Sample gespeichert: odds_sample_Pre-Match_(current)_18535258.json

...

📊 ZUSAMMENFASSUNG
========================================

✅ Erfolgreiche Endpunkte: 2/14

🎯 FUNKTIONIERT:
  - Pre-Match (current): odds/pre-match/fixtures/18535258
    Odds gefunden in: list

💡 EMPFEHLUNG
========================================

✅ Verwende diesen Endpunkt in sportmonks_xg_scraper.py:

   endpoint = 'odds/pre-match/fixtures/{fixture_id}'

   Odds-Location: list

   Sample-Datei: odds_sample_Pre-Match_(current)_18535258.json
```

### Schritt 4: xG-Daten testen

```bash
python debug_xg_data.py
```

**Was macht das Skript?**
- Testet verschiedene Include-Parameter für xG-Daten
- Durchsucht die API-Antwort nach xG-Feldern
- Zeigt die korrekte Struktur und Pfade

**Erwartete Ausgabe:**
```
🔍 SPORTMONKS xG DATA - STRUCTURE INSPECTOR
========================================

🏆 TESTE FIXTURE ID: 18535258
========================================

📡 Include: xGFixture
   ✅ 4 xG-Felder gefunden
   📄 Sample: xg_sample_xGFixture_18535258.json
      - root.data.xgfixture[0].data.value
      - root.data.xgfixture[0].location
      - root.data.xgfixture[1].data.value
      - root.data.xgfixture[1].location

...

📊 ZUSAMMENFASSUNG
========================================

✅ 3 erfolgreiche Include-Kombinationen gefunden

🎯 BESTE OPTION:
   Include: xGFixture
   xG-Felder gefunden: 4

   Gefundene Felder:
      - root.data.xgfixture[0].type_id: 5304
      - root.data.xgfixture[0].location: home
      - root.data.xgfixture[0].data.value: 1.85
      - root.data.xgfixture[1].location: away

🔧 CODE-ANPASSUNG
========================================

✅ xG-Werte gefunden in:
   - root.data.xgfixture[0].data.value

💡 Empfohlene Code-Änderung in extract_xg_from_fixture():

   # Statt:
   xg_data_list = fixture.get('xgfixture')

   # Verwende:
   xg_data_list = fixture.get('xgfixture')
```

---

## 📊 Ergebnis-Interpretation

### Szenario A: Beide Skripte finden Daten ✅

**Bedeutung:** Der Code ist korrekt, aber möglicherweise:
- Verwendest du Test-Fixture-IDs, die keine echten Spiele sind
- Die Filter-Logik ist zu streng

**Lösung:**
1. Verwende Fixture-IDs aus deinem Scraper-Output
2. Lockere die Filter-Logik (siehe Tiefenanalyse)

### Szenario B: Odds-Skript findet KEINE Daten ❌

**Bedeutung:** Die Sportmonks API gibt keine historischen Quoten zurück

**Mögliche Gründe:**
1. **Falscher API-Plan:** Historische Quoten nur in höheren Plänen
2. **API-Limitation:** Pre-Match Odds werden nach Spielbeginn gelöscht
3. **Falscher Endpunkt:** Anderer Endpunkt erforderlich

**Lösung:**
1. Kontaktiere Sportmonks Support:
   ```
   "Does my API plan support historical pre-match odds for finished fixtures?
    If yes, which endpoint should I use?"
   ```

2. Alternative Datenquellen:
   - **Odds API**: https://the-odds-api.com/ (historische Quoten)
   - **Football-Data.co.uk**: https://www.football-data.co.uk/ (kostenlos!)
   - **Betfair API**: https://docs.developer.betfair.com/

### Szenario C: xG-Skript findet KEINE Daten ❌

**Bedeutung:** xG-Daten sind nicht verfügbar oder im falschen Feld

**Mögliche Gründe:**
1. **xG Add-on nicht aktiviert:** Sportmonks xG-Daten sind ein kostenpflichtiges Add-on
2. **Falscher Include-Parameter:** Andere Schreibweise erforderlich
3. **Nur für aktuelle Spiele:** Historische xG-Daten nicht verfügbar

**Lösung:**
1. Prüfe deinen API-Plan:
   - Ist das "Expected Goals (xG)" Add-on aktiviert?
   - Unterstützt es historische Daten?

2. Kontaktiere Sportmonks Support:
   ```
   "I have the xG add-on, but I'm not getting xG data for historical fixtures.
    Which include parameter should I use, and does my plan support historical xG?"
   ```

### Szenario D: Beides findet Daten, aber Scraper speichert 0 Spiele ❌

**Bedeutung:** Filter-Logik oder Code-Problem

**Lösung:**
1. Füge Debug-Output zum Scraper hinzu (siehe Tiefenanalyse, Abschnitt "Test 3")
2. Prüfe, welcher Filter genau fehlschlägt
3. Lockere die Filter-Logik

---

## 🔧 Code-Anpassungen

### Wenn Odds-Endpunkt anders ist

**Beispiel:** Odds sind direkt im Fixture enthalten

```python
# In sportmonks_xg_scraper.py, Zeile 164

# STATT:
params = {
    'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;league;fixtures.xGFixture'
}

# VERWENDE:
params = {
    'include': 'fixtures.participants;fixtures.scores;fixtures.statistics;league;fixtures.xGFixture;fixtures.odds'
}
```

Dann in `get_odds_for_fixture()` (Zeile 195):

```python
# STATT:
def get_odds_for_fixture(self, fixture_id: int) -> Dict:
    endpoint = f'odds/pre-match/fixtures/{fixture_id}'
    # ...

# VERWENDE:
def get_odds_for_fixture(self, fixture_id: int) -> Dict:
    # Odds sind bereits im Fixture enthalten (via include)
    # Diese Methode wird dann gar nicht mehr gebraucht
    return {}  # Placeholder
```

Und in `scrape_league()` (Zeile 437):

```python
# STATT:
odds_data = self.client.get_odds_for_fixture(fixture['id'])

# VERWENDE:
odds_data = self._extract_odds_from_fixture(fixture)  # Neue Methode

# Neue Methode einfügen (z.B. nach extract_xg_from_fixture):
def _extract_odds_from_fixture(self, fixture: Dict) -> Dict:
    """Extrahiere Quoten aus dem Fixture (wenn via include geladen)"""
    odds_dict = {
        'odds_home': None,
        'odds_draw': None,
        'odds_away': None
    }

    odds_list = fixture.get('odds', [])
    # ... (ähnliche Logik wie _parse_sportmonks_odds)

    return odds_dict
```

### Wenn xG-Feld anders heißt

**Beispiel:** xG ist in `statistics.expected_goals`

```python
# In extract_xg_from_fixture(), Zeile 313

# STATT:
xg_data_list = fixture.get('xgfixture')

# VERWENDE:
# Option 1: Direkter Zugriff
statistics = fixture.get('statistics', {})
expected_goals = statistics.get('expected_goals', {})
result['home_xg'] = float(expected_goals.get('home', 0))
result['away_xg'] = float(expected_goals.get('away', 0))

# Option 2: Falls xG in anderem Feld
xg_data_list = fixture.get('xG') or fixture.get('xg') or fixture.get('expected_goals')
```

### Filter-Logik lockern

**Option 1:** Separate Speicherung für Spiele mit/ohne vollständige Daten

```python
# In scrape_league(), Zeile 442

# STATT:
if (combined_data.get('odds_home') and
    (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)):
    league_data.append(combined_data)

# VERWENDE:
has_odds = combined_data.get('odds_home') is not None
has_xg = (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)

if has_odds and has_xg:
    # Vollständige Daten
    league_data.append(combined_data)
    season_added_games_count += 1
elif has_odds:
    # Nur Quoten (speichere trotzdem, mit xG=0)
    league_data.append(combined_data)
    season_added_games_count += 1
    print(f"      (Spiel {fixture['id']}: Keine xG-Daten, nur Quoten)")
elif has_xg:
    # Nur xG (speichere trotzdem, ohne Quoten)
    league_data.append(combined_data)
    season_added_games_count += 1
    print(f"      (Spiel {fixture['id']}: Keine Quoten, nur xG)")
```

**Option 2:** Warnung ausgeben statt Überspringen

```python
# In scrape_league(), Zeile 442

has_odds = combined_data.get('odds_home') is not None
has_xg = (combined_data.get('home_xg', 0) > 0 or combined_data.get('away_xg', 0) > 0)

if not has_odds:
    print(f"⚠️  Fixture {fixture['id']}: Keine Quoten gefunden")
if not has_xg:
    print(f"⚠️  Fixture {fixture['id']}: Keine xG-Daten gefunden")

# Speichere trotzdem (wenn mindestens eins vorhanden)
if has_odds or has_xg:
    league_data.append(combined_data)
    season_added_games_count += 1
```

---

## 📞 Support-Kontakt

### Sportmonks Support

**Email:** support@sportmonks.com

**Fragen-Template:**

```
Subject: Historical Odds and xG Data Access - API Plan Question

Hello Sportmonks Support,

I'm using the Sportmonks API to build a football betting analysis system.
I have the following API plan: [YOUR PLAN NAME]

I'm trying to fetch:
1. Historical pre-match odds (3-Way Result) for finished fixtures
2. Expected Goals (xG) data for the same fixtures

Questions:
1. Does my plan support historical pre-match odds?
   - If yes, which endpoint should I use?
   - If no, which plan do I need to upgrade to?

2. Does my plan include the xG add-on?
   - If yes, which include parameter should I use to get xG data?
   - If no, how much does the xG add-on cost?

3. Are historical odds and xG data available indefinitely, or only for a
   certain period after the match?

Example fixtures I'm trying to fetch:
- Fixture ID: 18535258
- Date: [DATE OF FIXTURE]
- League: Premier League

Thank you for your help!

Best regards,
[YOUR NAME]
```

---

## ✅ Checkliste

Nach dem Ausführen der Debug-Skripte:

- [ ] `.env`-Datei erstellt und API-Token eingetragen
- [ ] `debug_odds_api.py` ausgeführt
  - [ ] Ergebnisse analysiert
  - [ ] JSON-Samples überprüft
  - [ ] Funktionierenden Endpunkt identifiziert (oder nicht)
- [ ] `debug_xg_data.py` ausgeführt
  - [ ] Ergebnisse analysiert
  - [ ] xG-Struktur überprüft
  - [ ] Korrektes xG-Feld identifiziert (oder nicht)
- [ ] Code-Anpassungen vorgenommen (falls nötig)
- [ ] Scraper erneut getestet
- [ ] Bei Problemen: Sportmonks Support kontaktiert

---

## 📚 Weiterführende Links

- **Tiefenanalyse:** `REPOSITORY_TIEFENANALYSE_SPORTMONKS_SCRAPER.md`
- **Sportmonks Dokumentation:** https://docs.sportmonks.com/football/
- **Sportmonks Support:** https://support.sportmonks.com/
- **API Playground:** https://www.sportmonks.com/sports/football-api/playground

---

**Erstellt:** 2025-10-30
**Version:** 1.0
**Autor:** Claude (Anthropic AI)
