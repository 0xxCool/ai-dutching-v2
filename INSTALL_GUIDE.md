# 🚀 AI Dutching v2 - Installations-Anleitung

## Problem: Dashboard startet, aber Buttons funktionieren nicht?

### Symptome:
- ✅ Dashboard lädt und ist sichtbar
- ❌ Buttons reagieren nicht beim Klicken
- ❌ Prozesse bleiben auf "STOPPED" oder "OFFLINE"
- ❌ Keine Logs erscheinen

### Ursache:
**Fehlende Dependencies!** Das Dashboard benötigt alle Python-Packages aus `requirements.txt`.

---

## ✅ SCHNELLE LÖSUNG

### Schritt 1: Dependencies installieren

```bash
# Minimal (nur Dashboard + Core-Funktionen):
pip install streamlit pandas numpy plotly python-dotenv requests

# Komplett (alle Features):
pip install -r requirements.txt
```

### Schritt 2: .env Datei erstellen

```bash
# Kopiere .env.example zu .env
cp .env.example .env

# Editiere .env und füge deinen API-Token ein:
SPORTMONKS_API_TOKEN=dein_token_hier
```

### Schritt 3: Dashboard starten

```bash
streamlit run dashboard.py
```

### Schritt 4: Status prüfen

Im Dashboard, Tab 2 "System Control":
- ✅ "System bereit - LogStreamManager initialisiert" = ALLES GUT
- ❌ "KRITISCH: LogStreamManager nicht initialisiert" = Dependencies fehlen

---

## 🔍 DEBUG-MODUS

Das Dashboard zeigt jetzt in der **Sidebar** detaillierte Debug-Informationen:

### Was bedeuten die Meldungen?

| Meldung | Bedeutung | Lösung |
|---------|-----------|--------|
| ✅ LogStreamManager OK | System funktioniert | Keine Aktion nötig |
| ✅ Config loaded | Konfiguration OK | Keine Aktion nötig |
| ⚠️ API Token nicht gesetzt | .env fehlt | .env Datei erstellen |
| ❌ LogStreamManager Fehler | Dependencies fehlen | `pip install` ausführen |
| ❌ Component Fehler | Package fehlt | Spezifisches Package installieren |

---

## 📦 DEPENDENCIES PRÜFEN

### Test 1: Python Version

```bash
python --version
# Sollte sein: Python 3.10 oder 3.11
```

### Test 2: Streamlit

```bash
python -c "import streamlit; print(streamlit.__version__)"
# Sollte sein: 1.40.0 oder höher
```

### Test 3: Alle Core-Dependencies

```bash
python -c "
import streamlit
import pandas
import numpy
import plotly
import requests
from dotenv import load_dotenv
print('✅ Alle Core-Dependencies OK')
"
```

Wenn dieser Test FEHLER zeigt, installiere:
```bash
pip install streamlit pandas numpy plotly requests python-dotenv
```

---

## 🐛 HÄUFIGE PROBLEME

### Problem 1: "ModuleNotFoundError: No module named 'streamlit'"

**Lösung:**
```bash
pip install streamlit
```

### Problem 2: "ModuleNotFoundError: No module named 'dotenv'"

**Lösung:**
```bash
pip install python-dotenv
```

### Problem 3: Buttons funktionieren nicht

**Diagnose:**
1. Öffne Dashboard
2. Gehe zu Tab 2 "System Control"
3. Schaue nach Fehlermeldung oben

**Wenn "LogStreamManager nicht initialisiert":**
```bash
# Installiere alle Dependencies
pip install -r requirements.txt

# Starte Dashboard neu
streamlit run dashboard.py
```

### Problem 4: "Config loaded" aber trotzdem Fehler

**Lösung:**
Prüfe welche Komponente fehlt in der Sidebar:
- ❌ SportmonksClient → API Token Problem
- ❌ PortfolioManager → Package fehlt
- ❌ AlertManager → Package fehlt

Installiere fehlende Packages:
```bash
pip install -r requirements.txt
```

---

## 🎯 MINIMAL-INSTALLATION (Nur Dashboard)

Wenn du NUR das Dashboard ohne ML/GPU Features brauchst:

```bash
# Minimal-Requirements installieren
pip install streamlit>=1.40.0 \
            pandas>=2.0.0 \
            numpy>=1.24.0 \
            plotly>=5.14.0 \
            requests>=2.31.0 \
            python-dotenv>=1.0.0 \
            pyyaml>=6.0 \
            tqdm>=4.65.0

# Dashboard starten
streamlit run dashboard.py
```

**Einschränkungen der Minimal-Installation:**
- ❌ Kein ML-Training
- ❌ Keine GPU-Features
- ❌ Kein Backtesting
- ✅ Dashboard funktioniert
- ✅ Prozess-Management funktioniert
- ✅ Log-Anzeige funktioniert

---

## 🚀 VOLL-INSTALLATION (Alle Features)

Für ALLE Features inkl. ML, GPU, Backtesting:

```bash
# Option 1: Mit requirements.txt
pip install -r requirements.txt

# Option 2: Manuell (Windows)
# 1. CUDA Toolkit installieren (für GPU)
#    Download: https://developer.nvidia.com/cuda-downloads

# 2. PyTorch mit CUDA installieren
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Alle anderen Packages
pip install -r requirements.txt

# 4. GPU testen
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## ⚙️ .env KONFIGURATION

Erstelle `.env` Datei im Projekt-Root:

```bash
# Sportmonks API
SPORTMONKS_API_TOKEN=dein_token_hier

# Optional: Alert-System
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=

# Optional: Email Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
ALERT_EMAIL=
```

---

## 📊 SYSTEM-STATUS PRÜFEN

### Im Dashboard

1. Starte Dashboard: `streamlit run dashboard.py`
2. Öffne Tab 2 "System Control"
3. Schaue nach:
   - ✅ "System bereit" oben = ALLES GUT!
   - ❌ Fehlermeldung = Siehe Debug-Info

### In Sidebar

Die Sidebar zeigt detaillierte Initialisierungs-Stati:
- ✅ LogStreamManager OK
- ✅ Config loaded
- ✅ SportmonksClient OK
- ✅ PortfolioManager OK
- ✅ AlertManager OK
- ✅ APICache OK
- ✅ ModelRegistry OK

**Alle ✅ = System bereit!**

---

## 🔧 TROUBLESHOOTING

### Dashboard startet nicht

```bash
# Prüfe Python Version
python --version

# Prüfe Streamlit
pip show streamlit

# Reinstall Streamlit
pip install --upgrade streamlit
```

### Buttons reagieren nicht

1. **Tab 2 öffnen** und Status prüfen
2. **Sidebar** checken für Fehler
3. **F5 drücken** zum Neuladen
4. Wenn Fehler: **Dependencies installieren**

### "FileNotFoundError" beim Button-Klick

**Bedeutet:** Script nicht gefunden

**Prüfe:**
```bash
ls -la | grep -E "(scraper|dutching|train_ml)"
# Sollte alle Scripts zeigen
```

### Process bleibt auf "STOPPED"

**Mögliche Ursachen:**
1. Script hat Fehler → Prüfe Logs
2. Dependencies fehlen → Installiere Packages
3. Python-Path falsch → Prüfe `cwd`

---

## ✅ ERFOLGS-CHECKLISTE

- [ ] Python 3.10+ installiert
- [ ] Dependencies installiert (`pip install -r requirements.txt`)
- [ ] .env Datei erstellt mit API Token
- [ ] Dashboard startet ohne Fehler
- [ ] Tab 2 zeigt "✅ System bereit"
- [ ] Sidebar zeigt alle Komponenten mit ✅
- [ ] Button-Klick zeigt "🚀 ... gestartet!"
- [ ] Logs erscheinen in Echtzeit
- [ ] Process-Status ändert sich zu "RUNNING"

**Wenn alle ✅ = READY TO USE! 🎉**

---

## 📞 SUPPORT

### Bei Problemen:

1. **Prüfe Debug-Output** in Sidebar
2. **Prüfe Tab 2** System-Status
3. **Prüfe Logs** in `logs/dashboard.log`
4. **Prüfe Dependencies:**
   ```bash
   pip list | grep -E "(streamlit|pandas|numpy|plotly)"
   ```

### Logs anschauen:

```bash
# Dashboard Logs
tail -f logs/dashboard.log

# Wenn Logs fehlen:
mkdir -p logs
streamlit run dashboard.py
```

---

**Version:** 6.0.0
**Letzte Aktualisierung:** 2025-11-02
**Status:** Production-Ready mit Debug-Mode
