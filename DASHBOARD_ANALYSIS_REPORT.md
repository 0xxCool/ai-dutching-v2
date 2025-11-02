# COMPREHENSIVE DEEP ANALYSIS: dashboard.py
## Analysis Date: 2025-11-02
## File: /home/user/ai-dutching-v2/dashboard.py

---

## 📋 EXECUTIVE SUMMARY

This report contains a comprehensive deep analysis of dashboard.py focusing on imports, function calls, session state management, button handlers, data flow, and critical issues.

**Total Issues Found: 13**
- 🔴 **Critical Issues: 5**
- 🟡 **Medium Issues: 5**
- 🟢 **Minor Issues: 3**

---

## 1. ✅ IMPORT VALIDATION

### 1.1 Valid Imports
All imported modules exist in the project:
- ✅ `unified_config.py` → `get_config`, `ConfigManager`
- ✅ `sportmonks_dutching_system.py` → `SportmonksClient`, `OptimizedDutchingCalculator`, `Config`
- ✅ `sportmonks_correct_score_system.py` → `CorrectScoreConfig`, `CorrectScorePoissonModel`, `CorrectScoreValueCalculator`
- ✅ `alert_system.py` → `AlertManager`, `AlertConfig`, `Alert`, `AlertLevel`, `AlertType`
- ✅ `portfolio_manager.py` → `PortfolioManager`
- ✅ `api_cache_system.py` → `FileCache` (aliased as `APICache`), `CacheConfig`
- ✅ `continuous_training_system.py` → `ModelRegistry`, `ContinuousTrainingEngine`
- ✅ `backtesting_framework.py` → `Backtester`, `BacktestConfig`

### 1.2 ❌ Import Issues

#### 🔴 CRITICAL: Duplicate Import (Lines 85 & 1115)
**Location:** Lines 85 and 1115
```python
# Line 85 - Top-level import
from backtesting_framework import Backtester as BacktestingEngine

# Line 1115 - Inside function (duplicate)
from backtesting_framework import Backtester, BacktestConfig
```
**Issue:**
- `Backtester` is imported twice, once as `BacktestingEngine` at the top level but never used
- Second import inside a function is redundant and inefficient
- `BacktestConfig` should be imported at the top level

**Impact:** Memory inefficiency, code confusion
**Recommendation:** Move both imports to top level and remove duplicate

---

## 2. ⚠️ FUNCTION CALLS & METHOD INVOCATIONS

### 2.1 Constructor Validation

#### ✅ Valid Constructors
- **PortfolioManager** (Line 476):
  ```python
  PortfolioManager(bankroll=10000.0)  # ✅ Correct
  ```
  Constructor signature: `def __init__(self, bankroll: float, config: PortfolioConfig = None)`

- **SportmonksClient** (Lines 470-473):
  ```python
  SportmonksClient(api_token=api_token, config=dutching_config_instance)  # ✅ Correct
  ```
  Constructor signature: `def __init__(self, api_token: str, config: Config)`

- **AlertManager** (Line 480):
  ```python
  AlertManager(alert_config)  # ✅ Correct
  ```
  Constructor signature: `def __init__(self, config: AlertConfig)`

- **FileCache/APICache** (Line 484):
  ```python
  APICache(cache_config)  # ✅ Correct
  ```
  Constructor signature: `def __init__(self, config: CacheConfig = None)`

- **ModelRegistry** (Line 487):
  ```python
  ModelRegistry()  # ✅ Correct
  ```
  Constructor signature: `def __init__(self, registry_dir: str = "models/registry")`

### 2.2 ❌ Method Call Issues

#### 🔴 CRITICAL: Invalid BacktestConfig Instantiation (Lines 1117-1122)
**Location:** Lines 1117-1122
```python
backtest_config = BacktestConfig(
    start_date=start_date,           # ❌ INVALID FIELD
    end_date=end_date,               # ❌ INVALID FIELD
    initial_bankroll=float(initial_balance),  # ✅ Valid
    strategy_type=strategy.lower().replace(" ", "_")  # ❌ INVALID FIELD
)
```

**Issue:** BacktestConfig is a @dataclass with specific fields. The following parameters are INVALID:
- `start_date` - Does not exist in dataclass
- `end_date` - Does not exist in dataclass
- `strategy_type` - Does not exist in dataclass

**Valid fields:**
```python
@dataclass
class BacktestConfig:
    initial_bankroll: float = 1000.0
    kelly_cap: float = 0.25
    min_odds: float = 1.1
    max_odds: float = 100.0
    min_edge: float = -0.05
    max_stake_percent: float = 0.10
    stop_loss_percent: float = 0.50
    take_profit_percent: float = 3.0
    track_daily_stats: bool = True
    save_all_bets: bool = True
```

**Impact:** This will cause a `TypeError` at runtime
**Recommendation:** Only pass valid fields or extend the BacktestConfig dataclass

#### 🔴 CRITICAL: Missing Required Parameters in run_backtest() (Line 1137)
**Location:** Line 1137
```python
results = backtester.run_backtest()  # ❌ Missing required parameters
```

**Issue:** The `run_backtest()` method requires TWO parameters:
```python
def run_backtest(
    self,
    historical_data: pd.DataFrame,      # ❌ MISSING
    prediction_func: Callable[[pd.Series], Dict]  # ❌ MISSING
) -> BacktestResult:
```

**Impact:** This will cause a `TypeError: missing 2 required positional arguments` at runtime
**Recommendation:** Provide historical data and prediction function

#### 🟡 MEDIUM: get_portfolio_statistics() Return Type Not Validated (Line 676)
**Location:** Line 676
```python
st.session_state.portfolio_stats = portfolio_mgr.get_portfolio_statistics()
```

**Issue:** While the method exists and is called correctly, there's no validation that the return type is Dict as expected. If the method fails internally, it may return None or raise an exception.

**Current error handling:** Lines 675-679 have try-except, which is good ✅
**Recommendation:** Consider validating the dictionary structure after retrieval

---

## 3. 🔍 SESSION STATE VARIABLES

### 3.1 Complete Session State Variable List

**Basic State (Lines 421-432):**
- `last_refresh` → datetime
- `auto_refresh` → bool
- `refresh_interval` → int
- `active_bets` → List
- `portfolio_stats` → Dict
- `system_alerts` → List

**Manager & Components (Lines 435-489):**
- `log_manager` → LogStreamManager
- `scraper_logs` → List[str]
- `dutching_logs` → List[str]
- `ml_logs` → List[str]
- `portfolio_logs` → List[str]
- `alert_logs` → List[str]
- `process_states` → Dict[str, str]
- `sportmonks_client` → SportmonksClient
- `portfolio_manager` → PortfolioManager
- `alert_manager` → AlertManager
- `api_cache` → FileCache/APICache
- `model_registry` → ModelRegistry
- `components_initialized` → bool

**Dynamic Variables:**
- `correct_score_logs` → List[str] (Line 1243)
- `backtest_results` → Dict (Line 1161)

### 3.2 ❌ Session State Issues

#### 🔴 CRITICAL: Missing process_states Initialization for 'correct_score'
**Location:** Lines 1210-1223 (usage) vs Lines 452-459 (initialization)

**Issue:** The `process_states` dictionary is initialized with only 5 processes:
```python
st.session_state.process_states = {
    'scraper': 'idle',
    'dutching': 'idle',
    'ml_training': 'idle',
    'portfolio': 'idle',
    'alerts': 'idle'
    # ❌ 'correct_score' is MISSING
}
```

But 'correct_score' is used in lines 1210-1223 without initialization.

**Impact:** Can cause KeyError if accessed before being set, or inconsistent state
**Recommendation:** Add 'correct_score': 'idle' to initial process_states

#### 🟡 MEDIUM: Conditional Session State Access (Line 1304)
**Location:** Line 1304
```python
stats = st.session_state.get('portfolio_stats', {})
```

**Issue:** Uses `.get()` with fallback while other places access directly (e.g., line 948)
```python
stats = st.session_state.portfolio_stats  # Line 948 - direct access
```

**Impact:** Inconsistent access pattern, potential for bugs if not initialized
**Recommendation:** Use consistent access pattern throughout (prefer `.get()` with defaults)

#### 🟢 MINOR: Late Initialization of correct_score_logs (Line 1243)
**Location:** Line 1243
```python
if 'correct_score_logs' not in st.session_state:
    st.session_state.correct_score_logs = []
```

**Issue:** Should be initialized in `init_session_state()` like other log buffers
**Impact:** Low - handled with defensive check, but inconsistent with other log initialization
**Recommendation:** Move to `init_session_state()` for consistency

---

## 4. 🖱️ BUTTON CLICK HANDLERS

### 4.1 Button Handler Overview

**System Control Buttons (Tab 2):**
- ✅ Line 766: "Start Scraper" → `start_scraper()`
- ✅ Line 770: "Stop Scraper" → `stop_scraper()`
- ✅ Line 774: "Refresh Logs" → `update_logs('scraper')`
- ✅ Line 789: "Start Dutching" → `start_dutching()`
- ✅ Line 793: "Stop Dutching" → `stop_dutching()`
- ✅ Line 797: "Refresh Logs" → `update_logs('dutching')`
- ✅ Line 813: "Start ML Training" → `start_ml_training()`
- ✅ Line 822: "Start Portfolio" → `start_portfolio_optimizer()`
- ✅ Line 831: "Start Alerts" → `start_alert_system()`
- ✅ Line 845: "START ALL SYSTEMS" → Multiple function calls
- ✅ Line 859: "STOP ALL SYSTEMS" → `log_manager.stop_all()`
- ✅ Line 867: "RESTART ALL" → Stop + Start all

**ML Models (Tab 3):**
- ✅ Line 933: "Start Training" → Mock training loop

**Settings (Tab 6):**
- ✅ Line 1072: "Manual Refresh Now" → `st.rerun()`
- ✅ Line 1086: "Save Settings" → Shows success message (no actual save)

**Backtesting (Tab 7):**
- ❌ Line 1106: "Run Backtest" → Has critical issues (see below)

**Correct Score (Tab 8):**
- ✅ Line 1205: "Start Correct Score" → Starts process
- ✅ Line 1220: "Stop Correct Score" → Stops process
- ✅ Line 1235: "Save CS Config" → Shows success message

**Sidebar:**
- ✅ Line 1352: "Dashboard neuladen" → `st.rerun()`
- ✅ Line 1355: "Start All Systems" → Multiple function calls
- ✅ Line 1365: "Stop All Systems" → `log_manager.stop_all()`

### 4.2 ❌ Button Handler Issues

#### 🔴 CRITICAL: Run Backtest Button Handler Issues (Lines 1106-1189)
**Location:** Lines 1106-1189

**Multiple Issues:**

1. **Invalid BacktestConfig** (Lines 1117-1122) - Already documented above
2. **Missing run_backtest() parameters** (Line 1137) - Already documented above
3. **FileNotFoundError not properly handled:**
   ```python
   data_file = Path.cwd() / "game_database_complete.csv"
   if not data_file.exists():
       st.warning("⚠️ Keine Daten gefunden...")
   else:
       # Creates backtester anyway without checking if data can be loaded
   ```

**Impact:** Button will fail with TypeError when clicked
**Recommendation:** Complete rewrite of backtest logic with proper parameters

#### 🟡 MEDIUM: No Validation in Save Settings Button (Line 1086)
**Location:** Line 1086
```python
if st.button("💾 Save Settings"):
    st.success("✅ Settings saved successfully!")
```

**Issue:**
- Button shows success but doesn't actually save anything
- Variables `max_stake`, `max_daily_loss`, `min_edge`, `kelly_fraction` are collected (lines 1081-1084) but never used
- No persistence to config file or session state

**Impact:** User expects settings to be saved but they're lost on refresh
**Recommendation:** Actually save to config or session state

#### 🟡 MEDIUM: No Validation in Save CS Config Button (Line 1235)
**Location:** Line 1235
```python
if st.button("💾 Save CS Config", key="save_cs_config"):
    st.success("✅ Konfiguration gespeichert!")
```

**Issue:** Same as above - collects `min_value_edge`, `max_odds`, `min_probability` (lines 1231-1233) but never saves them

**Impact:** User expects settings to be saved but they're lost on refresh
**Recommendation:** Actually save to config or session state

---

## 5. 🔄 DATA FLOW ANALYSIS

### 5.1 Data Flow Patterns

#### Portfolio Stats Flow
```
init_session_state()
  → portfolio_manager = PortfolioManager(bankroll=10000.0)
  → portfolio_stats = {}

main()
  → portfolio_mgr = st.session_state.portfolio_manager
  → portfolio_stats = portfolio_mgr.get_portfolio_statistics()
  → st.session_state.portfolio_stats = stats

Tab 4 & Sidebar
  → stats = st.session_state.portfolio_stats
  → Display metrics
```
**Status:** ✅ Correct flow with error handling

#### Log Streaming Flow
```
init_session_state()
  → log_manager = LogStreamManager()
  → {process}_logs = []

Button Click
  → start_{process}()
    → log_manager.start_process(name, command, cwd)
    → process_states[name] = 'running'

update_logs(process_name)
  → new_logs = log_manager.get_logs(process_name)
  → session_state.{process}_logs.extend(new_logs)
  → Keep last 100 lines

display_live_logs(logs, container)
  → Show logs in UI
```
**Status:** ✅ Correct flow

#### Auto-Refresh Flow (Lines 1390-1401)
```
if auto_refresh:
  → Calculate time_since_refresh
  → For each running process: update_logs()
  → If time >= interval: st.rerun()
```
**Status:** ✅ Correct flow

### 5.2 ❌ Data Flow Issues

#### 🟡 MEDIUM: Potential Race Condition in Log Updates
**Location:** Lines 1394-1396
```python
for process_name in ['scraper', 'dutching', 'ml_training', 'portfolio', 'alerts']:
    if st.session_state.log_manager.is_running(process_name):
        update_logs(process_name)
```

**Issue:** The list is hardcoded and doesn't include 'correct_score', so correct_score logs won't be auto-updated even if the process is running.

**Impact:** Correct Score logs won't update automatically
**Recommendation:** Use process_states.keys() instead of hardcoded list

#### 🟢 MINOR: Inefficient Log Slicing (Multiple locations)
**Location:** Lines 559-573
```python
st.session_state.scraper_logs.extend(new_logs)
st.session_state.scraper_logs = st.session_state.scraper_logs[-100:]
```

**Issue:** Creates a new list every time, could use deque for O(1) operations
**Impact:** Minor performance impact with many log updates
**Recommendation:** Use `collections.deque(maxlen=100)` instead of list

---

## 6. 🚨 CRITICAL ISSUES SUMMARY

### 6.1 Issues Ranked by Severity

#### 🔴 CRITICAL (Will Cause Runtime Errors)

1. **Invalid BacktestConfig Parameters** (Lines 1117-1122)
   - Will cause: `TypeError: __init__() got unexpected keyword arguments`
   - Fix Priority: **HIGH**

2. **Missing run_backtest() Parameters** (Line 1137)
   - Will cause: `TypeError: run_backtest() missing 2 required positional arguments`
   - Fix Priority: **HIGH**

3. **Duplicate Backtester Import** (Lines 85, 1115)
   - Will cause: Confusion, potential naming conflicts
   - Fix Priority: **MEDIUM**

4. **Missing process_states['correct_score'] Initialization**
   - Will cause: Inconsistent state, potential KeyError
   - Fix Priority: **MEDIUM**

5. **Redundant If-Else Logic** (Lines 1210-1213)
   - Will cause: No error, but illogical code
   - Fix Priority: **LOW**

#### 🟡 MEDIUM (May Cause Issues or Unexpected Behavior)

1. **No Actual Settings Save** (Lines 1086, 1235)
   - Impact: User confusion, lost settings
   - Fix Priority: **MEDIUM**

2. **Hardcoded Process List in Auto-Refresh** (Lines 1394-1396)
   - Impact: Correct Score logs won't auto-update
   - Fix Priority: **MEDIUM**

3. **Inconsistent Session State Access** (Lines 948 vs 1304)
   - Impact: Potential KeyError in some cases
   - Fix Priority: **LOW**

4. **No Return Type Validation** (Line 676)
   - Impact: Potential for unhandled exceptions
   - Fix Priority: **LOW**

5. **Inefficient Log Buffer** (Multiple locations)
   - Impact: Minor performance degradation
   - Fix Priority: **LOW**

#### 🟢 MINOR (Code Quality Issues)

1. **Late correct_score_logs Initialization** (Line 1243)
   - Impact: Inconsistency, no functional issue
   - Fix Priority: **LOW**

2. **Unused BacktestingEngine Import** (Line 85)
   - Impact: Wasted memory
   - Fix Priority: **LOW**

3. **No Validation in Mock Data Paths** (Line 1261)
   - Impact: Could fail if file doesn't exist
   - Fix Priority: **LOW**

---

## 7. 📊 STATISTICS

### Code Metrics
- **Total Lines:** 1419
- **Total Functions:** 15 (including main)
- **Total Classes:** 1 (LogStreamManager)
- **Session State Variables:** 18
- **Button Handlers:** 20
- **Import Statements:** 50+

### Issue Distribution
- **Import Issues:** 1 (duplicate)
- **Function Call Issues:** 3 (invalid params, missing params)
- **Session State Issues:** 3 (missing init, inconsistent access)
- **Button Handler Issues:** 3 (non-functional saves, broken backtest)
- **Data Flow Issues:** 2 (race condition, inefficiency)
- **Logic Issues:** 1 (redundant if-else)

---

## 8. ✅ POSITIVE FINDINGS

### Well-Implemented Features

1. **✅ Excellent Error Handling in Process Management**
   - All start/stop functions have try-except blocks
   - Proper logging of errors
   - User-friendly error messages

2. **✅ Comprehensive Session State Initialization**
   - Single initialization function
   - Prevents re-initialization with checks
   - Good separation of concerns

3. **✅ Robust Log Streaming System**
   - Queue-based architecture
   - Thread-safe operations
   - Proper cleanup on exit

4. **✅ Good UI/UX Design**
   - Well-organized tabs
   - Responsive layout
   - Live updates with auto-refresh

5. **✅ Proper Resource Cleanup**
   - atexit handler for cleanup (Line 1413)
   - Process termination with timeout
   - Graceful shutdown

---

## 9. 🔧 RECOMMENDED FIXES

### Priority 1: Critical Fixes (Must Fix Before Production)

1. **Fix BacktestConfig instantiation:**
   ```python
   # Current (BROKEN):
   backtest_config = BacktestConfig(
       start_date=start_date,
       end_date=end_date,
       initial_bankroll=float(initial_balance),
       strategy_type=strategy.lower().replace(" ", "_")
   )

   # Fixed:
   backtest_config = BacktestConfig(
       initial_bankroll=float(initial_balance)
   )
   # Handle start_date, end_date, strategy_type separately
   ```

2. **Fix run_backtest() call:**
   ```python
   # Current (BROKEN):
   results = backtester.run_backtest()

   # Fixed:
   # Load historical data
   historical_data = pd.read_csv("game_database_complete.csv")

   # Define prediction function
   def prediction_func(row):
       # Your prediction logic here
       return {'outcome': 'Home', 'probability': 0.6, 'odds': 2.5}

   results = backtester.run_backtest(historical_data, prediction_func)
   ```

3. **Remove duplicate import:**
   ```python
   # At top (Line 85):
   from backtesting_framework import Backtester, BacktestConfig

   # Remove line 1115 entirely
   ```

4. **Add correct_score to process_states initialization:**
   ```python
   if 'process_states' not in st.session_state:
       st.session_state.process_states = {
           'scraper': 'idle',
           'dutching': 'idle',
           'ml_training': 'idle',
           'portfolio': 'idle',
           'alerts': 'idle',
           'correct_score': 'idle'  # ADD THIS
       }
   ```

### Priority 2: Important Fixes

1. **Fix auto-refresh process list:**
   ```python
   # Current:
   for process_name in ['scraper', 'dutching', 'ml_training', 'portfolio', 'alerts']:

   # Fixed:
   for process_name in st.session_state.process_states.keys():
   ```

2. **Implement actual settings save:**
   ```python
   if st.button("💾 Save Settings"):
       st.session_state.betting_settings = {
           'max_stake': max_stake,
           'max_daily_loss': max_daily_loss,
           'min_edge': min_edge,
           'kelly_fraction': kelly_fraction
       }
       st.success("✅ Settings saved successfully!")
   ```

3. **Use consistent session state access:**
   ```python
   # Always use .get() with default:
   stats = st.session_state.get('portfolio_stats', {})
   ```

### Priority 3: Code Quality Improvements

1. **Use deque for log buffers:**
   ```python
   from collections import deque

   if 'scraper_logs' not in st.session_state:
       st.session_state.scraper_logs = deque(maxlen=100)
   ```

2. **Move correct_score_logs to init_session_state:**
   ```python
   if 'correct_score_logs' not in st.session_state:
       st.session_state.correct_score_logs = []
   ```

3. **Fix redundant if-else:**
   ```python
   # Current (WRONG):
   if 'correct_score' not in st.session_state.process_states:
       st.session_state.process_states['correct_score'] = 'running'
   else:
       st.session_state.process_states['correct_score'] = 'running'

   # Fixed:
   st.session_state.process_states['correct_score'] = 'running'
   ```

---

## 10. 🎯 CONCLUSION

The dashboard.py file is generally well-structured with good error handling and UI design. However, there are **5 critical issues** that will cause runtime errors, particularly in the backtesting functionality.

### Must Fix Before Running:
1. ❌ BacktestConfig invalid parameters
2. ❌ run_backtest() missing parameters
3. ❌ Duplicate imports
4. ⚠️ Missing process_states initialization
5. ⚠️ Auto-refresh missing correct_score

### Overall Assessment:
- **Functionality:** 75/100 (Critical bugs in backtesting)
- **Code Quality:** 80/100 (Good structure, minor issues)
- **Error Handling:** 90/100 (Excellent in most areas)
- **Maintainability:** 75/100 (Some inconsistencies)

**Recommendation:** Fix all Critical (🔴) issues before deploying to production. Address Medium (🟡) issues in next iteration.

---

## 📝 APPENDIX A: Complete Session State Variable Reference

| Variable | Type | Initialized | Used | Purpose |
|----------|------|-------------|------|---------|
| `last_refresh` | datetime | Line 422 | Lines 1076, 1381, 1391, 1400 | Track last UI refresh time |
| `auto_refresh` | bool | Line 424 | Lines 1047, 1051-1052, 1390 | Enable/disable auto-refresh |
| `refresh_interval` | int | Line 426 | Lines 1063, 1067-1068, 1399 | Refresh interval in seconds |
| `active_bets` | List | Line 428 | Not used | Placeholder for active bets |
| `portfolio_stats` | Dict | Line 430 | Lines 676, 679, 948, 1304 | Portfolio statistics |
| `system_alerts` | List | Line 432 | Not used | Placeholder for alerts |
| `log_manager` | LogStreamManager | Line 436 | Multiple | Manages subprocess logs |
| `scraper_logs` | List[str] | Line 441 | Lines 559, 561, 781 | Scraper log buffer |
| `dutching_logs` | List[str] | Line 443 | Lines 563-564, 804 | Dutching log buffer |
| `ml_logs` | List[str] | Line 445 | Lines 566-567, 818 | ML training log buffer |
| `portfolio_logs` | List[str] | Line 447 | Lines 569-570, 827 | Portfolio log buffer |
| `alert_logs` | List[str] | Line 449 | Lines 572-573, 836 | Alert log buffer |
| `process_states` | Dict | Line 453 | Multiple | Process running states |
| `sportmonks_client` | SportmonksClient | Line 470 | Not used | API client |
| `portfolio_manager` | PortfolioManager | Line 476 | Line 672, 676 | Portfolio management |
| `alert_manager` | AlertManager | Line 480 | Not used | Alert management |
| `api_cache` | FileCache | Line 484 | Not used | API response cache |
| `model_registry` | ModelRegistry | Line 487 | Not used | ML model registry |
| `components_initialized` | bool | Line 489 | Line 462 | Init flag |
| `correct_score_logs` | List[str] | Line 1243 | Lines 1249, 1251-1252, 1254 | Correct score log buffer |
| `backtest_results` | Dict | Line 1161 | None | Backtest results storage |

---

## 📝 APPENDIX B: Complete Function Reference

| Function | Lines | Called By | Purpose | Status |
|----------|-------|-----------|---------|--------|
| `init_session_state()` | 417-494 | Line 497 | Initialize session state | ✅ Working |
| `get_gpu_stats()` | 502-552 | Line 886 | Get GPU statistics | ✅ Working |
| `update_logs()` | 554-573 | Multiple buttons | Update log buffers | ✅ Working |
| `display_live_logs()` | 575-585 | Multiple locations | Display logs in UI | ✅ Working |
| `start_scraper()` | 587-597 | Lines 766, 847 | Start scraper process | ✅ Working |
| `stop_scraper()` | 599-607 | Lines 770, 602 | Stop scraper process | ✅ Working |
| `start_dutching()` | 609-619 | Lines 789, 849 | Start dutching process | ✅ Working |
| `stop_dutching()` | 621-629 | Lines 793, 624 | Stop dutching process | ✅ Working |
| `start_ml_training()` | 631-641 | Lines 813, 851 | Start ML training | ✅ Working |
| `start_portfolio_optimizer()` | 643-653 | Lines 822, 853 | Start portfolio optimizer | ✅ Working |
| `start_alert_system()` | 655-665 | Lines 831, 855 | Start alert system | ✅ Working |
| `main()` | 670-1292 | Line 1419 | Main application function | ⚠️ Has issues |
| `cleanup()` | 1408-1411 | atexit (1413) | Cleanup on exit | ✅ Working |

---

*End of Report*
