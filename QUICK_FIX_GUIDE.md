# QUICK FIX GUIDE - Minor Issues

## Summary
**Priority:** LOW (Code quality improvements)
**Impact:** Non-blocking
**Affected Files:** 4

---

## Bare Except Clauses to Fix

### 1. dashboard.py (2 occurrences)

**Line 534:**
```python
# Current:
except:
    pass

# Replace with:
except Exception as e:
    logging.warning(f"Error in dashboard: {e}")
```

**Line 541:**
```python
# Current:
except:
    pass

# Replace with:
except Exception as e:
    logging.warning(f"Error loading data: {e}")
```

---

### 2. gpu_performance_monitor.py (2 occurrences)

**Line 143:**
```python
# Current:
except:
    return None

# Replace with:
except (ImportError, RuntimeError) as e:
    logging.warning(f"GPU not available: {e}")
    return None
```

**Line 270:**
```python
# Current:
except:
    return {}

# Replace with:
except (ImportError, RuntimeError) as e:
    logging.warning(f"Failed to get GPU stats: {e}")
    return {}
```

---

### 3. sportmonks_correct_score_system.py (1 occurrence)

**Line 504:**
```python
# Current:
except:
    continue

# Replace with:
except (ValueError, KeyError, AttributeError) as e:
    logging.warning(f"Error parsing odds data: {e}")
    continue
```

---

### 4. sportmonks_hybrid_scraper_v3_FINAL.py (2 occurrences)

**Line 197:**
```python
# Current:
except:
    return None

# Replace with:
except (requests.RequestException, ValueError) as e:
    logging.error(f"API request failed: {e}")
    return None
```

**Line 253:**
```python
# Current:
except:
    continue

# Replace with:
except (ValueError, KeyError) as e:
    logging.warning(f"Error parsing match data: {e}")
    continue
```

---

## Why Fix These?

### Current Issues
1. **Catches everything** - Including KeyboardInterrupt, SystemExit
2. **Hard to debug** - No error information logged
3. **Hides bugs** - Silent failures can cause issues

### Benefits of Fix
1. **Better error messages** - Know what went wrong
2. **Easier debugging** - Error info in logs
3. **Proper exception handling** - Only catch expected errors
4. **Respects system signals** - Ctrl+C still works

---

## When to Apply

- ✅ **Recommended:** Before production deployment
- ✅ **Optional:** For code quality standards
- ❌ **Not Required:** System works fine without these fixes

---

## Verification

After making changes, run:

```bash
python validate_repository.py
```

Expected result: Confidence score improves from 96.2% to ~99%

---

## Quick Command to Find All Bare Excepts

```bash
grep -n "except:" *.py
```

This will show you all bare except clauses with line numbers.
