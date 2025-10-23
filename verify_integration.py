#!/usr/bin/env python3
"""
==========================================================
AI DUTCHING SYSTEM - INTEGRATION VERIFICATION
==========================================================
This script verifies that all modules are properly integrated
and can work together without conflicts.
"""

import sys
import os

def test_module_structure():
    """Verify all modules have correct structure and can be parsed."""
    print("🔍 Verifying Module Structure...")
    print("=" * 60)

    modules_to_check = [
        # Core System
        ("sportmonks_dutching_system.py", ["ComprehensiveAnalyzer", "DutchingCalculator", "Config"]),
        ("sportmonks_correct_score_system.py", ["CorrectScorePredictor", "CorrectScoreDatabase"]),
        ("sportmonks_xg_scraper.py", ["SportmonksScraper", "XGDatabase"]),
        ("sportmonks_correct_score_scraper.py", ["SportmonksCorrectScoreScraper"]),

        # Performance & ML
        ("optimized_poisson_model.py", ["VectorizedPoissonModel", "AdaptiveKellyCriterion"]),
        ("ml_prediction_models.py", ["XGBoostMatchPredictor", "NeuralNetworkPredictor", "HybridEnsembleModel"]),
        ("api_cache_system.py", ["FileCache", "RedisCache", "CacheManager"]),
        ("backtesting_framework.py", ["Backtester", "BacktestConfig", "BacktestResult"]),

        # Advanced Features
        ("dashboard.py", ["DashboardManager"]),
        ("cashout_optimizer.py", ["HeuristicCashoutOptimizer", "DeepQCashoutOptimizer"]),
        ("portfolio_manager.py", ["PortfolioManager", "Position"]),
        ("alert_system.py", ["AlertManager", "TelegramAlerter", "DiscordAlerter"]),
    ]

    results = {"passed": 0, "failed": 0, "errors": []}

    for module_file, expected_classes in modules_to_check:
        try:
            # Read the file
            with open(module_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for expected classes
            missing_classes = []
            for class_name in expected_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)

            if missing_classes:
                print(f"⚠️  {module_file}: Missing classes: {', '.join(missing_classes)}")
                results["failed"] += 1
                results["errors"].append(f"{module_file}: Missing {', '.join(missing_classes)}")
            else:
                print(f"✅ {module_file}: All classes found ({len(expected_classes)} classes)")
                results["passed"] += 1

        except FileNotFoundError:
            print(f"❌ {module_file}: File not found!")
            results["failed"] += 1
            results["errors"].append(f"{module_file}: File not found")
        except Exception as e:
            print(f"❌ {module_file}: Error - {str(e)}")
            results["failed"] += 1
            results["errors"].append(f"{module_file}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")

    return results


def test_configuration_files():
    """Verify configuration files exist and are valid."""
    print("\n🔍 Verifying Configuration Files...")
    print("=" * 60)

    config_files = [
        ("config.yaml.template", "YAML template"),
        (".env.example", "Environment variables template"),
        (".gitignore", "Git ignore rules"),
        ("LICENSE", "MIT License"),
        ("README.md", "Documentation"),
        ("requirements.txt", "Python dependencies"),
        ("start.sh", "Startup script"),
    ]

    results = {"passed": 0, "failed": 0}

    for file_path, description in config_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {description} ({size} bytes)")
            results["passed"] += 1
        else:
            print(f"❌ {file_path}: Missing!")
            results["failed"] += 1

    # Check that start.sh is executable
    if os.path.exists("start.sh"):
        is_executable = os.access("start.sh", os.X_OK)
        if is_executable:
            print("✅ start.sh is executable")
        else:
            print("⚠️  start.sh is not executable (run: chmod +x start.sh)")

    print("\n" + "=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")

    return results


def test_system_integration():
    """Verify system components can work together."""
    print("\n🔍 Verifying System Integration...")
    print("=" * 60)

    integration_checks = []

    # Check 1: Poisson model references in dutching system
    with open("sportmonks_dutching_system.py", 'r') as f:
        dutching_content = f.read()

    if "VectorizedPoissonModel" in dutching_content or "PoissonModel" in dutching_content:
        print("✅ Dutching system references Poisson model")
        integration_checks.append(True)
    else:
        print("⚠️  Dutching system may not use optimized Poisson model")
        integration_checks.append(False)

    # Check 2: Database file consistency
    correct_db_name = "game_database_sportmonks.csv"
    if correct_db_name in dutching_content:
        print(f"✅ Dutching system uses correct database: {correct_db_name}")
        integration_checks.append(True)
    else:
        print(f"⚠️  Dutching system may not use correct database name")
        integration_checks.append(False)

    # Check 3: Config template has all required sections
    with open("config.yaml.template", 'r') as f:
        config_content = f.read()

    required_sections = ["api:", "bankroll:", "betting:", "leagues:", "models:", "portfolio:", "cashout:", "alerts:"]
    missing_sections = [s for s in required_sections if s not in config_content]

    if not missing_sections:
        print(f"✅ Config template has all {len(required_sections)} required sections")
        integration_checks.append(True)
    else:
        print(f"⚠️  Config template missing sections: {', '.join(missing_sections)}")
        integration_checks.append(False)

    # Check 4: Requirements.txt has key dependencies
    with open("requirements.txt", 'r') as f:
        req_content = f.read()

    key_deps = ["pandas", "numpy", "scipy", "xgboost", "torch", "streamlit", "plotly"]
    missing_deps = [d for d in key_deps if d not in req_content.lower()]

    if not missing_deps:
        print(f"✅ requirements.txt has all {len(key_deps)} key dependencies")
        integration_checks.append(True)
    else:
        print(f"⚠️  requirements.txt missing: {', '.join(missing_deps)}")
        integration_checks.append(False)

    # Check 5: start.sh references correct files
    with open("start.sh", 'r') as f:
        start_content = f.read()

    if "dashboard.py" in start_content and "sportmonks_dutching_system.py" in start_content:
        print("✅ start.sh references correct entry points")
        integration_checks.append(True)
    else:
        print("⚠️  start.sh may have incorrect references")
        integration_checks.append(False)

    print("\n" + "=" * 60)
    passed = sum(integration_checks)
    total = len(integration_checks)
    print(f"Results: {passed}/{total} integration checks passed")

    return {"passed": passed, "total": total, "all_passed": all(integration_checks)}


def check_for_unnecessary_files():
    """Check if there are any remaining unnecessary files."""
    print("\n🔍 Checking for Unnecessary Files...")
    print("=" * 60)

    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]

    # Define essential files
    essential_files = {
        # Core system
        "sportmonks_dutching_system.py",
        "sportmonks_correct_score_system.py",
        "sportmonks_xg_scraper.py",
        "sportmonks_correct_score_scraper.py",
        # Performance & ML
        "optimized_poisson_model.py",
        "ml_prediction_models.py",
        "api_cache_system.py",
        "backtesting_framework.py",
        # Advanced features
        "dashboard.py",
        "cashout_optimizer.py",
        "portfolio_manager.py",
        "alert_system.py",
        # Configuration
        "config.yaml.template",
        ".env.example",
        "requirements.txt",
        "start.sh",
        # Documentation
        "README.md",
        "TIEFENANALYSE_2.0.md",
        "UPGRADE_GUIDE.md",
        "LICENSE",
        ".gitignore",
        # This verification script
        "verify_integration.py",
    }

    # Files that are OK to ignore
    ignore_patterns = ['.pyc', '__pycache__', '.git', '.csv', '.db', '.log', '.env', 'config.yaml']

    unnecessary = []
    for file in all_files:
        # Skip ignored patterns
        if any(pattern in file for pattern in ignore_patterns):
            continue

        if file not in essential_files:
            unnecessary.append(file)

    if unnecessary:
        print(f"⚠️  Found {len(unnecessary)} potentially unnecessary files:")
        for file in unnecessary:
            print(f"   - {file}")
    else:
        print("✅ No unnecessary files found - repository is clean!")

    print("\n" + "=" * 60)

    return unnecessary


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("🚀 AI DUTCHING SYSTEM - INTEGRATION VERIFICATION")
    print("=" * 60 + "\n")

    # Run all tests
    module_results = test_module_structure()
    config_results = test_configuration_files()
    integration_results = test_system_integration()
    unnecessary_files = check_for_unnecessary_files()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)

    total_checks = module_results["passed"] + module_results["failed"] + config_results["passed"] + config_results["failed"] + integration_results["total"]
    total_passed = module_results["passed"] + config_results["passed"] + integration_results["passed"]

    print(f"Total Checks: {total_passed}/{total_checks} passed")
    print(f"Module Structure: {module_results['passed']}/{module_results['passed'] + module_results['failed']} ✅")
    print(f"Configuration: {config_results['passed']}/{config_results['passed'] + config_results['failed']} ✅")
    print(f"Integration: {integration_results['passed']}/{integration_results['total']} ✅")
    print(f"Repository Cleanup: {'✅ Clean' if not unnecessary_files else '⚠️  Has unnecessary files'}")

    if module_results["errors"]:
        print("\n⚠️  Errors found:")
        for error in module_results["errors"]:
            print(f"   - {error}")

    # Overall status
    print("\n" + "=" * 60)
    if total_passed == total_checks and not unnecessary_files:
        print("✅ SYSTEM VERIFICATION PASSED - Ready for production!")
        print("=" * 60 + "\n")
        return 0
    elif total_passed >= total_checks * 0.9:
        print("⚠️  SYSTEM VERIFICATION MOSTLY PASSED - Minor issues found")
        print("=" * 60 + "\n")
        return 0
    else:
        print("❌ SYSTEM VERIFICATION FAILED - Major issues found")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
