#!/usr/bin/env python3
"""
Comprehensive Repository Validation Script
Validates syntax, imports, dependencies, and interfaces
"""

import sys
import os
import ast
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json

class RepositoryValidator:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.results = {}
        self.import_graph = defaultdict(set)
        self.errors = []
        self.warnings = []

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the repository."""
        return list(self.repo_path.glob("**/*.py"))

    def validate_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Validate Python syntax by compiling the file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(file_path), 'exec')
            return True, "✅ Syntax valid"
        except SyntaxError as e:
            return False, f"❌ Syntax Error: Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"❌ Compilation Error: {str(e)}"

    def extract_imports(self, file_path: Path) -> Tuple[List[str], List[str], str]:
        """Extract all imports from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            imports = []
            from_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        from_imports.append(f"{module}.{alias.name}" if module else alias.name)

            return imports, from_imports, "✅ Imports extracted"
        except Exception as e:
            return [], [], f"❌ Import extraction failed: {str(e)}"

    def validate_imports(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate that all imports can be resolved."""
        issues = []
        imports, from_imports, msg = self.extract_imports(file_path)

        if "❌" in msg:
            return False, [msg]

        # Check standard library and installed packages
        all_imports = imports + [fi.split('.')[0] for fi in from_imports]

        for imp in set(all_imports):
            # Skip relative imports and known local modules
            if imp.startswith('.') or imp == '':
                continue

            # Skip validation for local files
            local_file = self.repo_path / f"{imp}.py"
            if local_file.exists():
                continue

            try:
                importlib.util.find_spec(imp)
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                # Check if it's a local module
                if imp not in ['unified_config', 'portfolio_manager', 'sportmonks_dutching_system',
                              'sportmonks_correct_score_system', 'train_ml_models',
                              'sportmonks_hybrid_scraper_v3_FINAL', 'alert_system',
                              'api_cache_system', 'continuous_training_system', 'gpu_ml_models',
                              'backtesting_framework', 'cashout_optimizer', 'gpu_performance_monitor',
                              'gpu_deep_rl_cashout', 'optimized_poisson_model', 'simple_dashboard',
                              'sportmonks_correct_score_scraper', 'verify_installation',
                              'test_xgboost_gpu', 'dashboard']:
                    issues.append(f"⚠️  Import '{imp}' may not be available")

        return len(issues) == 0, issues

    def detect_circular_dependencies(self) -> List[str]:
        """Detect circular import dependencies."""
        def visit(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            cycles = []
            for neighbor in self.import_graph.get(node, []):
                if neighbor not in visited:
                    cycles.extend(visit(neighbor, visited, rec_stack, path[:]))
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(" → ".join(cycle))

            rec_stack.remove(node)
            return cycles

        visited = set()
        all_cycles = []

        for node in self.import_graph.keys():
            if node not in visited:
                cycles = visit(node, visited, set(), [])
                all_cycles.extend(cycles)

        return list(set(all_cycles))

    def validate_class_interfaces(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate class definitions and their methods."""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

                    # Check for basic class structure
                    if not methods:
                        issues.append(f"⚠️  Class '{class_name}' has no methods")

                    # Check for __init__ in non-dataclass classes
                    decorators = [d.id if isinstance(d, ast.Name) else
                                 d.attr if isinstance(d, ast.Attribute) else None
                                 for d in node.decorator_list]

                    if 'dataclass' not in decorators and '__init__' not in methods and methods:
                        issues.append(f"⚠️  Class '{class_name}' might need __init__ method")

            return len(issues) == 0, issues
        except Exception as e:
            return False, [f"❌ Class validation failed: {str(e)}"]

    def validate_config_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate configuration file structure."""
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            classes = []
            dataclasses = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    decorators = [d.id if isinstance(d, ast.Name) else
                                 d.attr if isinstance(d, ast.Attribute) else None
                                 for d in node.decorator_list]
                    if 'dataclass' in decorators:
                        dataclasses.append(node.name)

            if file_path.name == 'unified_config.py':
                required_configs = ['APIConfig', 'DatabaseConfig', 'MLConfig', 'BettingConfig']
                for config in required_configs:
                    if config not in classes:
                        issues.append(f"⚠️  Missing required config class: {config}")

            return len(issues) == 0, issues
        except Exception as e:
            return False, [f"❌ Config validation failed: {str(e)}"]

    def build_import_graph(self, files: List[Path]):
        """Build import dependency graph."""
        for file_path in files:
            file_name = file_path.stem
            imports, from_imports, _ = self.extract_imports(file_path)

            for imp in imports + from_imports:
                imp_base = imp.split('.')[0]
                # Only track local imports
                if imp_base in [f.stem for f in files]:
                    self.import_graph[file_name].add(imp_base)

    def validate_critical_scripts(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Perform additional validation for critical scripts."""
        issues = []
        critical_files = {
            'sportmonks_dutching_system.py': ['DutchingSystem', 'calculate_odds'],
            'sportmonks_correct_score_system.py': ['CorrectScoreSystem'],
            'portfolio_manager.py': ['PortfolioManager'],
            'train_ml_models.py': ['train_models', 'MLModelTrainer'],
            'alert_system.py': ['AlertSystem'],
            'api_cache_system.py': ['CacheSystem', 'APICache'],
            'continuous_training_system.py': ['ContinuousTraining'],
            'gpu_ml_models.py': ['GPUMLModels'],
            'backtesting_framework.py': ['BacktestingFramework'],
        }

        if file_path.name not in critical_files:
            return True, []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef)]

            required_items = critical_files[file_path.name]
            for item in required_items:
                if item not in classes and item not in functions:
                    issues.append(f"⚠️  Expected class/function '{item}' not found")

            return len(issues) == 0, issues
        except Exception as e:
            return False, [f"❌ Critical script validation failed: {str(e)}"]

    def run_full_validation(self) -> Dict:
        """Run complete validation suite."""
        print("🔍 Starting Comprehensive Repository Validation...\n")

        files = self.find_python_files()
        print(f"📁 Found {len(files)} Python files\n")

        # Build import graph
        print("🔗 Building import dependency graph...")
        self.build_import_graph(files)

        # Check for circular dependencies
        print("🔄 Checking for circular dependencies...")
        circular_deps = self.detect_circular_dependencies()

        validation_results = {}
        total_score = 0
        max_score = 0

        for file_path in sorted(files):
            print(f"\n{'='*80}")
            print(f"📄 Validating: {file_path.name}")
            print('='*80)

            file_result = {
                'path': str(file_path),
                'name': file_path.name,
                'checks': {},
                'issues': [],
                'status': 'PASS'
            }

            # 1. Syntax validation
            syntax_ok, syntax_msg = self.validate_syntax(file_path)
            file_result['checks']['syntax'] = syntax_msg
            print(f"  Syntax: {syntax_msg}")
            if not syntax_ok:
                file_result['status'] = 'FAIL'
            else:
                total_score += 20
            max_score += 20

            # 2. Import validation
            imports_ok, import_issues = self.validate_imports(file_path)
            if imports_ok:
                file_result['checks']['imports'] = "✅ All imports valid"
                print(f"  Imports: ✅ All imports valid")
                total_score += 20
            else:
                file_result['checks']['imports'] = "⚠️  Import issues found"
                file_result['issues'].extend(import_issues)
                print(f"  Imports: ⚠️  Issues found:")
                for issue in import_issues:
                    print(f"    {issue}")
                total_score += 10
            max_score += 20

            # 3. Class interface validation
            class_ok, class_issues = self.validate_class_interfaces(file_path)
            if class_ok:
                file_result['checks']['classes'] = "✅ Class interfaces valid"
                print(f"  Classes: ✅ Interfaces valid")
                total_score += 20
            else:
                file_result['checks']['classes'] = "⚠️  Class issues found"
                file_result['issues'].extend(class_issues)
                print(f"  Classes: ⚠️  Issues found:")
                for issue in class_issues:
                    print(f"    {issue}")
                total_score += 15
            max_score += 20

            # 4. Critical script validation
            critical_ok, critical_issues = self.validate_critical_scripts(file_path)
            if file_path.name in ['sportmonks_dutching_system.py',
                                  'sportmonks_correct_score_system.py',
                                  'portfolio_manager.py', 'train_ml_models.py',
                                  'alert_system.py', 'api_cache_system.py',
                                  'continuous_training_system.py', 'gpu_ml_models.py',
                                  'backtesting_framework.py']:
                if critical_ok:
                    file_result['checks']['critical'] = "✅ Critical interfaces present"
                    print(f"  Critical: ✅ All required interfaces present")
                    total_score += 20
                else:
                    file_result['checks']['critical'] = "⚠️  Missing critical interfaces"
                    file_result['issues'].extend(critical_issues)
                    print(f"  Critical: ⚠️  Issues found:")
                    for issue in critical_issues:
                        print(f"    {issue}")
                    total_score += 5
                max_score += 20

            # 5. Config validation
            if file_path.name == 'unified_config.py':
                config_ok, config_issues = self.validate_config_file(file_path)
                if config_ok:
                    file_result['checks']['config'] = "✅ Config structure valid"
                    print(f"  Config: ✅ Structure valid")
                    total_score += 20
                else:
                    file_result['checks']['config'] = "⚠️  Config issues found"
                    file_result['issues'].extend(config_issues)
                    print(f"  Config: ⚠️  Issues found:")
                    for issue in config_issues:
                        print(f"    {issue}")
                    total_score += 10
                max_score += 20

            validation_results[file_path.name] = file_result

        # Summary
        print(f"\n{'='*80}")
        print("📊 VALIDATION SUMMARY")
        print('='*80)

        if circular_deps:
            print("\n⚠️  CIRCULAR DEPENDENCIES DETECTED:")
            for cycle in circular_deps:
                print(f"  🔄 {cycle}")
        else:
            print("\n✅ No circular dependencies detected")

        print(f"\n📈 Overall Confidence Score: {(total_score/max_score*100):.1f}%")
        print(f"   ({total_score}/{max_score} points)")

        # Status summary
        passed = sum(1 for r in validation_results.values() if r['status'] == 'PASS')
        total = len(validation_results)
        print(f"\n✅ Passed: {passed}/{total} files")
        print(f"❌ Failed: {total - passed}/{total} files")

        # Files with issues
        files_with_issues = [name for name, r in validation_results.items() if r['issues']]
        if files_with_issues:
            print(f"\n⚠️  Files with warnings/issues: {len(files_with_issues)}")
            for name in files_with_issues:
                print(f"  • {name}")

        return {
            'results': validation_results,
            'circular_dependencies': circular_deps,
            'confidence_score': round(total_score/max_score*100, 1),
            'summary': {
                'total_files': total,
                'passed': passed,
                'failed': total - passed,
                'files_with_issues': len(files_with_issues)
            }
        }

if __name__ == '__main__':
    repo_path = '/home/user/ai-dutching-v2'
    validator = RepositoryValidator(repo_path)
    results = validator.run_full_validation()

    # Save detailed results
    with open('/home/user/ai-dutching-v2/validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed report saved to: validation_report.json")
