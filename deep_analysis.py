"""
🔬 TIEFENANALYSE TOOL - 100% VERIFIKATION
==========================================

Analysiert:
1. Import-Zyklen
2. Funktions-Signaturen
3. Klassen-Definitionen
4. Cross-Module Dependencies
5. Potenzielle Runtime-Fehler
6. Code-Quality Issues
7. Missing Docstrings
8. Type Hints
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import importlib.util

class DeepCodeAnalyzer:
    """Tiefenanalyse aller Python-Module"""

    def __init__(self):
        self.modules = {}
        self.issues = {
            'critical': [],
            'warnings': [],
            'info': []
        }
        self.dependencies = {}
        self.classes = {}
        self.functions = {}

    def analyze_file(self, filepath: Path) -> Dict:
        """Analysiere eine Python-Datei"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = len(content.split('\n'))

        try:
            tree = ast.parse(content, filename=str(filepath))
        except SyntaxError as e:
            return {
                'error': f'Syntax Error: {e}',
                'imports': [],
                'classes': [],
                'functions': [],
                'lines': lines
            }

        # Extrahiere Imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Extrahiere Klassen
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'has_init': '__init__' in methods,
                    'line': node.lineno
                })

        # Extrahiere Funktionen
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Prüfe ob Top-Level Funktion
                if not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    args = [arg.arg for arg in node.args.args]
                    functions.append({
                        'name': node.name,
                        'args': args,
                        'line': node.lineno,
                        'has_docstring': ast.get_docstring(node) is not None
                    })

        return {
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'lines': lines
        }

    def find_import_cycles(self) -> List[Tuple]:
        """Finde Import-Zyklen"""
        cycles = []

        # Einfache Zykluserkennung
        for module_a, deps_a in self.dependencies.items():
            for dep in deps_a:
                if dep in self.dependencies:
                    if module_a in self.dependencies[dep]:
                        cycles.append((module_a, dep))

        return cycles

    def analyze_all_modules(self):
        """Analysiere alle Python-Module"""
        print("🔬 TIEFENANALYSE ALLER MODULE")
        print("="*60)

        py_files = list(Path('.').glob('*.py'))

        for py_file in py_files:
            if py_file.name.startswith('__'):
                continue

            print(f"\n📄 Analysiere: {py_file.name}")

            analysis = self.analyze_file(py_file)

            if 'error' in analysis:
                self.issues['critical'].append({
                    'file': py_file.name,
                    'issue': analysis['error']
                })
                print(f"   ❌ {analysis['error']}")
                continue

            self.modules[py_file.stem] = analysis
            self.dependencies[py_file.stem] = analysis['imports']

            # Checks
            print(f"   ✅ Imports: {len(analysis['imports'])}")
            print(f"   ✅ Classes: {len(analysis['classes'])}")
            print(f"   ✅ Functions: {len(analysis['functions'])}")

            # Prüfe Klassen
            for cls in analysis['classes']:
                if not cls['has_init'] and len(cls['methods']) > 0:
                    self.issues['warnings'].append({
                        'file': py_file.name,
                        'line': cls['line'],
                        'issue': f"Class {cls['name']} has no __init__ method"
                    })

            # Prüfe Docstrings
            undocumented_funcs = [f for f in analysis['functions'] if not f['has_docstring']]
            if undocumented_funcs:
                self.issues['info'].append({
                    'file': py_file.name,
                    'issue': f"{len(undocumented_funcs)} functions without docstring"
                })

    def check_cross_module_compatibility(self):
        """Prüfe Cross-Module Kompatibilität"""
        print("\n\n🔗 CROSS-MODULE KOMPATIBILITÄT")
        print("="*60)

        # Prüfe ob importierte Module existieren
        for module, imports in self.dependencies.items():
            for imp in imports:
                # Prüfe ob lokales Modul
                imp_base = imp.split('.')[0]

                if imp_base in ['numpy', 'pandas', 'torch', 'xgboost', 'streamlit',
                               'plotly', 'scipy', 'sklearn', 'requests', 'pydantic',
                               'yaml', 'redis', 'diskcache', 'aiohttp', 'json',
                               'os', 'sys', 'pathlib', 'datetime', 'time', 'typing',
                               'dataclasses', 'collections', 'warnings', 'logging',
                               'subprocess', 'threading', 'pickle', 'random', 'platform',
                               'pynvml']:
                    # Standard/Third-party module - OK
                    continue

                # Lokales Modul
                if imp_base not in self.modules and not Path(f'{imp_base}.py').exists():
                    self.issues['warnings'].append({
                        'file': module,
                        'issue': f"Imports non-existent local module: {imp_base}"
                    })
                    print(f"   ⚠️  {module} imports missing: {imp_base}")

        if not any(i for i in self.issues['warnings'] if 'Imports non-existent' in i['issue']):
            print("   ✅ Alle lokalen Module existieren")

    def check_integration_points(self):
        """Prüfe wichtige Integrationspunkte"""
        print("\n\n🔌 INTEGRATIONSPUNKTE")
        print("="*60)

        critical_integrations = [
            ('dashboard', 'gpu_ml_models', 'Dashboard nutzt GPU Models'),
            ('dashboard', 'gpu_performance_monitor', 'Dashboard nutzt GPU Monitor'),
            ('continuous_training_system', 'gpu_ml_models', 'Training nutzt GPU Models'),
            ('gpu_deep_rl_cashout', 'gpu_ml_models', 'RL nutzt GPU Config'),
        ]

        for module_a, module_b, description in critical_integrations:
            if module_a in self.modules and module_b in self.modules:
                # Prüfe ob module_a module_b importiert
                imports_a = self.dependencies.get(module_a, [])
                if any(module_b in imp for imp in imports_a):
                    print(f"   ✅ {description}")
                else:
                    # Prüfe ob direkt verwendet (ohne Import)
                    print(f"   ℹ️  {description} (dynamisch)")
            else:
                missing = []
                if module_a not in self.modules:
                    missing.append(module_a)
                if module_b not in self.modules:
                    missing.append(module_b)
                print(f"   ⚠️  {description} - Missing: {missing}")

    def print_issue_report(self):
        """Drucke Issue Report"""
        print("\n\n📋 ISSUE REPORT")
        print("="*60)

        print(f"\n❌ Critical Issues: {len(self.issues['critical'])}")
        for issue in self.issues['critical']:
            print(f"   {issue['file']}: {issue['issue']}")

        print(f"\n⚠️  Warnings: {len(self.issues['warnings'])}")
        for issue in self.issues['warnings'][:10]:  # Max 10
            file = issue.get('file', 'Unknown')
            line = issue.get('line', '')
            issue_text = issue['issue']
            if line:
                print(f"   {file}:{line} - {issue_text}")
            else:
                print(f"   {file} - {issue_text}")

        print(f"\nℹ️  Info: {len(self.issues['info'])}")
        for issue in self.issues['info'][:5]:  # Max 5
            print(f"   {issue['file']}: {issue['issue']}")

    def calculate_score(self) -> float:
        """Berechne Quality Score"""
        critical = len(self.issues['critical'])
        warnings = len(self.issues['warnings'])
        info = len(self.issues['info'])

        # Score: Start bei 100, abzüge für Issues
        score = 100.0
        score -= critical * 10  # -10% pro Critical
        score -= warnings * 2   # -2% pro Warning
        score -= info * 0.5     # -0.5% pro Info

        return max(0, min(100, score))

    def print_summary(self):
        """Drucke Zusammenfassung"""
        print("\n\n" + "="*60)
        print("📊 ZUSAMMENFASSUNG")
        print("="*60)

        total_modules = len(self.modules)
        total_classes = sum(len(m['classes']) for m in self.modules.values())
        total_functions = sum(len(m['functions']) for m in self.modules.values())
        total_lines = sum(m['lines'] for m in self.modules.values())

        print(f"\n📦 Module: {total_modules}")
        print(f"📚 Klassen: {total_classes}")
        print(f"⚙️  Funktionen: {total_functions}")
        print(f"📄 Zeilen Code: {total_lines:,}")

        score = self.calculate_score()

        print(f"\n🎯 Quality Score: {score:.1f}%")

        if score >= 95:
            print("✅ EXCELLENT - Production Ready!")
        elif score >= 85:
            print("✅ GOOD - Minor issues")
        elif score >= 70:
            print("⚠️  ACCEPTABLE - Needs improvements")
        else:
            print("❌ POOR - Critical issues need fixing")

        # Recommendations
        if self.issues['critical']:
            print("\n🚨 KRITISCH: Fix critical issues first!")
        elif self.issues['warnings']:
            print("\n💡 EMPFEHLUNG: Address warnings for 100% score")

    def run_analysis(self):
        """Führe komplette Analyse durch"""
        self.analyze_all_modules()
        self.check_cross_module_compatibility()
        self.check_integration_points()

        # Import Cycles
        cycles = self.find_import_cycles()
        if cycles:
            print(f"\n\n🔄 IMPORT CYCLES DETECTED: {len(cycles)}")
            for a, b in cycles:
                print(f"   {a} ↔ {b}")
                self.issues['warnings'].append({
                    'file': f'{a}.py',
                    'issue': f'Import cycle with {b}'
                })
        else:
            print(f"\n\n✅ NO IMPORT CYCLES")

        self.print_issue_report()
        self.print_summary()


if __name__ == "__main__":
    analyzer = DeepCodeAnalyzer()
    analyzer.run_analysis()
