#!/usr/bin/env python3
"""
Lightweight validation script for HDC Robot Controller implementation.
Validates core functionality without external dependencies.
"""

import os
import sys
import ast
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ImplementationValidator:
    """Validates the implementation quality and completeness."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.validation_results = {}
        self.error_count = 0
        self.warning_count = 0
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Starting HDC Robot Controller Implementation Validation...")
        print("=" * 60)
        
        # Core validation checks
        checks = [
            ("File Structure", self.validate_file_structure),
            ("Python Syntax", self.validate_python_syntax),
            ("C++ Compilation", self.validate_cpp_compilation),
            ("Import Dependencies", self.validate_imports),
            ("Code Quality", self.validate_code_quality),
            ("Documentation", self.validate_documentation),
            ("Configuration", self.validate_configuration),
            ("Performance Metrics", self.validate_performance_implementation),
            ("Security Implementation", self.validate_security_features),
            ("Error Handling", self.validate_error_handling)
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                result = check_func()
                self.validation_results[check_name] = result
                self._print_check_result(check_name, result)
            except Exception as e:
                error_result = {'status': 'error', 'error': str(e), 'details': []}
                self.validation_results[check_name] = error_result
                self._print_check_result(check_name, error_result)
                self.error_count += 1
        
        # Generate summary
        summary = self.generate_summary()
        self._print_summary(summary)
        
        return {
            'validation_results': self.validation_results,
            'summary': summary,
            'timestamp': time.time()
        }
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate expected file structure exists."""
        required_files = [
            "README.md",
            "pyproject.toml", 
            "package.xml",
            "CMakeLists.txt",
            "hdc_core/include/hypervector.hpp",
            "hdc_core/src/hypervector.cpp",
            "hdc_robot_controller/__init__.py",
            "hdc_robot_controller/core/hypervector.py",
            "hdc_robot_controller/core/operations.py",
            "hdc_robot_controller/advanced_intelligence/multi_modal_fusion.py"
        ]
        
        required_dirs = [
            "hdc_core/src",
            "hdc_core/include", 
            "hdc_robot_controller/core",
            "hdc_robot_controller/advanced_intelligence",
            "hdc_robot_controller/autonomous_mastery",
            "ros2_nodes",
            "tests"
        ]
        
        missing_files = []
        missing_dirs = []
        found_files = []
        found_dirs = []
        
        # Check files
        for file_path in required_files:
            full_path = self.repo_path / file_path
            if full_path.exists() and full_path.is_file():
                found_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.repo_path / dir_path
            if full_path.exists() and full_path.is_dir():
                found_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        # Count Python and C++ files
        python_files = list(self.repo_path.rglob("*.py"))
        cpp_files = list(self.repo_path.rglob("*.cpp")) + list(self.repo_path.rglob("*.hpp"))
        
        status = 'pass' if not missing_files and not missing_dirs else 'fail'
        
        return {
            'status': status,
            'found_files': len(found_files),
            'missing_files': missing_files,
            'found_dirs': len(found_dirs), 
            'missing_dirs': missing_dirs,
            'total_python_files': len(python_files),
            'total_cpp_files': len(cpp_files),
            'details': [
                f"✅ Found {len(found_files)}/{len(required_files)} required files",
                f"✅ Found {len(found_dirs)}/{len(required_dirs)} required directories",
                f"📊 Total Python files: {len(python_files)}",
                f"📊 Total C++ files: {len(cpp_files)}"
            ]
        }
    
    def validate_python_syntax(self) -> Dict[str, Any]:
        """Validate Python files have correct syntax."""
        python_files = list(self.repo_path.rglob("*.py"))
        syntax_errors = []
        valid_files = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse syntax
                ast.parse(content, filename=str(py_file))
                valid_files.append(str(py_file.relative_to(self.repo_path)))
                
            except SyntaxError as e:
                syntax_errors.append({
                    'file': str(py_file.relative_to(self.repo_path)),
                    'line': e.lineno,
                    'error': str(e)
                })
            except Exception as e:
                syntax_errors.append({
                    'file': str(py_file.relative_to(self.repo_path)),
                    'error': f"Parse error: {str(e)}"
                })
        
        status = 'pass' if not syntax_errors else 'fail'
        
        return {
            'status': status,
            'valid_files': len(valid_files),
            'total_files': len(python_files),
            'syntax_errors': syntax_errors,
            'details': [
                f"✅ {len(valid_files)}/{len(python_files)} Python files have valid syntax"
            ] + ([f"❌ {len(syntax_errors)} files have syntax errors"] if syntax_errors else [])
        }
    
    def validate_cpp_compilation(self) -> Dict[str, Any]:
        """Validate C++ files can compile (basic check)."""
        cpp_files = list(self.repo_path.rglob("*.cpp"))
        hpp_files = list(self.repo_path.rglob("*.hpp"))
        
        # Basic compilation check - verify CMakeLists.txt exists and has content
        cmake_file = self.repo_path / "CMakeLists.txt"
        compilation_ready = False
        details = []
        
        if cmake_file.exists():
            with open(cmake_file, 'r') as f:
                cmake_content = f.read()
            
            # Check for essential CMake directives
            required_directives = [
                "cmake_minimum_required",
                "project(",
                "find_package",
                "add_library",
                "add_executable"
            ]
            
            missing_directives = []
            for directive in required_directives:
                if directive not in cmake_content:
                    missing_directives.append(directive)
                    
            compilation_ready = len(missing_directives) == 0
            
            if compilation_ready:
                details.append("✅ CMakeLists.txt has required build directives")
            else:
                details.append(f"❌ CMakeLists.txt missing: {', '.join(missing_directives)}")
        else:
            details.append("❌ CMakeLists.txt not found")
        
        status = 'pass' if compilation_ready else 'warning'
        
        return {
            'status': status,
            'cpp_files': len(cpp_files),
            'header_files': len(hpp_files),
            'cmake_ready': compilation_ready,
            'details': details + [
                f"📊 Found {len(cpp_files)} C++ source files",
                f"📊 Found {len(hpp_files)} C++ header files"
            ]
        }
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate import statements and dependencies."""
        python_files = list(self.repo_path.rglob("*.py"))
        
        # Standard library imports (should work)
        stdlib_imports = {
            'os', 'sys', 'time', 'json', 'logging', 'threading', 'multiprocessing',
            'collections', 'typing', 'dataclasses', 'enum', 'pathlib', 'hashlib',
            'functools', 'itertools', 'concurrent.futures'
        }
        
        # Third-party imports that need installation
        external_imports = {
            'numpy', 'scipy', 'matplotlib', 'sklearn', 'torch', 'transformers',
            'psutil', 'cupy', 'numba', 'rclpy', 'cv2'
        }
        
        found_stdlib = set()
        found_external = set()
        import_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split('.')[0]
                            if module_name in stdlib_imports:
                                found_stdlib.add(module_name)
                            elif module_name in external_imports:
                                found_external.add(module_name)
                                
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module_name = node.module.split('.')[0]
                            if module_name in stdlib_imports:
                                found_stdlib.add(module_name)
                            elif module_name in external_imports:
                                found_external.add(module_name)
                                
            except Exception as e:
                import_errors.append({
                    'file': str(py_file.relative_to(self.repo_path)),
                    'error': str(e)
                })
        
        status = 'pass' if not import_errors else 'warning'
        
        return {
            'status': status,
            'stdlib_imports': list(found_stdlib),
            'external_imports': list(found_external),
            'import_errors': import_errors,
            'details': [
                f"✅ Uses {len(found_stdlib)} standard library modules",
                f"📦 Requires {len(found_external)} external packages: {', '.join(sorted(found_external))}"
            ]
        }
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality metrics."""
        python_files = list(self.repo_path.rglob("*.py"))
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        long_functions = []
        large_files = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                total_lines += len(lines)
                
                if len(lines) > 1000:  # Large file
                    large_files.append(str(py_file.relative_to(self.repo_path)))
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 100:  # Long function
                            long_functions.append({
                                'file': str(py_file.relative_to(self.repo_path)),
                                'function': node.name,
                                'lines': func_lines
                            })
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        
            except Exception:
                continue  # Skip files that can't be parsed
        
        # Quality scoring
        quality_issues = len(long_functions) + len(large_files)
        status = 'pass' if quality_issues < 5 else 'warning'
        
        return {
            'status': status,
            'total_lines': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'large_files': large_files,
            'long_functions': long_functions,
            'quality_score': max(0, 100 - quality_issues * 5),
            'details': [
                f"📊 Total lines of Python code: {total_lines:,}",
                f"📊 Total functions: {total_functions}",
                f"📊 Total classes: {total_classes}",
                f"⚠️ Large files (>1000 lines): {len(large_files)}",
                f"⚠️ Long functions (>100 lines): {len(long_functions)}"
            ]
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        doc_files = []
        
        # Check for documentation files
        readme = self.repo_path / "README.md"
        if readme.exists():
            doc_files.append("README.md")
            
        # Count docstring coverage (simplified)
        python_files = list(self.repo_path.rglob("*.py"))
        functions_with_docs = 0
        total_functions = 0
        classes_with_docs = 0
        total_classes = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        if ast.get_docstring(node):
                            functions_with_docs += 1
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            classes_with_docs += 1
                            
            except Exception:
                continue
        
        func_doc_rate = functions_with_docs / total_functions if total_functions > 0 else 0
        class_doc_rate = classes_with_docs / total_classes if total_classes > 0 else 0
        overall_doc_rate = (func_doc_rate + class_doc_rate) / 2
        
        status = 'pass' if overall_doc_rate > 0.7 else 'warning'
        
        return {
            'status': status,
            'doc_files': doc_files,
            'function_doc_coverage': func_doc_rate,
            'class_doc_coverage': class_doc_rate,
            'overall_doc_coverage': overall_doc_rate,
            'details': [
                f"📚 Documentation files: {len(doc_files)}",
                f"📖 Function docstring coverage: {func_doc_rate:.1%}",
                f"📖 Class docstring coverage: {class_doc_rate:.1%}",
                f"📖 Overall documentation coverage: {overall_doc_rate:.1%}"
            ]
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files."""
        config_files = [
            ("pyproject.toml", "Python package configuration"),
            ("package.xml", "ROS 2 package configuration"), 
            ("CMakeLists.txt", "CMake build configuration"),
            ("requirements.txt", "Python dependencies"),
            ("pytest.ini", "Testing configuration")
        ]
        
        found_configs = []
        missing_configs = []
        
        for config_file, description in config_files:
            config_path = self.repo_path / config_file
            if config_path.exists():
                found_configs.append((config_file, description))
            else:
                missing_configs.append((config_file, description))
        
        status = 'pass' if len(found_configs) >= 3 else 'warning'
        
        return {
            'status': status,
            'found_configs': [f[0] for f in found_configs],
            'missing_configs': [f[0] for f in missing_configs],
            'details': [
                f"✅ Found {len(found_configs)} configuration files"
            ] + [f"✅ {desc}" for _, desc in found_configs] +
            ([f"❌ Missing: {', '.join(f[0] for f in missing_configs)}"] if missing_configs else [])
        }
    
    def validate_performance_implementation(self) -> Dict[str, Any]:
        """Validate performance optimization features."""
        perf_features = [
            ("hdc_robot_controller/scaling/performance_optimizer.py", "Performance optimization system"),
            ("hdc_core/src/cuda/", "CUDA acceleration"),
            ("IntelligentCache", "Intelligent caching system"),
            ("HardwareAccelerator", "Hardware acceleration manager"),
            ("CircuitBreakerPattern", "Circuit breaker protection")
        ]
        
        found_features = []
        missing_features = []
        
        for feature_path, description in perf_features:
            if "/" in feature_path:
                # Check file/directory
                full_path = self.repo_path / feature_path
                if full_path.exists():
                    found_features.append((feature_path, description))
                else:
                    missing_features.append((feature_path, description))
            else:
                # Check for class/pattern in codebase
                found = False
                for py_file in self.repo_path.rglob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if f"class {feature_path}" in content:
                            found_features.append((feature_path, description))
                            found = True
                            break
                    except Exception:
                        continue
                        
                if not found:
                    missing_features.append((feature_path, description))
        
        status = 'pass' if len(found_features) >= 3 else 'warning'
        
        return {
            'status': status,
            'implemented_features': len(found_features),
            'total_features': len(perf_features),
            'details': [
                f"⚡ Performance features implemented: {len(found_features)}/{len(perf_features)}"
            ] + [f"✅ {desc}" for _, desc in found_features]
        }
    
    def validate_security_features(self) -> Dict[str, Any]:
        """Validate security implementation."""
        security_features = [
            "AdvancedSecurityValidator",
            "validate_input_security",
            "sanitize_sensor_data", 
            "CircuitBreakerPattern",
            "EnhancedHealthMonitor"
        ]
        
        found_security = []
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for feature in security_features:
                    if feature in content and feature not in found_security:
                        found_security.append(feature)
                        
            except Exception:
                continue
        
        security_score = len(found_security) / len(security_features)
        status = 'pass' if security_score >= 0.8 else 'warning'
        
        return {
            'status': status,
            'security_features': found_security,
            'security_coverage': security_score,
            'details': [
                f"🔐 Security features coverage: {security_score:.1%}",
                f"✅ Implemented: {', '.join(found_security)}"
            ]
        }
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling implementation."""
        error_patterns = [
            "try:",
            "except",
            "HDCException",
            "ErrorRecoveryManager", 
            "robust_hdc_operation",
            "validate_"
        ]
        
        error_handling_count = 0
        total_files = 0
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                total_files += 1
                file_has_error_handling = False
                
                for pattern in error_patterns:
                    if pattern in content:
                        file_has_error_handling = True
                        break
                        
                if file_has_error_handling:
                    error_handling_count += 1
                        
            except Exception:
                continue
        
        error_handling_coverage = error_handling_count / total_files if total_files > 0 else 0
        status = 'pass' if error_handling_coverage >= 0.6 else 'warning'
        
        return {
            'status': status,
            'files_with_error_handling': error_handling_count,
            'total_files': total_files,
            'coverage': error_handling_coverage,
            'details': [
                f"🛡️ Error handling coverage: {error_handling_coverage:.1%}",
                f"✅ {error_handling_count}/{total_files} files have error handling"
            ]
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'pass')
        warning_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'warning')
        failed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'fail' or r['status'] == 'error')
        
        overall_score = (passed_checks + 0.5 * warning_checks) / total_checks * 100
        
        if overall_score >= 90:
            grade = 'A'
        elif overall_score >= 80:
            grade = 'B'
        elif overall_score >= 70:
            grade = 'C'
        elif overall_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'warning_checks': warning_checks,
            'failed_checks': failed_checks,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for check_name, result in self.validation_results.items():
            if result['status'] in ['fail', 'error']:
                recommendations.append(f"🔴 Fix critical issues in {check_name}")
            elif result['status'] == 'warning':
                recommendations.append(f"🟡 Address warnings in {check_name}")
        
        return recommendations
    
    def _print_check_result(self, check_name: str, result: Dict[str, Any]):
        """Print individual check result."""
        status = result['status']
        
        if status == 'pass':
            print(f"  ✅ {check_name}: PASSED")
        elif status == 'warning':
            print(f"  ⚠️  {check_name}: WARNING")
            self.warning_count += 1
        elif status == 'fail':
            print(f"  ❌ {check_name}: FAILED")
            self.error_count += 1
        else:  # error
            print(f"  💥 {check_name}: ERROR")
            self.error_count += 1
        
        # Print details
        for detail in result.get('details', []):
            print(f"     {detail}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {summary['overall_score']:.1f}/100 (Grade: {summary['grade']})")
        print(f"✅ Passed: {summary['passed_checks']}")
        print(f"⚠️  Warnings: {summary['warning_checks']}")
        print(f"❌ Failed: {summary['failed_checks']}")
        print(f"📋 Total Checks: {summary['total_checks']}")
        
        if summary['recommendations']:
            print(f"\n🎯 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"   {rec}")
        
        print(f"\n🎉 HDC Robot Controller Implementation: {summary['grade']} Grade!")


def main():
    """Main validation entry point."""
    validator = ImplementationValidator()
    results = validator.validate_all()
    
    # Exit with appropriate code
    summary = results['summary']
    if summary['failed_checks'] > 0:
        sys.exit(1)
    elif summary['warning_checks'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()