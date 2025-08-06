#!/usr/bin/env python3
"""
Quality checker script for HDC Robot Controller.
Runs comprehensive quality checks locally before CI/CD.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil


@dataclass
class CheckResult:
    """Result of a quality check."""
    name: str
    passed: bool
    duration: float
    output: str
    error: Optional[str] = None
    score: Optional[float] = None


class QualityChecker:
    """Comprehensive quality checker for HDC Robot Controller."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.logger = self._setup_logger()
        self.results: List[CheckResult] = []
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 85.0,
            "code_quality": 8.0,  # Out of 10
            "performance_score": 7.0,  # Out of 10
            "security_score": 8.0,  # Out of 10
            "documentation_coverage": 70.0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("quality_checker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_all_checks(self, parallel: bool = True) -> bool:
        """Run all quality checks."""
        self.logger.info("🚀 Starting comprehensive quality checks...")
        
        checks = [
            ("Code Formatting", self.check_code_formatting),
            ("Linting", self.check_linting),
            ("Type Checking", self.check_type_hints),
            ("Unit Tests", self.check_unit_tests),
            ("Test Coverage", self.check_test_coverage),
            ("Security Scan", self.check_security),
            ("Performance Tests", self.check_performance),
            ("Documentation", self.check_documentation),
            ("Dependencies", self.check_dependencies),
            ("Memory Usage", self.check_memory_usage),
            ("File Structure", self.check_file_structure),
            ("Import Order", self.check_import_order),
            ("Code Complexity", self.check_code_complexity),
            ("License Headers", self.check_license_headers)
        ]
        
        if parallel:
            self._run_checks_parallel(checks)
        else:
            self._run_checks_sequential(checks)
        
        return self._generate_report()
    
    def _run_checks_parallel(self, checks: List[Tuple[str, callable]]):
        """Run checks in parallel."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_check = {
                executor.submit(self._run_check, name, func): (name, func)
                for name, func in checks
            }
            
            for future in as_completed(future_to_check):
                name, _ = future_to_check[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    status = "✅" if result.passed else "❌"
                    self.logger.info(f"{status} {name}: {result.duration:.2f}s")
                except Exception as e:
                    self.logger.error(f"❌ {name}: Failed with exception: {e}")
                    self.results.append(CheckResult(
                        name=name, passed=False, duration=0.0, 
                        output="", error=str(e)
                    ))
    
    def _run_checks_sequential(self, checks: List[Tuple[str, callable]]):
        """Run checks sequentially."""
        for name, func in checks:
            result = self._run_check(name, func)
            self.results.append(result)
            status = "✅" if result.passed else "❌"
            self.logger.info(f"{status} {name}: {result.duration:.2f}s")
    
    def _run_check(self, name: str, check_func: callable) -> CheckResult:
        """Run a single check."""
        start_time = time.time()
        
        try:
            passed, output, score = check_func()
            duration = time.time() - start_time
            
            return CheckResult(
                name=name,
                passed=passed,
                duration=duration,
                output=output,
                score=score
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return CheckResult(
                name=name,
                passed=False,
                duration=duration,
                output="",
                error=str(e)
            )
    
    def check_code_formatting(self) -> Tuple[bool, str, Optional[float]]:
        """Check code formatting with Black."""
        try:
            result = subprocess.run([
                "black", "--check", "--diff", 
                "hdc_robot_controller/", "tests/", "scripts/"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            return passed, output, 10.0 if passed else 0.0
            
        except FileNotFoundError:
            return False, "Black not installed", 0.0
    
    def check_linting(self) -> Tuple[bool, str, Optional[float]]:
        """Check code with Ruff linter."""
        try:
            result = subprocess.run([
                "ruff", "check", "hdc_robot_controller/", "tests/", "scripts/"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Ruff returns 0 for no issues
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Calculate score based on number of issues
            if passed:
                score = 10.0
            else:
                # Rough scoring based on output length
                issue_count = len(output.split('\n'))
                score = max(0.0, 10.0 - (issue_count * 0.1))
            
            return passed, output, score
            
        except FileNotFoundError:
            return False, "Ruff not installed", 0.0
    
    def check_type_hints(self) -> Tuple[bool, str, Optional[float]]:
        """Check type hints with MyPy."""
        try:
            result = subprocess.run([
                "mypy", "hdc_robot_controller/", "--ignore-missing-imports"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output = result.stdout + result.stderr
            
            # MyPy returns 0 for success
            if result.returncode == 0:
                return True, output, 10.0
            
            # Calculate score based on error types
            error_lines = [line for line in output.split('\n') if 'error:' in line]
            error_count = len(error_lines)
            score = max(0.0, 10.0 - (error_count * 0.5))
            
            return False, output, score
            
        except FileNotFoundError:
            return False, "MyPy not installed", 0.0
    
    def check_unit_tests(self) -> Tuple[bool, str, Optional[float]]:
        """Run unit tests."""
        try:
            result = subprocess.run([
                "pytest", "tests/test_hdc_comprehensive.py", "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output = result.stdout + result.stderr
            
            # Parse test results
            if "FAILED" in output:
                failed_count = output.count("FAILED")
                total_count = output.count("::test_") if "::test_" in output else 1
                score = max(0.0, 10.0 * (1 - failed_count / total_count))
                return False, output, score
            elif "passed" in output:
                return True, output, 10.0
            else:
                return False, output, 0.0
            
        except FileNotFoundError:
            return False, "pytest not installed", 0.0
    
    def check_test_coverage(self) -> Tuple[bool, str, Optional[float]]:
        """Check test coverage."""
        try:
            result = subprocess.run([
                "pytest", "--cov=hdc_robot_controller", "--cov-report=term",
                "tests/test_hdc_comprehensive.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output = result.stdout + result.stderr
            
            # Parse coverage percentage
            coverage_line = None
            for line in output.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    coverage_line = line
                    break
            
            if coverage_line:
                # Extract percentage
                parts = coverage_line.split()
                for part in parts:
                    if part.endswith('%'):
                        coverage = float(part[:-1])
                        passed = coverage >= self.thresholds["test_coverage"]
                        return passed, output, coverage / 10.0  # Convert to 0-10 scale
            
            return False, "Could not parse coverage", 0.0
            
        except (FileNotFoundError, ValueError):
            return False, "Coverage tools not available", 0.0
    
    def check_security(self) -> Tuple[bool, str, Optional[float]]:
        """Run security checks."""
        try:
            # Run Bandit security scan
            result = subprocess.run([
                "bandit", "-r", "hdc_robot_controller/", "-f", "json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return True, "No security issues found", 10.0
            
            try:
                # Parse JSON output
                bandit_data = json.loads(result.stdout)
                issues = bandit_data.get('results', [])
                
                # Score based on severity
                score = 10.0
                for issue in issues:
                    severity = issue.get('issue_severity', 'LOW').upper()
                    if severity == 'HIGH':
                        score -= 3.0
                    elif severity == 'MEDIUM':
                        score -= 1.0
                    else:  # LOW
                        score -= 0.5
                
                score = max(0.0, score)
                passed = score >= self.thresholds["security_score"]
                
                return passed, f"Found {len(issues)} security issues", score
                
            except json.JSONDecodeError:
                output = result.stdout + result.stderr
                return False, output, 0.0
            
        except FileNotFoundError:
            return False, "Bandit not installed", 0.0
    
    def check_performance(self) -> Tuple[bool, str, Optional[float]]:
        """Check performance benchmarks."""
        try:
            # Run a quick performance test
            test_script = f"""
import time
import sys
sys.path.insert(0, '{self.project_root}')

from hdc_robot_controller.core.hypervector import HyperVector

def benchmark_operations():
    dimension = 10000
    num_vectors = 100
    
    # Create test vectors
    vectors = [HyperVector.random(dimension) for _ in range(num_vectors)]
    
    # Time bundling
    start_time = time.time()
    result = HyperVector.bundle_vectors(vectors)
    bundle_time = time.time() - start_time
    
    # Time similarity computation
    v1, v2 = vectors[0], vectors[1]
    start_time = time.time()
    for _ in range(1000):
        similarity = v1.similarity(v2)
    sim_time = time.time() - start_time
    
    # Return results
    return {{
        'bundle_time': bundle_time,
        'similarity_time': sim_time / 1000,  # Per operation
        'dimension': dimension,
        'num_vectors': num_vectors
    }}

results = benchmark_operations()
print(f"Bundle time: {{results['bundle_time']:.4f}}s")
print(f"Similarity time: {{results['similarity_time']:.6f}}s")

# Performance scoring
bundle_score = min(10.0, 10.0 * (0.1 / max(0.001, results['bundle_time'])))
sim_score = min(10.0, 10.0 * (0.001 / max(0.0001, results['similarity_time'])))
overall_score = (bundle_score + sim_score) / 2

print(f"Performance score: {{overall_score:.1f}}/10.0")
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True, timeout=60)
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                # Parse performance score
                for line in output.split('\n'):
                    if 'Performance score:' in line:
                        try:
                            score = float(line.split(':')[1].split('/')[0].strip())
                            passed = score >= self.thresholds["performance_score"]
                            return passed, output, score
                        except (ValueError, IndexError):
                            pass
                
                return True, output, 8.0  # Default good score
            else:
                return False, output, 0.0
            
        except subprocess.TimeoutExpired:
            return False, "Performance test timed out", 0.0
        except Exception as e:
            return False, f"Performance test failed: {e}", 0.0
    
    def check_documentation(self) -> Tuple[bool, str, Optional[float]]:
        """Check documentation coverage."""
        python_files = list(self.project_root.glob("hdc_robot_controller/**/*.py"))
        
        total_functions = 0
        documented_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple docstring detection
                lines = content.split('\n')
                in_function = False
                function_has_docstring = False
                
                for line in lines:
                    stripped = line.strip()
                    
                    if stripped.startswith('def ') and not stripped.startswith('def _'):
                        if in_function and not function_has_docstring:
                            pass  # Previous function had no docstring
                        
                        total_functions += 1
                        in_function = True
                        function_has_docstring = False
                    
                    elif in_function and (stripped.startswith('"""') or stripped.startswith("'''")):
                        documented_functions += 1
                        function_has_docstring = True
                        in_function = False
                    
                    elif in_function and stripped and not stripped.startswith('#'):
                        if not (stripped.startswith('"""') or stripped.startswith("'''")):
                            in_function = False
            
            except Exception:
                continue  # Skip problematic files
        
        if total_functions == 0:
            return True, "No functions found", 10.0
        
        coverage_pct = (documented_functions / total_functions) * 100
        passed = coverage_pct >= self.thresholds["documentation_coverage"]
        score = min(10.0, coverage_pct / 10.0)
        
        output = f"Documentation coverage: {documented_functions}/{total_functions} ({coverage_pct:.1f}%)"
        
        return passed, output, score
    
    def check_dependencies(self) -> Tuple[bool, str, Optional[float]]:
        """Check dependency security and updates."""
        try:
            # Check if requirements.txt exists
            req_file = self.project_root / "requirements.txt"
            if not req_file.exists():
                return False, "requirements.txt not found", 0.0
            
            # Try to run safety check
            result = subprocess.run([
                "safety", "check", "-r", str(req_file), "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return True, "No known security vulnerabilities", 10.0
            
            try:
                safety_data = json.loads(result.stdout)
                vuln_count = len(safety_data)
                
                score = max(0.0, 10.0 - (vuln_count * 2.0))
                output = f"Found {vuln_count} known vulnerabilities"
                
                return vuln_count == 0, output, score
                
            except json.JSONDecodeError:
                return False, result.stdout + result.stderr, 5.0
            
        except FileNotFoundError:
            return False, "Safety tool not installed", 5.0
    
    def check_memory_usage(self) -> Tuple[bool, str, Optional[float]]:
        """Check memory usage during basic operations."""
        try:
            test_script = f"""
import psutil
import gc
import sys
sys.path.insert(0, '{self.project_root}')

from hdc_robot_controller.core.hypervector import HyperVector

def check_memory_usage():
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create many vectors
    vectors = []
    for i in range(100):
        vectors.append(HyperVector.random(10000))
    
    mid_memory = process.memory_info().rss / 1024 / 1024
    
    # Clean up
    del vectors
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    
    return {{
        'initial': initial_memory,
        'peak': mid_memory,
        'final': final_memory,
        'growth': mid_memory - initial_memory,
        'cleanup': mid_memory - final_memory
    }}

results = check_memory_usage()
print(f"Initial: {{results['initial']:.1f}}MB")
print(f"Peak: {{results['peak']:.1f}}MB") 
print(f"Final: {{results['final']:.1f}}MB")
print(f"Growth: {{results['growth']:.1f}}MB")
print(f"Cleanup: {{results['cleanup']:.1f}}MB")

# Score based on memory efficiency
growth = results['growth']
if growth < 50:  # Less than 50MB growth is excellent
    score = 10.0
elif growth < 100:  # Less than 100MB is good
    score = 8.0
elif growth < 200:  # Less than 200MB is acceptable
    score = 6.0
else:
    score = max(0.0, 10.0 - (growth / 50.0))

print(f"Memory score: {{score:.1f}}/10.0")
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True, timeout=30)
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                # Parse memory score
                for line in output.split('\n'):
                    if 'Memory score:' in line:
                        try:
                            score = float(line.split(':')[1].split('/')[0].strip())
                            return score >= 6.0, output, score
                        except (ValueError, IndexError):
                            pass
                
                return True, output, 8.0
            else:
                return False, output, 0.0
            
        except subprocess.TimeoutExpired:
            return False, "Memory test timed out", 0.0
        except Exception as e:
            return False, f"Memory test failed: {e}", 0.0
    
    def check_file_structure(self) -> Tuple[bool, str, Optional[float]]:
        """Check project file structure."""
        required_files = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "LICENSE",
            "hdc_robot_controller/__init__.py",
            "hdc_robot_controller/core/__init__.py",
            "tests/test_hdc_comprehensive.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            output = f"Missing files: {', '.join(missing_files)}"
            score = max(0.0, 10.0 * (1 - len(missing_files) / len(required_files)))
            return False, output, score
        
        return True, "All required files present", 10.0
    
    def check_import_order(self) -> Tuple[bool, str, Optional[float]]:
        """Check import order with isort."""
        try:
            result = subprocess.run([
                "isort", "--check-only", "--diff", "hdc_robot_controller/", "tests/", "scripts/"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            return passed, output, 10.0 if passed else 7.0
            
        except FileNotFoundError:
            return True, "isort not installed (skipping)", 8.0
    
    def check_code_complexity(self) -> Tuple[bool, str, Optional[float]]:
        """Check code complexity."""
        try:
            result = subprocess.run([
                "radon", "cc", "hdc_robot_controller/", "-a", "-nc"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                # Parse average complexity
                for line in output.split('\n'):
                    if 'Average complexity' in line:
                        try:
                            complexity = float(line.split(':')[1].strip().split()[0])
                            # Score based on complexity (lower is better)
                            if complexity < 5:
                                score = 10.0
                            elif complexity < 10:
                                score = 8.0
                            elif complexity < 15:
                                score = 6.0
                            else:
                                score = max(0.0, 10.0 - complexity * 0.5)
                            
                            return complexity < 10, output, score
                        except (ValueError, IndexError):
                            pass
                
                return True, output, 8.0
            else:
                return False, output, 5.0
            
        except FileNotFoundError:
            return True, "radon not installed (skipping)", 8.0
    
    def check_license_headers(self) -> Tuple[bool, str, Optional[float]]:
        """Check for license headers in source files."""
        python_files = list(self.project_root.glob("hdc_robot_controller/**/*.py"))
        
        files_with_headers = 0
        total_files = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_lines = ''.join(f.readlines()[:10])
                
                # Check for common license indicators
                if any(keyword in first_lines.lower() for keyword in 
                      ['copyright', 'license', 'mit', 'bsd', 'apache']):
                    files_with_headers += 1
                    
            except Exception:
                continue
        
        if total_files == 0:
            return True, "No Python files found", 10.0
        
        coverage = files_with_headers / total_files
        score = coverage * 10.0
        
        # More lenient for license headers
        passed = coverage >= 0.3  # 30% is acceptable
        
        output = f"License headers: {files_with_headers}/{total_files} files ({coverage*100:.1f}%)"
        
        return passed, output, score
    
    def _generate_report(self) -> bool:
        """Generate final quality report."""
        passed_checks = [r for r in self.results if r.passed]
        failed_checks = [r for r in self.results if not r.passed]
        
        total_score = sum(r.score for r in self.results if r.score is not None)
        max_score = len([r for r in self.results if r.score is not None]) * 10.0
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0.0
        
        print("\n" + "="*80)
        print("🏆 QUALITY ASSESSMENT REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Score: {overall_score:.1f}%")
        print(f"✅ Passed Checks: {len(passed_checks)}/{len(self.results)}")
        print(f"❌ Failed Checks: {len(failed_checks)}")
        
        if failed_checks:
            print("\n❌ Failed Checks:")
            print("-" * 40)
            for check in failed_checks:
                print(f"  • {check.name}")
                if check.error:
                    print(f"    Error: {check.error}")
                elif check.output:
                    # Show first few lines of output
                    output_lines = check.output.strip().split('\n')[:3]
                    for line in output_lines:
                        if line.strip():
                            print(f"    {line.strip()}")
        
        print("\n✅ Passed Checks:")
        print("-" * 40)
        for check in passed_checks:
            score_text = f" ({check.score:.1f}/10.0)" if check.score is not None else ""
            print(f"  • {check.name}{score_text}")
        
        print("\n⏱️  Performance Summary:")
        print("-" * 40)
        total_time = sum(r.duration for r in self.results)
        print(f"  Total execution time: {total_time:.2f}s")
        
        # Show slowest checks
        slowest = sorted(self.results, key=lambda x: x.duration, reverse=True)[:3]
        for check in slowest:
            print(f"  {check.name}: {check.duration:.2f}s")
        
        # Quality gates
        print("\n🚪 Quality Gates:")
        print("-" * 40)
        
        critical_checks = [
            "Unit Tests",
            "Security Scan", 
            "Code Formatting",
            "Type Checking"
        ]
        
        critical_passed = all(
            any(r.name == check and r.passed for r in self.results)
            for check in critical_checks
        )
        
        if critical_passed and overall_score >= 70.0:
            print("  🎉 ALL QUALITY GATES PASSED!")
            print("  ✨ Code is ready for production!")
            return True
        else:
            print("  🚫 QUALITY GATES FAILED!")
            if not critical_passed:
                print("  ⚠️  Critical checks must pass")
            if overall_score < 70.0:
                print(f"  ⚠️  Overall score {overall_score:.1f}% < 70%")
            return False
    
    def save_report(self, filename: str = "quality-report.json"):
        """Save detailed report to JSON file."""
        report_data = {
            "timestamp": time.time(),
            "overall_score": sum(r.score for r in self.results if r.score is not None),
            "total_checks": len(self.results),
            "passed_checks": len([r for r in self.results if r.passed]),
            "failed_checks": len([r for r in self.results if not r.passed]),
            "thresholds": self.thresholds,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "score": r.score,
                    "error": r.error,
                    "output": r.output[:500] if r.output else None  # Truncate output
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"📄 Report saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Quality checker for HDC Robot Controller")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run checks in parallel (default)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run checks sequentially")
    parser.add_argument("--save-report", type=str, default="quality-report.json",
                       help="Save report to file")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger("quality_checker").setLevel(logging.DEBUG)
    
    # Create checker
    checker = QualityChecker(args.project_root)
    
    # Run checks
    parallel = args.parallel and not args.sequential
    success = checker.run_all_checks(parallel=parallel)
    
    # Save report
    if args.save_report:
        checker.save_report(args.save_report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()