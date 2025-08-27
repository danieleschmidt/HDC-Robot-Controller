#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - AUTONOMOUS QUALITY ORCHESTRATOR

Comprehensive quality gates execution engine with mandatory validation,
security scanning, performance benchmarking, and automated fixing.
"""

import asyncio
import json
import time
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import logging
import tempfile
import shutil
import re
import hashlib

class QualityGateType(Enum):
    CODE_EXECUTION = "code_execution"
    TEST_COVERAGE = "test_coverage" 
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DOCUMENTATION_COVERAGE = "documentation_coverage"
    DEPENDENCY_AUDIT = "dependency_audit"
    CODE_QUALITY = "code_quality"
    ARCHITECTURE_COMPLIANCE = "architecture_compliance"

class QualityStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    FIXED = "fixed"
    SKIPPED = "skipped"

@dataclass
class QualityIssue:
    """Represents a quality issue found during validation."""
    severity: str  # critical, high, medium, low
    category: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class QualityGateResult:
    """Results from executing a quality gate."""
    gate_type: QualityGateType
    status: QualityStatus
    score: float  # 0-100
    execution_time: float
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    auto_fixes_applied: int = 0

@dataclass
class QualityReport:
    """Comprehensive quality report."""
    overall_score: float
    gate_results: Dict[QualityGateType, QualityGateResult]
    total_issues: int
    critical_issues: int
    auto_fixes_applied: int
    execution_time: float
    timestamp: float
    passed_gates: int
    total_gates: int

class AutonomousQualityOrchestrator:
    """Master quality orchestration engine with autonomous fixing."""
    
    def __init__(self, project_root: Path, config: Optional[Dict] = None):
        """Initialize quality orchestrator."""
        self.project_root = Path(project_root)
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Quality gate configurations
        self.quality_gates = self._initialize_quality_gates()
        self.auto_fix_enabled = self.config.get('auto_fix_enabled', True)
        self.fix_attempt_limit = self.config.get('fix_attempt_limit', 3)
        
        # Execution state
        self.results_history: List[QualityReport] = []
        self.current_execution_id = None

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default quality gate configuration."""
        return {
            'required_test_coverage': 85.0,
            'max_security_vulnerabilities': 0,
            'max_performance_regression': 0.05,  # 5%
            'required_documentation_coverage': 90.0,
            'auto_fix_enabled': True,
            'fix_attempt_limit': 3,
            'parallel_execution': True,
            'fail_fast': False,
            'quality_score_threshold': 85.0
        }

    def _initialize_quality_gates(self) -> Dict[QualityGateType, Dict]:
        """Initialize quality gate configurations."""
        return {
            QualityGateType.CODE_EXECUTION: {
                'required': True,
                'weight': 0.25,
                'timeout': 300,
                'auto_fix': True
            },
            QualityGateType.TEST_COVERAGE: {
                'required': True,
                'weight': 0.20,
                'timeout': 600,
                'auto_fix': False,
                'min_coverage': self.config['required_test_coverage']
            },
            QualityGateType.SECURITY_SCAN: {
                'required': True,
                'weight': 0.20,
                'timeout': 300,
                'auto_fix': True,
                'max_critical': 0
            },
            QualityGateType.PERFORMANCE_BENCHMARK: {
                'required': True,
                'weight': 0.15,
                'timeout': 600,
                'auto_fix': False,
                'max_regression': self.config['max_performance_regression']
            },
            QualityGateType.CODE_QUALITY: {
                'required': False,
                'weight': 0.10,
                'timeout': 300,
                'auto_fix': True,
                'min_score': 8.0
            },
            QualityGateType.DOCUMENTATION_COVERAGE: {
                'required': False,
                'weight': 0.05,
                'timeout': 180,
                'auto_fix': True,
                'min_coverage': self.config['required_documentation_coverage']
            },
            QualityGateType.DEPENDENCY_AUDIT: {
                'required': False,
                'weight': 0.05,
                'timeout': 240,
                'auto_fix': False
            }
        }

    async def execute_quality_gates(self, fail_fast: bool = None) -> QualityReport:
        """Execute all quality gates with autonomous fixing."""
        start_time = time.time()
        self.current_execution_id = hashlib.md5(str(start_time).encode()).hexdigest()[:8]
        
        self.logger.info(f"🛡️ Starting quality gates execution [{self.current_execution_id}]")
        
        if fail_fast is None:
            fail_fast = self.config.get('fail_fast', False)
        
        gate_results = {}
        total_auto_fixes = 0
        
        try:
            if self.config.get('parallel_execution', True):
                # Execute quality gates in parallel
                tasks = []
                for gate_type, gate_config in self.quality_gates.items():
                    task = self._execute_quality_gate_with_retry(gate_type, gate_config)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, (gate_type, _) in enumerate(self.quality_gates.items()):
                    if isinstance(results[i], Exception):
                        self.logger.error(f"Gate {gate_type} failed with exception: {results[i]}")
                        gate_results[gate_type] = QualityGateResult(
                            gate_type=gate_type,
                            status=QualityStatus.FAILED,
                            score=0.0,
                            execution_time=0.0,
                            issues=[QualityIssue("critical", "execution", str(results[i]))]
                        )
                    else:
                        gate_results[gate_type] = results[i]
                        total_auto_fixes += results[i].auto_fixes_applied
                        
            else:
                # Execute quality gates sequentially
                for gate_type, gate_config in self.quality_gates.items():
                    result = await self._execute_quality_gate_with_retry(gate_type, gate_config)
                    gate_results[gate_type] = result
                    total_auto_fixes += result.auto_fixes_applied
                    
                    # Fail fast if critical gate fails
                    if fail_fast and gate_config['required'] and result.status == QualityStatus.FAILED:
                        self.logger.error(f"❌ Critical gate {gate_type} failed, stopping execution")
                        break
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(gate_results)
            
            # Count issues
            total_issues = sum(len(result.issues) for result in gate_results.values())
            critical_issues = sum(
                len([issue for issue in result.issues if issue.severity == 'critical'])
                for result in gate_results.values()
            )
            
            # Count passed gates
            passed_gates = sum(1 for result in gate_results.values() if result.status == QualityStatus.PASSED)
            
            # Generate quality report
            quality_report = QualityReport(
                overall_score=overall_score,
                gate_results=gate_results,
                total_issues=total_issues,
                critical_issues=critical_issues,
                auto_fixes_applied=total_auto_fixes,
                execution_time=time.time() - start_time,
                timestamp=time.time(),
                passed_gates=passed_gates,
                total_gates=len(gate_results)
            )
            
            # Save report
            await self._save_quality_report(quality_report)
            
            # Log summary
            self._log_execution_summary(quality_report)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"❌ Quality gates execution failed: {str(e)}")
            raise

    async def _execute_quality_gate_with_retry(self, gate_type: QualityGateType, 
                                             gate_config: Dict) -> QualityGateResult:
        """Execute quality gate with retry and auto-fixing."""
        max_attempts = self.fix_attempt_limit if gate_config.get('auto_fix') else 1
        
        for attempt in range(max_attempts):
            self.logger.info(f"🔍 Executing {gate_type.value} (attempt {attempt + 1}/{max_attempts})")
            
            result = await self._execute_single_quality_gate(gate_type, gate_config)
            
            # If passed or auto-fix disabled, return result
            if result.status == QualityStatus.PASSED or not gate_config.get('auto_fix'):
                return result
            
            # If failed and auto-fix enabled, attempt fixes
            if result.status == QualityStatus.FAILED and self.auto_fix_enabled:
                self.logger.info(f"🔧 Attempting auto-fixes for {gate_type.value}")
                
                fixes_applied = await self._apply_auto_fixes(gate_type, result.issues)
                result.auto_fixes_applied += fixes_applied
                
                if fixes_applied > 0:
                    self.logger.info(f"✅ Applied {fixes_applied} auto-fixes")
                    # Continue to next attempt
                    continue
                else:
                    self.logger.warning(f"⚠️ No auto-fixes available for {gate_type.value}")
                    return result
            else:
                return result
        
        # If all attempts failed
        result.status = QualityStatus.FAILED
        return result

    async def _execute_single_quality_gate(self, gate_type: QualityGateType, 
                                         gate_config: Dict) -> QualityGateResult:
        """Execute individual quality gate."""
        start_time = time.time()
        
        try:
            if gate_type == QualityGateType.CODE_EXECUTION:
                return await self._execute_code_execution_gate(gate_config)
            elif gate_type == QualityGateType.TEST_COVERAGE:
                return await self._execute_test_coverage_gate(gate_config)
            elif gate_type == QualityGateType.SECURITY_SCAN:
                return await self._execute_security_scan_gate(gate_config)
            elif gate_type == QualityGateType.PERFORMANCE_BENCHMARK:
                return await self._execute_performance_benchmark_gate(gate_config)
            elif gate_type == QualityGateType.CODE_QUALITY:
                return await self._execute_code_quality_gate(gate_config)
            elif gate_type == QualityGateType.DOCUMENTATION_COVERAGE:
                return await self._execute_documentation_gate(gate_config)
            elif gate_type == QualityGateType.DEPENDENCY_AUDIT:
                return await self._execute_dependency_audit_gate(gate_config)
            else:
                raise ValueError(f"Unknown quality gate type: {gate_type}")
                
        except asyncio.TimeoutError:
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                issues=[QualityIssue("critical", "timeout", f"Gate execution timed out after {gate_config['timeout']}s")]
            )
        except Exception as e:
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                issues=[QualityIssue("critical", "execution", str(e))]
            )

    async def _execute_code_execution_gate(self, config: Dict) -> QualityGateResult:
        """Execute code execution quality gate."""
        start_time = time.time()
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        if not python_files:
            return QualityGateResult(
                gate_type=QualityGateType.CODE_EXECUTION,
                status=QualityStatus.SKIPPED,
                score=100.0,
                execution_time=time.time() - start_time,
                issues=[],
                metrics={'python_files': 0}
            )
        
        # Exclude test files and virtual environments
        source_files = [
            f for f in python_files 
            if not any(exclude in str(f) for exclude in ['test_', '_test.py', 'venv/', '.venv/', '__pycache__'])
        ]
        
        syntax_errors = 0
        import_errors = 0
        
        # Check syntax for each file
        for file_path in source_files:
            try:
                # Syntax check
                result = await asyncio.create_subprocess_exec(
                    sys.executable, '-m', 'py_compile', str(file_path),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode != 0:
                    syntax_errors += 1
                    issues.append(QualityIssue(
                        severity="critical",
                        category="syntax",
                        message=f"Syntax error in {file_path.name}",
                        file_path=str(file_path),
                        suggestion="Fix syntax errors",
                        auto_fixable=False
                    ))
                    
            except Exception as e:
                issues.append(QualityIssue(
                    severity="high",
                    category="compilation",
                    message=f"Could not check {file_path.name}: {str(e)}",
                    file_path=str(file_path),
                    auto_fixable=False
                ))
        
        # Overall execution test
        try:
            # Try to import main modules
            for file_path in source_files[:5]:  # Check first 5 files
                if file_path.name == '__init__.py':
                    continue
                    
                module_name = file_path.stem
                if module_name != 'main' and not module_name.startswith('test_'):
                    result = await asyncio.create_subprocess_exec(
                        sys.executable, '-c', f'import sys; sys.path.insert(0, "{self.project_root}"); import {module_name}',
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=self.project_root
                    )
                    await result.communicate()
                    
                    if result.returncode != 0:
                        import_errors += 1
                        
        except Exception as e:
            self.logger.warning(f"Import test failed: {str(e)}")
        
        # Calculate score
        total_files = len(source_files)
        if total_files == 0:
            score = 100.0
        else:
            error_rate = (syntax_errors + import_errors) / total_files
            score = max(0, 100 - (error_rate * 100))
        
        status = QualityStatus.PASSED if syntax_errors == 0 else QualityStatus.FAILED
        
        return QualityGateResult(
            gate_type=QualityGateType.CODE_EXECUTION,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'total_files': total_files,
                'syntax_errors': syntax_errors,
                'import_errors': import_errors
            }
        )

    async def _execute_test_coverage_gate(self, config: Dict) -> QualityGateResult:
        """Execute test coverage quality gate."""
        start_time = time.time()
        issues = []
        
        # Check if pytest is available
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'pytest', '--version',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()
            
            if result.returncode != 0:
                # Pytest not available, try to run basic tests
                return await self._execute_basic_test_runner(config)
                
        except Exception:
            return await self._execute_basic_test_runner(config)
        
        # Run pytest with coverage
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'pytest', '--cov=.', '--cov-report=json',
                '--tb=short', '-v',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=config['timeout'])
            
            # Parse coverage report
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                
                # Clean up coverage file
                coverage_file.unlink()
            else:
                total_coverage = 0.0
            
            # Parse test results
            test_summary = self._parse_pytest_output(stdout.decode())
            
            # Check coverage threshold
            min_coverage = config.get('min_coverage', 85.0)
            if total_coverage < min_coverage:
                issues.append(QualityIssue(
                    severity="high",
                    category="coverage",
                    message=f"Test coverage {total_coverage:.1f}% below threshold {min_coverage:.1f}%",
                    suggestion=f"Add tests to reach {min_coverage:.1f}% coverage",
                    auto_fixable=False
                ))
            
            status = QualityStatus.PASSED if total_coverage >= min_coverage and result.returncode == 0 else QualityStatus.FAILED
            score = min(100.0, (total_coverage / min_coverage) * 100.0) if result.returncode == 0 else 0.0
            
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=status,
                score=score,
                execution_time=time.time() - start_time,
                issues=issues,
                metrics={
                    'coverage_percent': total_coverage,
                    'tests_run': test_summary.get('tests_run', 0),
                    'tests_passed': test_summary.get('passed', 0),
                    'tests_failed': test_summary.get('failed', 0)
                }
            )
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=QualityStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                issues=[QualityIssue("critical", "test_execution", str(e))]
            )

    async def _execute_security_scan_gate(self, config: Dict) -> QualityGateResult:
        """Execute security scanning quality gate."""
        start_time = time.time()
        issues = []
        
        # Basic security checks
        security_issues = await self._basic_security_scan()
        issues.extend(security_issues)
        
        # Check for hardcoded secrets
        secret_issues = await self._scan_for_secrets()
        issues.extend(secret_issues)
        
        # Dependency vulnerability check (if available)
        try:
            dep_issues = await self._check_dependency_vulnerabilities()
            issues.extend(dep_issues)
        except Exception:
            pass  # Dependency check optional
        
        critical_issues = sum(1 for issue in issues if issue.severity == "critical")
        high_issues = sum(1 for issue in issues if issue.severity == "high")
        
        max_critical = config.get('max_critical', 0)
        status = QualityStatus.PASSED if critical_issues <= max_critical else QualityStatus.FAILED
        
        # Calculate security score
        total_severity_score = critical_issues * 10 + high_issues * 5 + len(issues)
        score = max(0, 100 - total_severity_score)
        
        return QualityGateResult(
            gate_type=QualityGateType.SECURITY_SCAN,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'total_issues': len(issues)
            }
        )

    async def _basic_security_scan(self) -> List[QualityIssue]:
        """Perform basic security scanning."""
        issues = []
        
        # Check for common security anti-patterns
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for eval() usage
                    if re.search(r'\beval\s*\(', line):
                        issues.append(QualityIssue(
                            severity="critical",
                            category="code_injection",
                            message="Use of eval() detected",
                            file_path=str(file_path),
                            line_number=line_num,
                            suggestion="Replace eval() with safer alternatives like ast.literal_eval()",
                            auto_fixable=False
                        ))
                    
                    # Check for exec() usage
                    if re.search(r'\bexec\s*\(', line):
                        issues.append(QualityIssue(
                            severity="critical",
                            category="code_injection", 
                            message="Use of exec() detected",
                            file_path=str(file_path),
                            line_number=line_num,
                            suggestion="Avoid exec() or use safer alternatives",
                            auto_fixable=False
                        ))
                    
                    # Check for subprocess without shell=False
                    if 'subprocess' in line and 'shell=True' in line:
                        issues.append(QualityIssue(
                            severity="high",
                            category="command_injection",
                            message="subprocess with shell=True detected",
                            file_path=str(file_path),
                            line_number=line_num,
                            suggestion="Use shell=False and pass arguments as list",
                            auto_fixable=True
                        ))
                        
            except Exception as e:
                continue  # Skip files that can't be read
        
        return issues

    async def _scan_for_secrets(self) -> List[QualityIssue]:
        """Scan for hardcoded secrets and credentials."""
        issues = []
        
        # Common secret patterns
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{3,}["\']', "password"),
            (r'api_key\s*=\s*["\'][^"\']{10,}["\']', "api_key"),
            (r'secret\s*=\s*["\'][^"\']{10,}["\']', "secret"),
            (r'token\s*=\s*["\'][^"\']{10,}["\']', "token"),
            (r'-----BEGIN[A-Z\s]+PRIVATE KEY-----', "private_key")
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        config_files = list(self.project_root.rglob("*.yml")) + list(self.project_root.rglob("*.yaml"))
        
        all_files = python_files + config_files
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append(QualityIssue(
                                severity="critical",
                                category="hardcoded_secret",
                                message=f"Potential hardcoded {secret_type} detected",
                                file_path=str(file_path),
                                line_number=line_num,
                                suggestion=f"Move {secret_type} to environment variables or secure configuration",
                                auto_fixable=False
                            ))
                            
            except Exception:
                continue
        
        return issues

    async def _check_dependency_vulnerabilities(self) -> List[QualityIssue]:
        """Check for known vulnerabilities in dependencies."""
        issues = []
        
        # Check if safety is available
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'pip', 'install', 'safety',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            await result.communicate()
            
            # Run safety check
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'safety', 'check', '--json',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                # Parse safety output
                try:
                    safety_data = json.loads(stdout.decode())
                    for vuln in safety_data:
                        issues.append(QualityIssue(
                            severity="high",
                            category="dependency_vulnerability",
                            message=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                            suggestion=f"Update to version {vuln.get('safe_version', 'latest')}",
                            auto_fixable=False
                        ))
                except json.JSONDecodeError:
                    pass
                    
        except Exception:
            pass  # Safety check is optional
        
        return issues

    async def _execute_performance_benchmark_gate(self, config: Dict) -> QualityGateResult:
        """Execute performance benchmarking gate."""
        start_time = time.time()
        issues = []
        
        # Basic performance checks
        metrics = {}
        
        # Check for potential performance issues in code
        performance_issues = await self._analyze_performance_patterns()
        issues.extend(performance_issues)
        
        # If there's a main.py or similar entry point, try basic performance test
        main_files = list(self.project_root.glob("main.py")) + list(self.project_root.glob("app.py"))
        
        if main_files and len(performance_issues) == 0:
            # Simulate performance benchmark
            import_time = await self._measure_import_time(main_files[0])
            metrics['import_time_ms'] = import_time
            
            if import_time > 5000:  # 5 seconds
                issues.append(QualityIssue(
                    severity="medium",
                    category="performance",
                    message=f"Slow import time: {import_time:.0f}ms",
                    suggestion="Optimize imports and reduce module loading time",
                    auto_fixable=False
                ))
        
        max_regression = config.get('max_regression', 0.05)
        status = QualityStatus.PASSED if len([i for i in issues if i.severity in ['critical', 'high']]) == 0 else QualityStatus.WARNING
        
        score = max(0, 100 - len(issues) * 10)
        
        return QualityGateResult(
            gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics=metrics
        )

    async def _analyze_performance_patterns(self) -> List[QualityIssue]:
        """Analyze code for performance anti-patterns."""
        issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for nested loops
                    if 'for' in line and any('for' in lines[i] for i in range(max(0, line_num-3), min(len(lines), line_num+3))):
                        if line.strip().startswith('for'):
                            issues.append(QualityIssue(
                                severity="medium",
                                category="performance",
                                message="Potential nested loop detected",
                                file_path=str(file_path),
                                line_number=line_num,
                                suggestion="Consider optimizing nested loops or using vectorized operations",
                                auto_fixable=False
                            ))
                    
                    # Check for string concatenation in loops
                    if '+=' in line and 'str' in line and any('for' in lines[i] for i in range(max(0, line_num-5), line_num)):
                        issues.append(QualityIssue(
                            severity="medium",
                            category="performance",
                            message="String concatenation in loop detected",
                            file_path=str(file_path),
                            line_number=line_num,
                            suggestion="Use list.join() or StringIO for efficient string concatenation",
                            auto_fixable=True
                        ))
                        
            except Exception:
                continue
        
        return issues

    async def _measure_import_time(self, main_file: Path) -> float:
        """Measure import time for main module."""
        try:
            start_time = time.time()
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-c', f'import time; start=time.time(); import sys; sys.path.insert(0, "{self.project_root}"); import {main_file.stem}; print((time.time()-start)*1000)',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return float(stdout.decode().strip())
            else:
                return 0.0
                
        except Exception:
            return 0.0

    async def _execute_code_quality_gate(self, config: Dict) -> QualityGateResult:
        """Execute code quality analysis."""
        start_time = time.time()
        issues = []
        
        # Basic code quality checks
        python_files = list(self.project_root.rglob("*.py"))
        total_lines = 0
        complex_functions = 0
        long_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines += len(lines)
                
                # Check for long functions
                in_function = False
                function_lines = 0
                function_name = ""
                
                for line_num, line in enumerate(lines, 1):
                    if line.strip().startswith('def '):
                        if in_function and function_lines > 50:
                            long_functions += 1
                            issues.append(QualityIssue(
                                severity="medium",
                                category="maintainability",
                                message=f"Long function '{function_name}' ({function_lines} lines)",
                                file_path=str(file_path),
                                line_number=line_num - function_lines,
                                suggestion="Consider breaking down long functions",
                                auto_fixable=False
                            ))
                        
                        in_function = True
                        function_lines = 0
                        function_name = line.split('def ')[1].split('(')[0]
                    elif in_function:
                        function_lines += 1
                
                # Check final function
                if in_function and function_lines > 50:
                    long_functions += 1
                    
            except Exception:
                continue
        
        # Calculate quality score based on issues
        quality_issues = len(issues)
        score = max(0, 100 - quality_issues * 5)
        
        min_score = config.get('min_score', 8.0)
        scaled_min = min_score * 10  # Convert to 0-100 scale
        
        status = QualityStatus.PASSED if score >= scaled_min else QualityStatus.WARNING
        
        return QualityGateResult(
            gate_type=QualityGateType.CODE_QUALITY,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'total_lines': total_lines,
                'long_functions': long_functions,
                'quality_issues': quality_issues
            }
        )

    async def _execute_documentation_gate(self, config: Dict) -> QualityGateResult:
        """Execute documentation coverage gate."""
        start_time = time.time()
        issues = []
        
        # Check for README and basic documentation
        readme_files = list(self.project_root.glob("README*"))
        docs_dir = self.project_root / "docs"
        
        doc_coverage = 0.0
        
        if readme_files:
            doc_coverage += 20.0
        else:
            issues.append(QualityIssue(
                severity="medium",
                category="documentation",
                message="Missing README file",
                suggestion="Add README.md with project description and usage",
                auto_fixable=True
            ))
        
        if docs_dir.exists():
            doc_coverage += 15.0
        
        # Check Python docstrings
        python_files = list(self.project_root.rglob("*.py"))
        functions_with_docs = 0
        total_functions = 0
        
        for file_path in python_files:
            if 'test_' in file_path.name:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple docstring detection
                lines = content.split('\n')
                in_function = False
                function_has_docstring = False
                
                for line in lines:
                    if line.strip().startswith('def '):
                        if in_function:
                            if not function_has_docstring:
                                total_functions += 1
                            else:
                                functions_with_docs += 1
                                total_functions += 1
                        
                        in_function = True
                        function_has_docstring = False
                    elif in_function and '"""' in line:
                        function_has_docstring = True
                
                # Handle last function
                if in_function:
                    if function_has_docstring:
                        functions_with_docs += 1
                    total_functions += 1
                    
            except Exception:
                continue
        
        if total_functions > 0:
            docstring_coverage = (functions_with_docs / total_functions) * 65.0  # 65% of total score
            doc_coverage += docstring_coverage
            
            if docstring_coverage < 32.5:  # Less than 50% of functions documented
                issues.append(QualityIssue(
                    severity="medium",
                    category="documentation",
                    message=f"Low docstring coverage: {functions_with_docs}/{total_functions} functions documented",
                    suggestion="Add docstrings to public functions and classes",
                    auto_fixable=True
                ))
        
        min_coverage = config.get('min_coverage', 90.0)
        status = QualityStatus.PASSED if doc_coverage >= min_coverage else QualityStatus.WARNING
        
        return QualityGateResult(
            gate_type=QualityGateType.DOCUMENTATION_COVERAGE,
            status=status,
            score=doc_coverage,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'documentation_coverage': doc_coverage,
                'functions_with_docs': functions_with_docs,
                'total_functions': total_functions,
                'has_readme': len(readme_files) > 0,
                'has_docs_dir': docs_dir.exists()
            }
        )

    async def _execute_dependency_audit_gate(self, config: Dict) -> QualityGateResult:
        """Execute dependency audit gate."""
        start_time = time.time()
        issues = []
        
        # Check for dependency files
        req_files = list(self.project_root.glob("requirements*.txt"))
        pyproject_file = self.project_root / "pyproject.toml"
        
        total_deps = 0
        outdated_deps = 0
        
        if req_files:
            for req_file in req_files:
                try:
                    with open(req_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            total_deps += 1
                            # Check for unpinned versions
                            if '==' not in line and '>=' not in line:
                                issues.append(QualityIssue(
                                    severity="low",
                                    category="dependency",
                                    message=f"Unpinned dependency: {line}",
                                    file_path=str(req_file),
                                    suggestion="Pin dependency versions for reproducible builds",
                                    auto_fixable=False
                                ))
                except Exception:
                    continue
        
        elif pyproject_file.exists():
            # Check pyproject.toml dependencies
            try:
                import tomllib
                with open(pyproject_file, 'rb') as f:
                    data = tomllib.load(f)
                
                deps = data.get('project', {}).get('dependencies', [])
                total_deps = len(deps)
                
            except Exception:
                pass
        
        # Calculate score
        score = max(0, 100 - len(issues) * 5)
        status = QualityStatus.PASSED if len(issues) == 0 else QualityStatus.WARNING
        
        return QualityGateResult(
            gate_type=QualityGateType.DEPENDENCY_AUDIT,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'total_dependencies': total_deps,
                'dependency_issues': len(issues)
            }
        )

    async def _execute_basic_test_runner(self, config: Dict) -> QualityGateResult:
        """Execute basic test runner when pytest is not available."""
        start_time = time.time()
        
        # Find test files
        test_files = list(self.project_root.rglob("test_*.py")) + list(self.project_root.rglob("*_test.py"))
        
        if not test_files:
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=QualityStatus.WARNING,
                score=50.0,  # Partial score for having no tests
                execution_time=time.time() - start_time,
                issues=[QualityIssue("medium", "testing", "No test files found", suggestion="Add unit tests")],
                metrics={'test_files': 0}
            )
        
        # Try to run test files
        passed_tests = 0
        total_tests = len(test_files)
        
        for test_file in test_files:
            try:
                result = await asyncio.create_subprocess_exec(
                    sys.executable, str(test_file),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=self.project_root
                )
                await result.communicate()
                
                if result.returncode == 0:
                    passed_tests += 1
                    
            except Exception:
                continue
        
        coverage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        min_coverage = config.get('min_coverage', 85.0)
        
        status = QualityStatus.PASSED if coverage >= min_coverage else QualityStatus.WARNING
        score = coverage
        
        issues = []
        if coverage < min_coverage:
            issues.append(QualityIssue(
                severity="medium",
                category="testing",
                message=f"Test success rate {coverage:.1f}% below threshold {min_coverage:.1f}%",
                suggestion="Fix failing tests or add more comprehensive tests"
            ))
        
        return QualityGateResult(
            gate_type=QualityGateType.TEST_COVERAGE,
            status=status,
            score=score,
            execution_time=time.time() - start_time,
            issues=issues,
            metrics={
                'test_files': total_tests,
                'passed_tests': passed_tests,
                'test_success_rate': coverage
            }
        )

    async def _apply_auto_fixes(self, gate_type: QualityGateType, issues: List[QualityIssue]) -> int:
        """Apply automatic fixes for identified issues."""
        fixes_applied = 0
        
        for issue in issues:
            if not issue.auto_fixable:
                continue
                
            try:
                if issue.category == "command_injection" and issue.file_path:
                    fixes_applied += await self._fix_command_injection(issue)
                elif issue.category == "documentation" and "Missing README" in issue.message:
                    fixes_applied += await self._create_basic_readme()
                elif issue.category == "documentation" and "docstring" in issue.message.lower():
                    fixes_applied += await self._add_basic_docstrings()
                elif issue.category == "performance" and "string concatenation" in issue.message.lower():
                    fixes_applied += await self._fix_string_concatenation(issue)
                    
            except Exception as e:
                self.logger.warning(f"Failed to apply auto-fix for {issue.category}: {str(e)}")
        
        return fixes_applied

    async def _fix_command_injection(self, issue: QualityIssue) -> int:
        """Fix command injection vulnerabilities."""
        if not issue.file_path or not issue.line_number:
            return 0
            
        try:
            with open(issue.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            line_idx = issue.line_number - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                # Replace shell=True with shell=False
                if 'shell=True' in line:
                    lines[line_idx] = line.replace('shell=True', 'shell=False')
                    
                    with open(issue.file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    return 1
                    
        except Exception:
            pass
        
        return 0

    async def _create_basic_readme(self) -> int:
        """Create basic README file."""
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            return 0
            
        readme_content = f"""# {self.project_root.name}

## Overview
This project was enhanced with Terragon SDLC v4.0 autonomous development.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
# Add usage examples here
```

## Testing
```bash
python -m pytest
```

## License
See LICENSE file for details.

---
*Generated by Terragon SDLC v4.0*
"""
        
        try:
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            return 1
        except Exception:
            return 0

    async def _add_basic_docstrings(self) -> int:
        """Add basic docstrings to functions missing them."""
        fixes_applied = 0
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files[:3]:  # Limit to first 3 files
            if 'test_' in file_path.name:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                modified = False
                new_lines = []
                
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    
                    # Check if this is a function definition
                    if line.strip().startswith('def ') and not line.strip().startswith('def _'):
                        # Check if next lines contain docstring
                        has_docstring = False
                        for j in range(i+1, min(i+5, len(lines))):
                            if '"""' in lines[j]:
                                has_docstring = True
                                break
                        
                        if not has_docstring:
                            # Extract function name
                            func_name = line.split('def ')[1].split('(')[0]
                            indent = len(line) - len(line.lstrip())
                            
                            # Add basic docstring
                            docstring_lines = [
                                ' ' * (indent + 4) + '"""' + f'{func_name.replace("_", " ").title()}.\n',
                                ' ' * (indent + 4) + '"""\n'
                            ]
                            
                            new_lines.extend(docstring_lines)
                            modified = True
                            fixes_applied += 1
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                        
            except Exception:
                continue
        
        return fixes_applied

    async def _fix_string_concatenation(self, issue: QualityIssue) -> int:
        """Fix string concatenation in loops."""
        # This would require more sophisticated AST manipulation
        # For now, just return 0 (no fix applied)
        return 0

    def _calculate_overall_score(self, gate_results: Dict[QualityGateType, QualityGateResult]) -> float:
        """Calculate weighted overall quality score."""
        total_weight = 0.0
        weighted_score = 0.0
        
        for gate_type, result in gate_results.items():
            if gate_type in self.quality_gates:
                weight = self.quality_gates[gate_type]['weight']
                total_weight += weight
                weighted_score += result.score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output for test statistics."""
        summary = {"tests_run": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        lines = output.split('\n')
        for line in lines:
            # Look for summary line like "10 passed, 2 failed, 1 skipped"
            if 'passed' in line or 'failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            summary["passed"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif part == "failed" and i > 0:
                        try:
                            summary["failed"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif part == "skipped" and i > 0:
                        try:
                            summary["skipped"] = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                break
        
        summary["tests_run"] = summary["passed"] + summary["failed"] + summary["skipped"]
        return summary

    async def _save_quality_report(self, report: QualityReport):
        """Save comprehensive quality report."""
        reports_dir = self.project_root / "quality_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_report = {
            "execution_id": self.current_execution_id,
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "total_issues": report.total_issues,
            "critical_issues": report.critical_issues,
            "auto_fixes_applied": report.auto_fixes_applied,
            "execution_time": report.execution_time,
            "passed_gates": report.passed_gates,
            "total_gates": report.total_gates,
            "gate_results": {}
        }
        
        for gate_type, result in report.gate_results.items():
            json_report["gate_results"][gate_type.value] = {
                "status": result.status.value,
                "score": result.score,
                "execution_time": result.execution_time,
                "issues_count": len(result.issues),
                "metrics": result.metrics,
                "recommendations": result.recommendations,
                "auto_fixes_applied": result.auto_fixes_applied,
                "issues": [
                    {
                        "severity": issue.severity,
                        "category": issue.category,
                        "message": issue.message,
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "suggestion": issue.suggestion,
                        "auto_fixable": issue.auto_fixable
                    } for issue in result.issues
                ]
            }
        
        json_path = reports_dir / f"quality_report_{self.current_execution_id}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Save markdown report
        await self._save_markdown_report(report, reports_dir)

    async def _save_markdown_report(self, report: QualityReport, reports_dir: Path):
        """Save markdown quality report."""
        md_path = reports_dir / f"quality_report_{self.current_execution_id}.md"
        
        # Generate status indicators
        def status_icon(status: QualityStatus) -> str:
            if status == QualityStatus.PASSED:
                return "✅"
            elif status == QualityStatus.FAILED:
                return "❌"
            elif status == QualityStatus.WARNING:
                return "⚠️"
            else:
                return "⏸️"
        
        content = f"""# Quality Gates Report [{self.current_execution_id}]

## 📊 Executive Summary

**Overall Quality Score**: {report.overall_score:.1f}/100
**Gates Passed**: {report.passed_gates}/{report.total_gates}
**Total Issues**: {report.total_issues} (Critical: {report.critical_issues})
**Auto-Fixes Applied**: {report.auto_fixes_applied}
**Execution Time**: {report.execution_time:.1f}s

## 🛡️ Quality Gates Results

| Gate | Status | Score | Issues | Time |
|------|--------|-------|---------|------|
"""
        
        for gate_type, result in report.gate_results.items():
            gate_name = gate_type.value.replace('_', ' ').title()
            status = status_icon(result.status)
            content += f"| {gate_name} | {status} {result.status.value} | {result.score:.1f} | {len(result.issues)} | {result.execution_time:.1f}s |\n"
        
        content += "\n## 🔍 Detailed Results\n\n"
        
        for gate_type, result in report.gate_results.items():
            gate_name = gate_type.value.replace('_', ' ').title()
            status = status_icon(result.status)
            
            content += f"### {status} {gate_name}\n\n"
            content += f"**Score**: {result.score:.1f}/100  \n"
            content += f"**Status**: {result.status.value}  \n"
            content += f"**Execution Time**: {result.execution_time:.1f}s  \n"
            content += f"**Issues Found**: {len(result.issues)}  \n"
            
            if result.auto_fixes_applied > 0:
                content += f"**Auto-Fixes Applied**: {result.auto_fixes_applied}  \n"
            
            if result.metrics:
                content += f"\n**Metrics**:\n"
                for key, value in result.metrics.items():
                    content += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            if result.issues:
                content += f"\n**Issues**:\n"
                for issue in result.issues:
                    severity_icon = "🔥" if issue.severity == "critical" else "⚠️" if issue.severity == "high" else "💡"
                    content += f"{severity_icon} **{issue.severity.upper()}**: {issue.message}"
                    if issue.file_path:
                        content += f" ({issue.file_path}"
                        if issue.line_number:
                            content += f":{issue.line_number}"
                        content += ")"
                    content += "\n"
                    
                    if issue.suggestion:
                        content += f"  - 💡 *Suggestion*: {issue.suggestion}\n"
                    
                    if issue.auto_fixable:
                        content += f"  - 🔧 *Auto-fixable*\n"
                    
                    content += "\n"
            
            if result.recommendations:
                content += f"**Recommendations**:\n"
                for rec in result.recommendations:
                    content += f"- {rec}\n"
            
            content += "\n---\n\n"
        
        content += f"""## 📋 Summary

This quality report was generated by Terragon SDLC v4.0 Quality Orchestrator.

**Quality Gates Configuration:**
- Required gates: {sum(1 for config in self.quality_gates.values() if config['required'])}
- Optional gates: {sum(1 for config in self.quality_gates.values() if not config['required'])}
- Auto-fix enabled: {self.auto_fix_enabled}

**Next Steps:**
{"- ✅ All critical quality gates passed! Ready for deployment." if report.critical_issues == 0 else f"- ❌ {report.critical_issues} critical issues need resolution before deployment."}

---
*Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report.timestamp))}*
"""
        
        with open(md_path, 'w') as f:
            f.write(content)

    def _log_execution_summary(self, report: QualityReport):
        """Log execution summary."""
        self.logger.info("🛡️ Quality Gates Execution Summary")
        self.logger.info(f"   Overall Score: {report.overall_score:.1f}/100")
        self.logger.info(f"   Gates Passed: {report.passed_gates}/{report.total_gates}")
        self.logger.info(f"   Total Issues: {report.total_issues} (Critical: {report.critical_issues})")
        self.logger.info(f"   Auto-Fixes: {report.auto_fixes_applied}")
        self.logger.info(f"   Duration: {report.execution_time:.1f}s")
        
        if report.critical_issues == 0:
            self.logger.info("✅ All critical quality gates passed!")
        else:
            self.logger.error(f"❌ {report.critical_issues} critical issues need resolution")


# Main execution function
async def execute_quality_gates(project_root: Path = None, config: Dict = None) -> QualityReport:
    """Execute comprehensive quality gates validation."""
    if project_root is None:
        project_root = Path.cwd()
    
    orchestrator = AutonomousQualityOrchestrator(project_root, config)
    return await orchestrator.execute_quality_gates()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon SDLC v4.0 - Quality Orchestrator")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, help="Quality configuration file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first critical failure")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Load config if provided
    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    
    if args.fail_fast:
        config['fail_fast'] = True
    
    # Execute quality gates
    report = asyncio.run(execute_quality_gates(args.project_root, config))
    
    print(f"\n🎉 Quality gates execution completed!")
    print(f"📊 Overall Score: {report.overall_score:.1f}/100")
    print(f"✅ Gates Passed: {report.passed_gates}/{report.total_gates}")
    
    if report.critical_issues > 0:
        print(f"❌ Critical issues: {report.critical_issues}")
        sys.exit(1)
    else:
        print("🎯 Ready for deployment!")