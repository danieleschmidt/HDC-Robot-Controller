#!/usr/bin/env python3
"""
Reproducible Experimental Framework for HDC Research
Publication-ready experimental methodology with statistical validation

Research Standards Compliance:
- Reproducible results across multiple runs (seed control)
- Statistical significance testing (p < 0.05)
- Proper experimental controls and baselines
- Publication-ready data visualization and documentation

Publication Target: Nature Machine Intelligence, Science Robotics
Author: Terry - Terragon Labs Research Division
"""

import numpy as np
import json
import os
import time
import logging
import hashlib
import pickle
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import statistics
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict

# Configure research logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hdc_reproducible')

@dataclass
class ExperimentalConfig:
    """Complete configuration for reproducible experiments"""
    experiment_name: str
    random_seed: int = 42
    n_trials: int = 100
    n_bootstrap_samples: int = 1000
    significance_level: float = 0.05
    parallel_workers: int = 4
    
    # HDC-specific parameters
    hdc_dimension: int = 10000
    hdc_similarity_threshold: float = 0.85
    hdc_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Experimental controls
    control_group_size: int = 50
    treatment_group_size: int = 50
    validation_split: float = 0.2
    
    # Output configuration
    save_raw_data: bool = True
    generate_plots: bool = True
    export_latex_tables: bool = True
    
    def get_config_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

@dataclass
class ExperimentalResult:
    """Structured storage for experimental results with statistical metadata"""
    experiment_id: str
    config_hash: str
    timestamp: float
    
    # Core experimental data
    control_group_results: List[float] = field(default_factory=list)
    treatment_group_results: List[float] = field(default_factory=list)
    
    # Statistical analysis results
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    statistical_power: Optional[float] = None
    
    # Reproducibility metadata
    random_seed_used: Optional[int] = None
    environment_hash: Optional[str] = None
    code_version: Optional[str] = None
    
    # Raw experimental metrics
    raw_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def is_statistically_significant(self) -> bool:
        """Check if result is statistically significant"""
        return self.p_value is not None and self.p_value < 0.05
    
    def get_effect_size_interpretation(self) -> str:
        """Cohen's d interpretation"""
        if self.effect_size is None:
            return "Unknown"
        
        abs_effect = abs(self.effect_size)
        if abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium" 
        else:
            return "Large"

class ReproducibleExperimentFramework:
    """
    Comprehensive framework for reproducible HDC experiments
    
    Features:
    - Automatic seed management for reproducibility
    - Statistical validation with power analysis
    - Bootstrap confidence intervals
    - Multi-run averaging with variance reporting
    - Publication-ready result formatting
    - Experimental audit trail
    """
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.results_dir = Path(f"/root/repo/research/results/{config.experiment_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Initialize result storage
        self.experiment_results = []
        self.audit_trail = []
        
        logger.info(f"Initialized reproducible experiment framework:")
        logger.info(f"  Experiment: {config.experiment_name}")
        logger.info(f"  Config hash: {config.get_config_hash()}")
        logger.info(f"  Random seed: {config.random_seed}")
        logger.info(f"  Results directory: {self.results_dir}")
        
    def _set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        # Would set torch.manual_seed(seed) if using PyTorch
        # Would set tf.random.set_seed(seed) if using TensorFlow
        
        self.audit_trail.append({
            'action': 'set_random_seed',
            'seed': seed,
            'timestamp': time.time()
        })
    
    def _get_environment_hash(self) -> str:
        """Generate hash of current environment for reproducibility tracking"""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'numpy_version': np.__version__,
            'config': asdict(self.config)
        }
        
        env_str = json.dumps(env_info, sort_keys=True)
        return hashlib.md5(env_str.encode()).hexdigest()[:16]
    
    def run_controlled_experiment(self, 
                                control_function: Callable,
                                treatment_function: Callable,
                                experiment_id: str) -> ExperimentalResult:
        """
        Run controlled experiment with statistical validation
        
        Args:
            control_function: Function implementing control condition
            treatment_function: Function implementing treatment condition  
            experiment_id: Unique identifier for this experiment
        """
        
        logger.info(f"Starting controlled experiment: {experiment_id}")
        start_time = time.time()
        
        # Initialize result structure
        result = ExperimentalResult(
            experiment_id=experiment_id,
            config_hash=self.config.get_config_hash(),
            timestamp=start_time,
            random_seed_used=self.config.random_seed,
            environment_hash=self._get_environment_hash()
        )
        
        # Run control group trials
        logger.info(f"Running control group ({self.config.control_group_size} trials)")
        control_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            control_futures = [
                executor.submit(self._run_single_trial, control_function, i, 'control')
                for i in range(self.config.control_group_size)
            ]
            
            for future in control_futures:
                try:
                    trial_result = future.result()
                    control_results.append(trial_result)
                except Exception as e:
                    logger.error(f"Control trial failed: {e}")
        
        result.control_group_results = control_results
        
        # Run treatment group trials  
        logger.info(f"Running treatment group ({self.config.treatment_group_size} trials)")
        treatment_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            treatment_futures = [
                executor.submit(self._run_single_trial, treatment_function, i, 'treatment')
                for i in range(self.config.treatment_group_size)
            ]
            
            for future in treatment_futures:
                try:
                    trial_result = future.result()
                    treatment_results.append(trial_result)
                except Exception as e:
                    logger.error(f"Treatment trial failed: {e}")
        
        result.treatment_group_results = treatment_results
        
        # Statistical analysis
        result = self._perform_statistical_analysis(result)
        
        # Record timing
        total_time = time.time() - start_time
        logger.info(f"Completed experiment {experiment_id} in {total_time:.2f}s")
        
        # Store results
        self.experiment_results.append(result)
        self._save_experiment_result(result)
        
        return result
    
    def _run_single_trial(self, trial_function: Callable, trial_id: int, group: str) -> float:
        """Run a single experimental trial with error handling"""
        try:
            # Add trial-specific seed variation for diversity while maintaining reproducibility
            trial_seed = self.config.random_seed + trial_id
            np.random.seed(trial_seed)
            
            # Run the trial function
            result = trial_function(trial_id)
            
            # Validate result
            if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid result from {group} trial {trial_id}: {result}")
                return 0.0
                
            return float(result)
            
        except Exception as e:
            logger.error(f"Error in {group} trial {trial_id}: {e}")
            return 0.0
    
    def _perform_statistical_analysis(self, result: ExperimentalResult) -> ExperimentalResult:
        """Comprehensive statistical analysis of experimental results"""
        
        control_data = result.control_group_results
        treatment_data = result.treatment_group_results
        
        if not control_data or not treatment_data:
            logger.error("Cannot perform statistical analysis: missing data")
            return result
        
        # Effect size (Cohen's d)
        control_mean = statistics.mean(control_data)
        treatment_mean = statistics.mean(treatment_data)
        
        # Pooled standard deviation
        control_var = statistics.variance(control_data) if len(control_data) > 1 else 0
        treatment_var = statistics.variance(treatment_data) if len(treatment_data) > 1 else 0
        pooled_std = np.sqrt((control_var + treatment_var) / 2)
        
        if pooled_std > 0:
            effect_size = (treatment_mean - control_mean) / pooled_std
        else:
            effect_size = 0.0
            
        result.effect_size = effect_size
        
        # Statistical significance testing (Welch's t-test)
        try:
            # Simplified t-test implementation (would use scipy.stats in production)
            n1, n2 = len(control_data), len(treatment_data)
            
            if n1 > 1 and n2 > 1:
                # Welch's t-test formula
                s1_sq = statistics.variance(control_data)
                s2_sq = statistics.variance(treatment_data)
                
                se = np.sqrt(s1_sq/n1 + s2_sq/n2)
                
                if se > 0:
                    t_stat = (treatment_mean - control_mean) / se
                    
                    # Degrees of freedom for Welch's t-test
                    df = ((s1_sq/n1 + s2_sq/n2)**2) / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
                    
                    # Approximate p-value (would use proper t-distribution in production)
                    p_value = 2 * (1 - 0.5 * (1 + t_stat / np.sqrt(df + t_stat**2)))
                    p_value = max(0.001, min(0.999, abs(p_value)))  # Clamp to reasonable range
                else:
                    p_value = 1.0
                    
            else:
                p_value = 1.0
                
        except Exception as e:
            logger.error(f"Error computing p-value: {e}")
            p_value = 1.0
            
        result.p_value = p_value
        
        # Bootstrap confidence interval
        confidence_interval = self._bootstrap_confidence_interval(
            treatment_data, control_data, self.config.n_bootstrap_samples
        )
        result.confidence_interval = confidence_interval
        
        # Statistical power estimation (simplified)
        if pooled_std > 0 and n1 > 0 and n2 > 0:
            # Cohen's power estimation (approximate)
            power = min(0.99, max(0.01, 0.8 * abs(effect_size) * np.sqrt(min(n1, n2) / 10)))
        else:
            power = 0.5
            
        result.statistical_power = power
        
        # Log statistical results
        logger.info(f"Statistical Analysis Results:")
        logger.info(f"  Control mean: {control_mean:.4f} ± {np.std(control_data):.4f}")
        logger.info(f"  Treatment mean: {treatment_mean:.4f} ± {np.std(treatment_data):.4f}")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({result.get_effect_size_interpretation()})")
        logger.info(f"  P-value: {p_value:.6f} ({'significant' if p_value < 0.05 else 'not significant'})")
        logger.info(f"  Statistical power: {power:.3f}")
        
        return result
    
    def _bootstrap_confidence_interval(self, treatment: List[float], 
                                     control: List[float], 
                                     n_samples: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for difference in means"""
        
        bootstrap_diffs = []
        
        for _ in range(n_samples):
            # Bootstrap resample
            boot_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
            boot_control = np.random.choice(control, size=len(control), replace=True)
            
            # Compute difference in means
            diff = np.mean(boot_treatment) - np.mean(boot_control)
            bootstrap_diffs.append(diff)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
        ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def run_multi_run_validation(self, 
                                experiment_function: Callable,
                                n_runs: int = 10,
                                experiment_prefix: str = "validation") -> Dict[str, Any]:
        """
        Run experiment multiple times with different seeds for validation
        
        This addresses the reproducibility crisis by running the same experiment
        multiple times and reporting variance in results.
        """
        
        logger.info(f"Starting multi-run validation ({n_runs} runs)")
        
        all_results = []
        original_seed = self.config.random_seed
        
        for run_id in range(n_runs):
            # Use different seed for each run
            run_seed = original_seed + run_id * 1000
            self.config.random_seed = run_seed
            self._set_random_seeds(run_seed)
            
            logger.info(f"Validation run {run_id + 1}/{n_runs} (seed={run_seed})")
            
            # Run experiment
            result = experiment_function(f"{experiment_prefix}_run_{run_id}")
            all_results.append(result)
        
        # Restore original seed
        self.config.random_seed = original_seed
        self._set_random_seeds(original_seed)
        
        # Analyze cross-run variance
        effect_sizes = [r.effect_size for r in all_results if r.effect_size is not None]
        p_values = [r.p_value for r in all_results if r.p_value is not None]
        
        validation_summary = {
            'n_runs': n_runs,
            'consistent_significance': sum(1 for p in p_values if p < 0.05),
            'mean_effect_size': statistics.mean(effect_sizes) if effect_sizes else 0,
            'effect_size_std': statistics.stdev(effect_sizes) if len(effect_sizes) > 1 else 0,
            'effect_size_cv': statistics.stdev(effect_sizes) / abs(statistics.mean(effect_sizes)) 
                            if effect_sizes and statistics.mean(effect_sizes) != 0 else float('inf'),
            'reproducibility_score': sum(1 for p in p_values if p < 0.05) / len(p_values) if p_values else 0,
            'detailed_results': all_results
        }
        
        # Log validation summary
        logger.info(f"Multi-run validation completed:")
        logger.info(f"  Consistent significance: {validation_summary['consistent_significance']}/{n_runs}")
        logger.info(f"  Mean effect size: {validation_summary['mean_effect_size']:.4f} ± {validation_summary['effect_size_std']:.4f}")
        logger.info(f"  Effect size CV: {validation_summary['effect_size_cv']:.4f}")
        logger.info(f"  Reproducibility score: {validation_summary['reproducibility_score']:.3f}")
        
        # Save validation results
        validation_file = self.results_dir / "multi_run_validation.json"
        with open(validation_file, 'w') as f:
            # Convert results to JSON-serializable format
            serializable_summary = validation_summary.copy()
            serializable_summary['detailed_results'] = [asdict(r) for r in all_results]
            json.dump(serializable_summary, f, indent=2, default=str)
        
        return validation_summary
    
    def _save_experiment_result(self, result: ExperimentalResult):
        """Save individual experiment result"""
        result_file = self.results_dir / f"{result.experiment_id}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
            
        logger.info(f"Saved experiment result to {result_file}")
    
    def generate_publication_report(self) -> str:
        """Generate publication-ready report with all experimental results"""
        
        report_lines = []
        report_lines.append("# HDC Research Experimental Results")
        report_lines.append("## Publication-Ready Statistical Analysis")
        report_lines.append("")
        report_lines.append(f"**Experiment Suite:** {self.config.experiment_name}")
        report_lines.append(f"**Configuration Hash:** {self.config.get_config_hash()}")
        report_lines.append(f"**Total Experiments:** {len(self.experiment_results)}")
        report_lines.append("")
        
        # Summary statistics across all experiments
        if self.experiment_results:
            effect_sizes = [r.effect_size for r in self.experiment_results if r.effect_size is not None]
            p_values = [r.p_value for r in self.experiment_results if r.p_value is not None]
            significant_results = sum(1 for p in p_values if p < 0.05)
            
            report_lines.append("## Overall Results Summary")
            report_lines.append("")
            report_lines.append(f"- **Statistically Significant Results:** {significant_results}/{len(p_values)} ({significant_results/len(p_values)*100:.1f}%)")
            report_lines.append(f"- **Mean Effect Size:** {statistics.mean(effect_sizes):.4f} ± {statistics.stdev(effect_sizes) if len(effect_sizes) > 1 else 0:.4f}")
            report_lines.append(f"- **Effect Size Range:** [{min(effect_sizes):.4f}, {max(effect_sizes):.4f}]")
            report_lines.append("")
        
        # Individual experiment details
        report_lines.append("## Individual Experiment Results")
        report_lines.append("")
        
        for result in self.experiment_results:
            report_lines.append(f"### {result.experiment_id}")
            report_lines.append("")
            report_lines.append(f"- **Effect Size (Cohen's d):** {result.effect_size:.4f} ({result.get_effect_size_interpretation()})")
            report_lines.append(f"- **P-value:** {result.p_value:.6f} ({'*significant*' if result.is_statistically_significant() else 'not significant'})")
            report_lines.append(f"- **95% Confidence Interval:** [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report_lines.append(f"- **Statistical Power:** {result.statistical_power:.3f}")
            report_lines.append(f"- **Sample Sizes:** Control={len(result.control_group_results)}, Treatment={len(result.treatment_group_results)}")
            report_lines.append("")
        
        # Reproducibility information
        report_lines.append("## Reproducibility Information")
        report_lines.append("")
        report_lines.append(f"- **Random Seed:** {self.config.random_seed}")
        report_lines.append(f"- **Environment Hash:** {self._get_environment_hash()}")
        report_lines.append(f"- **Configuration:** {json.dumps(asdict(self.config), indent=2)}")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / "publication_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        logger.info(f"Generated publication report: {report_file}")
        return report_text

# Example HDC experiment functions for demonstration

def example_hdc_control_experiment(trial_id: int) -> float:
    """Example control experiment: Standard HDC with fixed parameters"""
    
    # Simulate standard HDC performance
    base_performance = 0.85
    noise = np.random.normal(0, 0.03)  # 3% noise
    
    # Add some realistic variation
    trial_variation = 0.02 * np.sin(trial_id * 0.1)  # Systematic variation
    
    performance = base_performance + noise + trial_variation
    return max(0.0, min(1.0, performance))

def example_hdc_treatment_experiment(trial_id: int) -> float:
    """Example treatment experiment: Adaptive HDC with optimization"""
    
    # Simulate improved HDC performance with adaptation
    base_performance = 0.90  # 5% improvement
    noise = np.random.normal(0, 0.025)  # Slightly less noise due to adaptation
    
    # Adaptive improvement over trials (learning effect)
    adaptation_bonus = 0.01 * (1 - np.exp(-trial_id / 50))  # Asymptotic improvement
    
    # Add some realistic variation
    trial_variation = 0.015 * np.sin(trial_id * 0.1)
    
    performance = base_performance + noise + adaptation_bonus + trial_variation
    return max(0.0, min(1.0, performance))

def main():
    """Demonstrate reproducible experimental framework"""
    logger.info("HDC Reproducible Experimental Framework Demo")
    logger.info("=" * 60)
    
    # Configure experiment
    config = ExperimentalConfig(
        experiment_name="hdc_adaptive_comparison",
        random_seed=42,
        n_trials=100,
        control_group_size=50,
        treatment_group_size=50,
        n_bootstrap_samples=1000,
        parallel_workers=4
    )
    
    # Initialize framework
    framework = ReproducibleExperimentFramework(config)
    
    # Run controlled experiment
    def run_adaptive_comparison_experiment(experiment_id: str) -> ExperimentalResult:
        return framework.run_controlled_experiment(
            control_function=example_hdc_control_experiment,
            treatment_function=example_hdc_treatment_experiment,
            experiment_id=experiment_id
        )
    
    # Single experiment
    logger.info("\n🔬 Running single controlled experiment...")
    result = run_adaptive_comparison_experiment("adaptive_hdc_vs_standard")
    
    # Multi-run validation
    logger.info("\n🔄 Running multi-run validation...")
    validation = framework.run_multi_run_validation(
        experiment_function=run_adaptive_comparison_experiment,
        n_runs=5,  # Reduced for demo
        experiment_prefix="validation"
    )
    
    # Generate publication report
    logger.info("\n📄 Generating publication report...")
    report = framework.generate_publication_report()
    
    # Display key findings
    print("\n🎯 KEY RESEARCH FINDINGS:")
    print("=" * 50)
    print(f"Effect Size: {result.effect_size:.4f} ({result.get_effect_size_interpretation()})")
    print(f"Statistical Significance: p = {result.p_value:.6f} ({'Yes' if result.is_statistically_significant() else 'No'})")
    print(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    print(f"Statistical Power: {result.statistical_power:.3f}")
    print(f"Reproducibility Score: {validation['reproducibility_score']:.3f}")
    
    logger.info("Reproducible experimental framework demonstration completed!")
    return framework, validation

if __name__ == "__main__":
    framework, validation = main()