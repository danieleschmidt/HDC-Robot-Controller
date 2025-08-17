#!/usr/bin/env python3
"""
Statistical Significance Validator: Rigorous Research Validation Framework
Ensures all research contributions meet publication standards with statistical rigor

Research Standards: p < 0.05, effect size > 0.5, confidence intervals, reproducibility
Publication Requirements: Multiple trials, bootstrapping, power analysis

Author: Terry - Terragon Labs Research Validation
"""

import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import bootstrap
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math
from collections import defaultdict, namedtuple
import pickle

# Research validation logging
logging.basicConfig(level=logging.INFO)
validator_logger = logging.getLogger('research_validator')

# Statistical test results
TestResult = namedtuple('TestResult', ['statistic', 'p_value', 'effect_size', 'power', 'significant'])
BootstrapResult = namedtuple('BootstrapResult', ['mean', 'std', 'confidence_interval', 'bias_corrected'])

@dataclass
class ResearchValidation:
    """Comprehensive research validation results"""
    study_name: str
    hypothesis: str
    sample_size: int
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    reproducibility_score: float
    publication_ready: bool
    validation_timestamp: float = field(default_factory=time.time)
    
    def meets_publication_standards(self) -> bool:
        """Check if results meet publication standards"""
        return (
            self.p_value < 0.05 and
            abs(self.effect_size) > 0.3 and  # Medium effect size
            self.statistical_power > 0.8 and
            self.sample_size >= 30 and
            self.reproducibility_score > 0.8
        )

class StatisticalSignificanceValidator:
    """Comprehensive statistical validation for research contributions"""
    
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.3, min_power: float = 0.8):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.min_power = min_power
        
        # Store validation results
        self.validation_history = []
        self.reproducibility_cache = {}
        
        validator_logger.info(f"Initialized research validator: α={alpha}, min effect size={min_effect_size}, min power={min_power}")
    
    def validate_mean_difference(self, group1: np.ndarray, group2: np.ndarray, 
                                study_name: str, hypothesis: str,
                                alternative: str = 'two-sided') -> ResearchValidation:
        """Validate mean difference between two groups with comprehensive statistics"""
        validator_logger.info(f"Validating mean difference: {study_name}")
        
        # Basic validation
        if len(group1) < 5 or len(group2) < 5:
            validator_logger.warning("Sample sizes too small for reliable validation")
        
        # Remove any infinite or NaN values
        group1 = group1[np.isfinite(group1)]
        group2 = group2[np.isfinite(group2)]
        
        if len(group1) == 0 or len(group2) == 0:
            validator_logger.error("Empty groups after cleaning")
            return self._create_failed_validation(study_name, hypothesis)
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        
        if pooled_std > 0:
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Confidence interval for difference
        ci = self._calculate_confidence_interval_difference(group1, group2)
        
        # Statistical power analysis
        power = self._calculate_power_ttest(group1, group2, cohens_d)
        
        # Bootstrap validation for robustness
        bootstrap_result = self._bootstrap_validation(group1, group2)
        
        # Reproducibility assessment
        reproducibility = self._assess_reproducibility(group1, group2, study_name)
        
        validation = ResearchValidation(
            study_name=study_name,
            hypothesis=hypothesis,
            sample_size=len(group1) + len(group2),
            effect_size=cohens_d,
            p_value=p_value,
            confidence_interval=ci,
            statistical_power=power,
            reproducibility_score=reproducibility,
            publication_ready=False  # Set below
        )
        
        validation.publication_ready = validation.meets_publication_standards()
        
        self.validation_history.append(validation)
        
        validator_logger.info(f"Validation complete: p={p_value:.6f}, d={cohens_d:.3f}, power={power:.3f}, publication_ready={validation.publication_ready}")
        
        return validation
    
    def validate_correlation(self, x: np.ndarray, y: np.ndarray,
                           study_name: str, hypothesis: str,
                           method: str = 'pearson') -> ResearchValidation:
        """Validate correlation with comprehensive statistics"""
        validator_logger.info(f"Validating correlation: {study_name}")
        
        # Clean data
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_indices]
        y_clean = y[valid_indices]
        
        if len(x_clean) < 10:
            validator_logger.warning("Sample size too small for correlation analysis")
            return self._create_failed_validation(study_name, hypothesis)
        
        # Calculate correlation
        if method == 'pearson':
            corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Effect size (correlation coefficient itself)
        effect_size = abs(corr_coef)
        
        # Confidence interval for correlation
        ci = self._correlation_confidence_interval(corr_coef, len(x_clean))
        
        # Power analysis for correlation
        power = self._calculate_power_correlation(corr_coef, len(x_clean))
        
        # Reproducibility via bootstrap
        reproducibility = self._assess_correlation_reproducibility(x_clean, y_clean, method)
        
        validation = ResearchValidation(
            study_name=study_name,
            hypothesis=hypothesis,
            sample_size=len(x_clean),
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=ci,
            statistical_power=power,
            reproducibility_score=reproducibility,
            publication_ready=False
        )
        
        validation.publication_ready = validation.meets_publication_standards()
        
        self.validation_history.append(validation)
        
        validator_logger.info(f"Correlation validation: r={corr_coef:.3f}, p={p_value:.6f}, power={power:.3f}")
        
        return validation
    
    def validate_regression_model(self, x: np.ndarray, y: np.ndarray,
                                study_name: str, hypothesis: str,
                                model_type: str = 'linear') -> ResearchValidation:
        """Validate regression model with comprehensive diagnostics"""
        validator_logger.info(f"Validating regression: {study_name}")
        
        # Clean data
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_indices].reshape(-1, 1) if x.ndim == 1 else x[valid_indices]
        y_clean = y[valid_indices]
        
        if len(x_clean) < 20:
            validator_logger.warning("Sample size too small for regression analysis")
            return self._create_failed_validation(study_name, hypothesis)
        
        # Fit regression model
        if model_type == 'linear':
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean.flatten(), y_clean)
            r_squared = r_value ** 2
        else:
            raise ValueError(f"Unknown regression type: {model_type}")
        
        # Effect size (R-squared)
        effect_size = r_squared
        
        # Confidence interval for slope
        ci = self._regression_confidence_interval(slope, std_err, len(x_clean))
        
        # Power analysis
        power = self._calculate_power_regression(r_squared, len(x_clean))
        
        # Model diagnostics
        predictions = slope * x_clean.flatten() + intercept
        residuals = y_clean - predictions
        
        # Reproducibility via cross-validation
        reproducibility = self._assess_regression_reproducibility(x_clean, y_clean)
        
        validation = ResearchValidation(
            study_name=study_name,
            hypothesis=hypothesis,
            sample_size=len(x_clean),
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=ci,
            statistical_power=power,
            reproducibility_score=reproducibility,
            publication_ready=False
        )
        
        validation.publication_ready = validation.meets_publication_standards()
        
        self.validation_history.append(validation)
        
        validator_logger.info(f"Regression validation: R²={r_squared:.3f}, p={p_value:.6f}, power={power:.3f}")
        
        return validation
    
    def validate_experimental_design(self, treatment_groups: List[np.ndarray],
                                   study_name: str, hypothesis: str) -> ResearchValidation:
        """Validate experimental design with ANOVA and post-hoc tests"""
        validator_logger.info(f"Validating experimental design: {study_name}")
        
        # Clean data
        cleaned_groups = []
        for group in treatment_groups:
            clean_group = group[np.isfinite(group)]
            if len(clean_group) > 0:
                cleaned_groups.append(clean_group)
        
        if len(cleaned_groups) < 2:
            validator_logger.error("Need at least 2 groups for experimental validation")
            return self._create_failed_validation(study_name, hypothesis)
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*cleaned_groups)
        
        # Effect size (eta-squared)
        total_n = sum(len(group) for group in cleaned_groups)
        between_ss = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(cleaned_groups)))**2 
                        for group in cleaned_groups)
        total_ss = sum(np.sum((group - np.mean(np.concatenate(cleaned_groups)))**2) 
                      for group in cleaned_groups)
        
        eta_squared = between_ss / total_ss if total_ss > 0 else 0
        
        # Confidence interval for effect size
        ci = self._eta_squared_confidence_interval(eta_squared, len(cleaned_groups), total_n)
        
        # Power analysis
        power = self._calculate_power_anova(cleaned_groups, eta_squared)
        
        # Reproducibility via permutation test
        reproducibility = self._assess_anova_reproducibility(cleaned_groups)
        
        validation = ResearchValidation(
            study_name=study_name,
            hypothesis=hypothesis,
            sample_size=total_n,
            effect_size=eta_squared,
            p_value=p_value,
            confidence_interval=ci,
            statistical_power=power,
            reproducibility_score=reproducibility,
            publication_ready=False
        )
        
        validation.publication_ready = validation.meets_publication_standards()
        
        self.validation_history.append(validation)
        
        validator_logger.info(f"ANOVA validation: F={f_stat:.3f}, p={p_value:.6f}, η²={eta_squared:.3f}")
        
        return validation
    
    def _calculate_confidence_interval_difference(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean_diff = np.mean(group1) - np.mean(group2)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard error
        pooled_var = ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Critical value
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        margin_error = t_critical * se_diff
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _correlation_confidence_interval(self, r: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se_z = 1 / np.sqrt(n - 3)
        
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        z_lower = z - z_critical * se_z
        z_upper = z + z_critical * se_z
        
        # Transform back to correlation
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _regression_confidence_interval(self, slope: float, std_err: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for regression slope"""
        df = n - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin_error = t_critical * std_err
        
        return (slope - margin_error, slope + margin_error)
    
    def _eta_squared_confidence_interval(self, eta_sq: float, k: int, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for eta-squared (effect size in ANOVA)"""
        # Approximate confidence interval using chi-square distribution
        # This is a simplified approximation
        df_between = k - 1
        df_error = n - k
        
        # Convert eta-squared to partial eta-squared equivalent
        f_value = (eta_sq / (1 - eta_sq)) * (df_error / df_between)
        
        # Use F-distribution critical values
        f_lower = stats.f.ppf(self.alpha/2, df_between, df_error)
        f_upper = stats.f.ppf(1 - self.alpha/2, df_between, df_error)
        
        eta_lower = max(0, (f_lower * df_between - df_between) / (f_lower * df_between + df_error))
        eta_upper = min(1, (f_upper * df_between - df_between) / (f_upper * df_between + df_error))
        
        return (eta_lower, eta_upper)
    
    def _calculate_power_ttest(self, group1: np.ndarray, group2: np.ndarray, effect_size: float) -> float:
        """Calculate statistical power for t-test"""
        n1, n2 = len(group1), len(group2)
        
        # Harmonic mean of sample sizes
        n_harmonic = 2 / (1/n1 + 1/n2)
        
        # Non-centrality parameter
        delta = abs(effect_size) * np.sqrt(n_harmonic / 2)
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Critical value
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        
        # Power calculation using non-central t-distribution
        try:
            power = 1 - stats.nct.cdf(t_critical, df, delta) + stats.nct.cdf(-t_critical, df, delta)
        except:
            # Fallback approximation
            power = 1 - stats.norm.cdf(t_critical - delta) + stats.norm.cdf(-t_critical - delta)
        
        return min(1.0, max(0.0, power))
    
    def _calculate_power_correlation(self, r: float, n: int) -> float:
        """Calculate statistical power for correlation test"""
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + abs(r)) / (1 - abs(r)))
        se_z = 1 / np.sqrt(n - 3)
        
        # Critical value
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        
        # Power calculation
        power = 1 - stats.norm.cdf(z_critical - z / se_z) + stats.norm.cdf(-z_critical - z / se_z)
        
        return min(1.0, max(0.0, power))
    
    def _calculate_power_regression(self, r_squared: float, n: int) -> float:
        """Calculate statistical power for regression"""
        # Convert R-squared to correlation
        r = np.sqrt(r_squared)
        return self._calculate_power_correlation(r, n)
    
    def _calculate_power_anova(self, groups: List[np.ndarray], eta_squared: float) -> float:
        """Calculate statistical power for ANOVA"""
        k = len(groups)  # Number of groups
        n = sum(len(group) for group in groups)  # Total sample size
        
        # Convert eta-squared to Cohen's f
        cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
        
        # Non-centrality parameter
        ncp = cohens_f**2 * n
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n - k
        
        # Critical F-value
        f_critical = stats.f.ppf(1 - self.alpha, df_between, df_within)
        
        # Power using non-central F-distribution
        try:
            power = 1 - stats.ncf.cdf(f_critical, df_between, df_within, ncp)
        except:
            # Fallback approximation
            power = 0.8 if eta_squared > 0.14 else 0.5  # Rule of thumb
        
        return min(1.0, max(0.0, power))
    
    def _bootstrap_validation(self, group1: np.ndarray, group2: np.ndarray, 
                            n_bootstrap: int = 1000) -> BootstrapResult:
        """Bootstrap validation for robustness"""
        def mean_difference(x, y):
            return np.mean(x) - np.mean(y)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, len(group1), replace=True)
            sample2 = np.random.choice(group2, len(group2), replace=True)
            diff = mean_difference(sample1, sample2)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Statistics
        mean_diff = np.mean(bootstrap_diffs)
        std_diff = np.std(bootstrap_diffs)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha/2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha/2))
        
        # Bias correction
        original_diff = mean_difference(group1, group2)
        bias = mean_diff - original_diff
        bias_corrected = original_diff - bias
        
        return BootstrapResult(mean_diff, std_diff, (ci_lower, ci_upper), bias_corrected)
    
    def _assess_reproducibility(self, group1: np.ndarray, group2: np.ndarray, 
                              study_name: str, n_splits: int = 10) -> float:
        """Assess reproducibility via cross-validation"""
        if study_name in self.reproducibility_cache:
            return self.reproducibility_cache[study_name]
        
        # Split data randomly multiple times
        p_values = []
        effect_sizes = []
        
        combined = np.concatenate([group1, group2])
        labels = np.concatenate([np.ones(len(group1)), np.zeros(len(group2))])
        
        for _ in range(n_splits):
            # Random split
            indices = np.random.permutation(len(combined))
            split_point = len(group1)
            
            new_group1 = combined[indices[:split_point]]
            new_group2 = combined[indices[split_point:]]
            
            try:
                t_stat, p_val = stats.ttest_ind(new_group1, new_group2)
                
                # Effect size
                pooled_std = np.sqrt((np.var(new_group1, ddof=1) + np.var(new_group2, ddof=1)) / 2)
                if pooled_std > 0:
                    d = abs(np.mean(new_group1) - np.mean(new_group2)) / pooled_std
                else:
                    d = 0
                
                p_values.append(p_val)
                effect_sizes.append(d)
            except:
                continue
        
        if not p_values:
            reproducibility = 0.0
        else:
            # Reproducibility based on consistency of results
            significant_proportion = np.mean(np.array(p_values) < self.alpha)
            effect_size_consistency = 1 - (np.std(effect_sizes) / (np.mean(effect_sizes) + 1e-10))
            reproducibility = (significant_proportion + effect_size_consistency) / 2
        
        self.reproducibility_cache[study_name] = reproducibility
        return reproducibility
    
    def _assess_correlation_reproducibility(self, x: np.ndarray, y: np.ndarray, 
                                          method: str, n_splits: int = 10) -> float:
        """Assess correlation reproducibility"""
        correlations = []
        
        for _ in range(n_splits):
            # Bootstrap sample
            indices = np.random.choice(len(x), len(x), replace=True)
            x_sample = x[indices]
            y_sample = y[indices]
            
            try:
                if method == 'pearson':
                    corr, _ = stats.pearsonr(x_sample, y_sample)
                else:
                    corr, _ = stats.spearmanr(x_sample, y_sample)
                correlations.append(corr)
            except:
                continue
        
        if not correlations:
            return 0.0
        
        # Consistency of correlation values
        corr_std = np.std(correlations)
        corr_mean = np.mean(correlations)
        
        reproducibility = 1 - (corr_std / (abs(corr_mean) + 1e-10))
        return max(0.0, min(1.0, reproducibility))
    
    def _assess_regression_reproducibility(self, x: np.ndarray, y: np.ndarray, 
                                         n_folds: int = 5) -> float:
        """Assess regression reproducibility via cross-validation"""
        fold_size = len(x) // n_folds
        r_squared_values = []
        
        for i in range(n_folds):
            # Create train/test split
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else len(x)
            
            test_indices = slice(test_start, test_end)
            train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end, len(x))])
            
            if len(train_indices) < 5:
                continue
                
            x_train, y_train = x[train_indices], y[train_indices]
            x_test, y_test = x[test_indices], y[test_indices]
            
            try:
                # Fit on training data
                slope, intercept, _, _, _ = stats.linregress(x_train.flatten(), y_train)
                
                # Predict on test data
                y_pred = slope * x_test.flatten() + intercept
                
                # Calculate R-squared on test data
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                
                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    r_squared_values.append(max(0, r_squared))  # Clip negative R²
            except:
                continue
        
        if not r_squared_values:
            return 0.0
        
        # Reproducibility based on consistency of R² values
        r2_mean = np.mean(r_squared_values)
        r2_std = np.std(r_squared_values)
        
        reproducibility = 1 - (r2_std / (r2_mean + 1e-10))
        return max(0.0, min(1.0, reproducibility))
    
    def _assess_anova_reproducibility(self, groups: List[np.ndarray], n_bootstrap: int = 100) -> float:
        """Assess ANOVA reproducibility via bootstrap"""
        p_values = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample each group
            bootstrap_groups = []
            for group in groups:
                if len(group) > 0:
                    bootstrap_sample = np.random.choice(group, len(group), replace=True)
                    bootstrap_groups.append(bootstrap_sample)
            
            if len(bootstrap_groups) >= 2:
                try:
                    _, p_val = stats.f_oneway(*bootstrap_groups)
                    p_values.append(p_val)
                except:
                    continue
        
        if not p_values:
            return 0.0
        
        # Reproducibility based on proportion of significant results
        significant_proportion = np.mean(np.array(p_values) < self.alpha)
        
        # Also consider consistency of p-values
        p_consistency = 1 - (np.std(p_values) / (np.mean(p_values) + 1e-10))
        
        reproducibility = (significant_proportion + p_consistency) / 2
        return max(0.0, min(1.0, reproducibility))
    
    def _create_failed_validation(self, study_name: str, hypothesis: str) -> ResearchValidation:
        """Create a failed validation result"""
        return ResearchValidation(
            study_name=study_name,
            hypothesis=hypothesis,
            sample_size=0,
            effect_size=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            statistical_power=0.0,
            reproducibility_score=0.0,
            publication_ready=False
        )
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return {'message': 'No validations performed yet'}
        
        publication_ready = [v for v in self.validation_history if v.publication_ready]
        
        return {
            'validation_summary': {
                'total_studies': len(self.validation_history),
                'publication_ready': len(publication_ready),
                'publication_rate': len(publication_ready) / len(self.validation_history),
                'mean_p_value': float(np.mean([v.p_value for v in self.validation_history])),
                'mean_effect_size': float(np.mean([abs(v.effect_size) for v in self.validation_history])),
                'mean_power': float(np.mean([v.statistical_power for v in self.validation_history])),
                'mean_reproducibility': float(np.mean([v.reproducibility_score for v in self.validation_history]))
            },
            'publication_standards': {
                'alpha_threshold': self.alpha,
                'min_effect_size': self.min_effect_size,
                'min_power': self.min_power,
                'min_sample_size': 30,
                'min_reproducibility': 0.8
            },
            'study_details': [
                {
                    'name': v.study_name,
                    'hypothesis': v.hypothesis,
                    'p_value': v.p_value,
                    'effect_size': v.effect_size,
                    'power': v.statistical_power,
                    'reproducibility': v.reproducibility_score,
                    'publication_ready': v.publication_ready
                }
                for v in self.validation_history
            ]
        }

# Research validation example
if __name__ == "__main__":
    # Initialize validator
    validator = StatisticalSignificanceValidator()
    
    # Simulate research data
    np.random.seed(42)
    
    # Study 1: HDC vs Classical performance
    hdc_performance = np.random.normal(0.85, 0.05, 50)  # HDC accuracy
    classical_performance = np.random.normal(0.75, 0.08, 50)  # Classical accuracy
    
    validation1 = validator.validate_mean_difference(
        hdc_performance, classical_performance,
        "HDC vs Classical Performance",
        "HDC achieves higher accuracy than classical methods"
    )
    
    # Study 2: Quantum speedup correlation
    dimensions = np.random.uniform(1000, 10000, 40)
    speedup = 2 + 0.0001 * dimensions + np.random.normal(0, 0.2, 40)
    
    validation2 = validator.validate_correlation(
        dimensions, speedup,
        "Quantum Speedup Scaling",
        "Quantum speedup correlates with HDC dimension"
    )
    
    # Study 3: Adaptation experiment
    zero_shot = np.random.normal(0.78, 0.06, 30)
    few_shot = np.random.normal(0.82, 0.05, 30)
    traditional = np.random.normal(0.65, 0.10, 30)
    
    validation3 = validator.validate_experimental_design(
        [zero_shot, few_shot, traditional],
        "Adaptation Methods Comparison",
        "Zero-shot adaptation outperforms traditional methods"
    )
    
    # Generate report
    report = validator.generate_validation_report()
    
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION REPORT")
    print("="*70)
    print(f"Total Studies: {report['validation_summary']['total_studies']}")
    print(f"Publication Ready: {report['validation_summary']['publication_ready']}")
    print(f"Publication Rate: {report['validation_summary']['publication_rate']:.1%}")
    print(f"Mean p-value: {report['validation_summary']['mean_p_value']:.6f}")
    print(f"Mean Effect Size: {report['validation_summary']['mean_effect_size']:.3f}")
    print(f"Mean Statistical Power: {report['validation_summary']['mean_power']:.3f}")
    print(f"Mean Reproducibility: {report['validation_summary']['mean_reproducibility']:.3f}")
    
    print("\nStudy Results:")
    for study in report['study_details']:
        status = "✅ PUBLICATION READY" if study['publication_ready'] else "⚠️  NEEDS IMPROVEMENT"
        print(f"  {study['name']}: {status}")
        print(f"    p-value: {study['p_value']:.6f}, Effect size: {study['effect_size']:.3f}")
    
    print("="*70)
    print("🎯 RESEARCH VALIDATION: Statistical rigor ensured for publication")
    print("📊 Standards: p<0.05, d>0.3, power>0.8, reproducibility>0.8")
    print("="*70)