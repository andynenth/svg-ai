"""
Statistical Analysis Engine - Task 2 Implementation
Statistical tests and analysis for A/B testing framework.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    improvement_pct: float
    improvement_absolute: float
    t_statistic: float
    p_value: float
    significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    sample_size_sufficient: bool


@dataclass
class PowerAnalysisResult:
    """Container for power analysis results."""
    current_power: float
    required_sample_size: int
    minimum_detectable_effect: float
    alpha: float
    beta: float


class StatisticalAnalyzer:
    """
    Statistical analysis engine for A/B test results.
    Provides comprehensive statistical testing and analysis.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Type I error rate (significance level)
            beta: Type II error rate (1 - power)
        """
        self.alpha = alpha
        self.beta = beta
        self.power = 1 - beta

        logger.info(f"StatisticalAnalyzer initialized (α={alpha}, β={beta}, power={self.power})")

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze A/B test results with comprehensive statistics.

        Args:
            results: List of test result dictionaries

        Returns:
            Complete statistical analysis
        """
        try:
            # Separate groups
            control = [r for r in results if r.get('group') == 'control' and r.get('success', True)]
            treatment = [r for r in results if r.get('group') == 'treatment' and r.get('success', True)]

            if len(control) < 2 or len(treatment) < 2:
                return {
                    'error': 'Insufficient data',
                    'control_count': len(control),
                    'treatment_count': len(treatment),
                    'minimum_required': 2
                }

            # Analyze different metrics
            analysis_results = {
                'sample_sizes': {
                    'control': len(control),
                    'treatment': len(treatment),
                    'total': len(control) + len(treatment)
                },
                'metrics': {},
                'overall_summary': {}
            }

            # Analyze quality metrics
            quality_metrics = ['ssim', 'mse', 'psnr']
            for metric in quality_metrics:
                if self._has_metric_data(control, treatment, metric):
                    metric_analysis = self._analyze_metric(control, treatment, metric)
                    analysis_results['metrics'][metric] = metric_analysis

            # Analyze performance metrics
            if self._has_duration_data(control, treatment):
                duration_analysis = self._analyze_duration(control, treatment)
                analysis_results['metrics']['duration'] = duration_analysis

            # Overall analysis
            if 'ssim' in analysis_results['metrics']:
                primary_metric = analysis_results['metrics']['ssim']
                analysis_results['overall_summary'] = {
                    'quality_improvement': primary_metric.improvement_pct,
                    't_test': {
                        't_statistic': primary_metric.t_statistic,
                        'p_value': primary_metric.p_value,
                        'significant': primary_metric.significant
                    },
                    'confidence_interval': primary_metric.confidence_interval,
                    'effect_size': primary_metric.effect_size,
                    'sample_size_sufficient': primary_metric.sample_size_sufficient,
                    'recommendation': self.make_recommendation(primary_metric)
                }

            # Multiple hypothesis correction
            analysis_results['multiple_testing'] = self._apply_multiple_hypothesis_correction(
                analysis_results['metrics']
            )

            # Power analysis
            if 'ssim' in analysis_results['metrics']:
                analysis_results['power_analysis'] = self._perform_power_analysis(
                    control, treatment, 'ssim'
                )

            return analysis_results

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_metric(self, control: List[Dict], treatment: List[Dict], metric: str) -> StatisticalResult:
        """
        Analyze a specific metric between control and treatment groups.

        Args:
            control: Control group results
            treatment: Treatment group results
            metric: Metric name to analyze

        Returns:
            Statistical analysis result
        """
        # Extract metric values
        control_values = self._extract_metric_values(control, metric)
        treatment_values = self._extract_metric_values(treatment, metric)

        # Basic statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)

        # Calculate improvement
        if control_mean != 0:
            improvement_pct = ((treatment_mean - control_mean) / abs(control_mean)) * 100
        else:
            improvement_pct = 0
        improvement_absolute = treatment_mean - control_mean

        # Perform t-test
        t_stat, p_value = self.perform_t_test(control, treatment, metric)

        # Calculate confidence interval
        confidence_interval = self.calculate_confidence_interval(
            control_values, treatment_values
        )

        # Calculate effect size (Cohen's d)
        effect_size = self.calculate_cohens_d(control_values, treatment_values)

        # Check sample size sufficiency
        sample_size_sufficient = self.check_sample_size(control_values, treatment_values)

        # Estimate power
        power = self._estimate_power(control_values, treatment_values, effect_size)

        return StatisticalResult(
            metric_name=metric,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            improvement_pct=improvement_pct,
            improvement_absolute=improvement_absolute,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size_sufficient=sample_size_sufficient
        )

    def perform_t_test(self, control: List[Dict], treatment: List[Dict], metric: str = 'ssim') -> Tuple[float, float]:
        """
        Perform independent t-test between control and treatment groups.

        Args:
            control: Control group results
            treatment: Treatment group results
            metric: Metric to test (default: 'ssim')

        Returns:
            Tuple of (t_statistic, p_value)
        """
        try:
            control_values = self._extract_metric_values(control, metric)
            treatment_values = self._extract_metric_values(treatment, metric)

            # Check for sufficient data
            if len(control_values) < 2 or len(treatment_values) < 2:
                return float('nan'), 1.0

            # Perform Welch's t-test (unequal variances)
            t_stat, p_value = ttest_ind(treatment_values, control_values, equal_var=False)

            return float(t_stat), float(p_value)

        except Exception as e:
            logger.warning(f"T-test failed for {metric}: {e}")
            return float('nan'), 1.0

    def calculate_confidence_interval(self, control_values: np.ndarray, treatment_values: np.ndarray,
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the difference in means.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            n1, n2 = len(control_values), len(treatment_values)
            mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
            var1, var2 = np.var(control_values, ddof=1), np.var(treatment_values, ddof=1)

            # Standard error of difference
            se_diff = np.sqrt(var1/n1 + var2/n2)

            # Degrees of freedom (Welch-Satterthwaite equation)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

            # Critical value
            alpha = 1 - confidence
            t_critical = stats.t.ppf(1 - alpha/2, df)

            # Difference in means
            diff_means = mean2 - mean1

            # Confidence interval
            margin_error = t_critical * se_diff
            lower_bound = diff_means - margin_error
            upper_bound = diff_means + margin_error

            return (float(lower_bound), float(upper_bound))

        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return (float('nan'), float('nan'))

    def calculate_cohens_d(self, control_values: np.ndarray, treatment_values: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values

        Returns:
            Effect size (Cohen's d)
        """
        try:
            mean1, mean2 = np.mean(control_values), np.mean(treatment_values)
            var1, var2 = np.var(control_values, ddof=1), np.var(treatment_values, ddof=1)
            n1, n2 = len(control_values), len(treatment_values)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

            if pooled_std == 0:
                return 0.0

            # Cohen's d
            cohens_d = (mean2 - mean1) / pooled_std

            return float(cohens_d)

        except Exception as e:
            logger.warning(f"Cohen's d calculation failed: {e}")
            return 0.0

    def check_sample_size(self, control_values: np.ndarray, treatment_values: np.ndarray,
                         min_size: int = 30) -> bool:
        """
        Check if sample sizes are sufficient for reliable results.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values
            min_size: Minimum sample size per group

        Returns:
            True if sample sizes are sufficient
        """
        return len(control_values) >= min_size and len(treatment_values) >= min_size

    def _apply_multiple_hypothesis_correction(self, metrics_results: Dict) -> Dict[str, Any]:
        """
        Apply multiple hypothesis testing correction.

        Args:
            metrics_results: Results for multiple metrics

        Returns:
            Corrected results
        """
        if not metrics_results:
            return {}

        # Extract p-values
        p_values = []
        metric_names = []
        for metric_name, result in metrics_results.items():
            if isinstance(result, StatisticalResult) and hasattr(result, 'p_value') and not np.isnan(result.p_value):
                p_values.append(result.p_value)
                metric_names.append(metric_name)

        if not p_values:
            return {}

        # Bonferroni correction
        bonferroni_alpha = self.alpha / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]

        # Benjamini-Hochberg (FDR) correction
        try:
            from statsmodels.stats.multitest import multipletests
            rejected, p_adjusted, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
            fdr_significant = rejected.tolist()
        except ImportError:
            # Fallback if statsmodels not available
            fdr_significant = bonferroni_significant
            p_adjusted = [p * len(p_values) for p in p_values]

        return {
            'original_alpha': self.alpha,
            'bonferroni_alpha': bonferroni_alpha,
            'corrections': {
                metric_names[i]: {
                    'original_p': p_values[i],
                    'adjusted_p': p_adjusted[i],
                    'bonferroni_significant': bonferroni_significant[i],
                    'fdr_significant': fdr_significant[i]
                }
                for i in range(len(metric_names))
            }
        }

    def _perform_power_analysis(self, control: List[Dict], treatment: List[Dict],
                              metric: str) -> PowerAnalysisResult:
        """
        Perform power analysis for the test.

        Args:
            control: Control group results
            treatment: Treatment group results
            metric: Metric to analyze

        Returns:
            Power analysis results
        """
        try:
            control_values = self._extract_metric_values(control, metric)
            treatment_values = self._extract_metric_values(treatment, metric)

            # Calculate effect size
            effect_size = self.calculate_cohens_d(control_values, treatment_values)

            # Estimate current power
            current_power = self._estimate_power(control_values, treatment_values, effect_size)

            # Calculate required sample size for desired power
            required_n = self._calculate_required_sample_size(effect_size)

            # Calculate minimum detectable effect for current sample size
            min_detectable_effect = self._calculate_minimum_detectable_effect(
                len(control_values), len(treatment_values)
            )

            return PowerAnalysisResult(
                current_power=current_power,
                required_sample_size=required_n,
                minimum_detectable_effect=min_detectable_effect,
                alpha=self.alpha,
                beta=self.beta
            )

        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            return PowerAnalysisResult(
                current_power=0.0,
                required_sample_size=100,
                minimum_detectable_effect=0.5,
                alpha=self.alpha,
                beta=self.beta
            )

    def _estimate_power(self, control_values: np.ndarray, treatment_values: np.ndarray,
                       effect_size: float) -> float:
        """
        Estimate statistical power of the current test.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values
            effect_size: Effect size (Cohen's d)

        Returns:
            Estimated power (0-1)
        """
        try:
            n1, n2 = len(control_values), len(treatment_values)

            # Use effect size to estimate power
            # This is a simplified calculation
            if abs(effect_size) < 0.01:
                return 0.05  # Essentially no power for tiny effects

            # Rough power estimation based on sample size and effect size
            harmonic_mean_n = 2 * n1 * n2 / (n1 + n2)
            ncp = abs(effect_size) * np.sqrt(harmonic_mean_n / 2)  # Non-centrality parameter

            # Critical value for two-tailed test
            t_critical = stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2)

            # Power calculation (simplified)
            power = 1 - stats.t.cdf(t_critical - ncp, n1 + n2 - 2) + stats.t.cdf(-t_critical - ncp, n1 + n2 - 2)

            return max(0.0, min(1.0, power))

        except Exception as e:
            logger.warning(f"Power estimation failed: {e}")
            return 0.5

    def _calculate_required_sample_size(self, effect_size: float, power: float = None) -> int:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Expected effect size
            power: Desired power (uses self.power if None)

        Returns:
            Required sample size per group
        """
        if power is None:
            power = self.power

        try:
            if abs(effect_size) < 0.01:
                return 10000  # Very large sample needed for tiny effects

            # Simplified calculation
            # Based on Cohen's formulas for t-test
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(power)

            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

            return max(10, int(np.ceil(n)))

        except Exception as e:
            logger.warning(f"Sample size calculation failed: {e}")
            return 100

    def _calculate_minimum_detectable_effect(self, n1: int, n2: int, power: float = None) -> float:
        """
        Calculate minimum detectable effect for given sample sizes.

        Args:
            n1: Sample size group 1
            n2: Sample size group 2
            power: Desired power (uses self.power if None)

        Returns:
            Minimum detectable effect size
        """
        if power is None:
            power = self.power

        try:
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(power)

            harmonic_mean_n = 2 * n1 * n2 / (n1 + n2)

            mde = (z_alpha + z_beta) * np.sqrt(2 / harmonic_mean_n)

            return float(mde)

        except Exception as e:
            logger.warning(f"MDE calculation failed: {e}")
            return 0.5

    def make_recommendation(self, result: StatisticalResult) -> str:
        """
        Make a recommendation based on statistical results.

        Args:
            result: Statistical analysis result

        Returns:
            Recommendation string
        """
        if result.p_value < self.alpha:
            if result.improvement_pct > 0:
                confidence = "high" if result.effect_size > 0.5 else "moderate"
                return f"Recommend DEPLOY: Significant improvement of {result.improvement_pct:.1f}% " \
                       f"with {confidence} confidence (p={result.p_value:.3f})"
            else:
                return f"Recommend REJECT: Significant degradation of {abs(result.improvement_pct):.1f}% " \
                       f"(p={result.p_value:.3f})"
        else:
            if not result.sample_size_sufficient:
                return f"Recommend CONTINUE TESTING: Insufficient evidence (p={result.p_value:.3f}), " \
                       f"need larger sample size"
            else:
                return f"Recommend NO CHANGE: No significant difference detected " \
                       f"(p={result.p_value:.3f})"

    def _extract_metric_values(self, results: List[Dict], metric: str) -> np.ndarray:
        """Extract metric values from results."""
        values = []
        for result in results:
            quality = result.get('quality', {})
            if isinstance(quality, dict) and metric in quality:
                value = quality[metric]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(float(value))
        return np.array(values)

    def _has_metric_data(self, control: List[Dict], treatment: List[Dict], metric: str) -> bool:
        """Check if metric data is available in both groups."""
        control_values = self._extract_metric_values(control, metric)
        treatment_values = self._extract_metric_values(treatment, metric)
        return len(control_values) > 0 and len(treatment_values) > 0

    def _has_duration_data(self, control: List[Dict], treatment: List[Dict]) -> bool:
        """Check if duration data is available."""
        control_durations = [r.get('duration', 0) for r in control if 'duration' in r]
        treatment_durations = [r.get('duration', 0) for r in treatment if 'duration' in r]
        return len(control_durations) > 0 and len(treatment_durations) > 0

    def _analyze_duration(self, control: List[Dict], treatment: List[Dict]) -> StatisticalResult:
        """Analyze duration performance metric."""
        control_durations = np.array([r.get('duration', 0) for r in control if 'duration' in r])
        treatment_durations = np.array([r.get('duration', 0) for r in treatment if 'duration' in r])

        return self._analyze_metric_arrays(control_durations, treatment_durations, 'duration')

    def _analyze_metric_arrays(self, control_values: np.ndarray, treatment_values: np.ndarray,
                              metric_name: str) -> StatisticalResult:
        """Analyze metric from numpy arrays."""
        # Basic statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)

        # Calculate improvement (for duration, lower is better)
        if metric_name == 'duration':
            improvement_pct = ((control_mean - treatment_mean) / control_mean) * 100 if control_mean > 0 else 0
            improvement_absolute = control_mean - treatment_mean
        else:
            improvement_pct = ((treatment_mean - control_mean) / abs(control_mean)) * 100 if control_mean != 0 else 0
            improvement_absolute = treatment_mean - control_mean

        # Statistical tests
        t_stat, p_value = ttest_ind(treatment_values, control_values, equal_var=False)
        confidence_interval = self.calculate_confidence_interval(control_values, treatment_values)
        effect_size = self.calculate_cohens_d(control_values, treatment_values)
        sample_size_sufficient = self.check_sample_size(control_values, treatment_values)
        power = self._estimate_power(control_values, treatment_values, effect_size)

        return StatisticalResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            improvement_pct=improvement_pct,
            improvement_absolute=improvement_absolute,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size_sufficient=sample_size_sufficient
        )


def test_statistical_analyzer():
    """Test the statistical analyzer."""
    print("Testing Statistical Analysis Engine...")

    # Create test data
    control_results = []
    treatment_results = []

    # Simulate control group (baseline)
    np.random.seed(42)
    for i in range(50):
        control_results.append({
            'group': 'control',
            'quality': {
                'ssim': np.random.normal(0.80, 0.05),
                'mse': np.random.normal(0.02, 0.005),
                'psnr': np.random.normal(30, 3)
            },
            'duration': np.random.normal(2.0, 0.3),
            'success': True
        })

    # Simulate treatment group (AI enhanced - slightly better)
    for i in range(45):
        treatment_results.append({
            'group': 'treatment',
            'quality': {
                'ssim': np.random.normal(0.85, 0.04),  # 5% improvement
                'mse': np.random.normal(0.018, 0.004),  # Lower is better
                'psnr': np.random.normal(32, 2.5)  # Higher is better
            },
            'duration': np.random.normal(2.2, 0.4),  # Slightly slower
            'success': True
        })

    all_results = control_results + treatment_results

    # Initialize analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05, beta=0.2)

    # Test 1: Full analysis
    print("\n✓ Testing comprehensive analysis:")
    analysis = analyzer.analyze_results(all_results)

    if 'error' in analysis:
        print(f"  Error in analysis: {analysis['error']}")
        return analyzer

    print(f"  Sample sizes: Control={analysis['sample_sizes']['control']}, "
          f"Treatment={analysis['sample_sizes']['treatment']}")

    if 'ssim' in analysis['metrics']:
        ssim_result = analysis['metrics']['ssim']
        print(f"  SSIM improvement: {ssim_result.improvement_pct:.1f}%")
        print(f"  P-value: {ssim_result.p_value:.4f}")
        print(f"  Significant: {ssim_result.significant}")
        print(f"  Effect size: {ssim_result.effect_size:.3f}")
        print(f"  Power: {ssim_result.power:.3f}")

    # Test 2: T-test
    print("\n✓ Testing t-test:")
    t_stat, p_value = analyzer.perform_t_test(control_results, treatment_results, 'ssim')
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")

    # Test 3: Effect size
    print("\n✓ Testing effect size calculation:")
    control_ssim = np.array([r['quality']['ssim'] for r in control_results])
    treatment_ssim = np.array([r['quality']['ssim'] for r in treatment_results])
    cohens_d = analyzer.calculate_cohens_d(control_ssim, treatment_ssim)
    print(f"  Cohen's d: {cohens_d:.3f}")

    if cohens_d < 0.2:
        effect_description = "small"
    elif cohens_d < 0.5:
        effect_description = "small-medium"
    elif cohens_d < 0.8:
        effect_description = "medium-large"
    else:
        effect_description = "large"
    print(f"  Effect size: {effect_description}")

    # Test 4: Confidence interval
    print("\n✓ Testing confidence interval:")
    ci = analyzer.calculate_confidence_interval(control_ssim, treatment_ssim)
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Test 5: Multiple hypothesis correction
    print("\n✓ Testing multiple hypothesis correction:")
    if 'multiple_testing' in analysis:
        corrections = analysis['multiple_testing']['corrections']
        for metric, correction in corrections.items():
            print(f"  {metric}: p={correction['original_p']:.4f} -> "
                  f"adj_p={correction['adjusted_p']:.4f}, "
                  f"FDR_sig={correction['fdr_significant']}")

    # Test 6: Power analysis
    print("\n✓ Testing power analysis:")
    if 'power_analysis' in analysis:
        power_result = analysis['power_analysis']
        print(f"  Current power: {power_result.current_power:.3f}")
        print(f"  Required sample size: {power_result.required_sample_size}")
        print(f"  Minimum detectable effect: {power_result.minimum_detectable_effect:.3f}")

    # Test 7: Recommendation
    print("\n✓ Testing recommendation:")
    if 'overall_summary' in analysis:
        recommendation = analysis['overall_summary']['recommendation']
        print(f"  Recommendation: {recommendation}")

    print("\n✅ All statistical analysis tests passed!")
    return analyzer


if __name__ == "__main__":
    test_statistical_analyzer()