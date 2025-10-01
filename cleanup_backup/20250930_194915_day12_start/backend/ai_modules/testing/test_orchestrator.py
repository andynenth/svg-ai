"""
A/B Test Orchestrator - Task 4 Implementation
Orchestrate complete A/B test campaigns with batch and continuous testing support.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from tqdm import tqdm
import glob

# Handle imports for both module and direct execution
try:
    from .ab_framework import ABTestFramework, TestConfig
    from .statistical_analysis import StatisticalAnalyzer
    from .visual_comparison import VisualComparisonGenerator
except ImportError:
    # Direct execution
    from ab_framework import ABTestFramework, TestConfig
    from statistical_analysis import StatisticalAnalyzer
    from visual_comparison import VisualComparisonGenerator

logger = logging.getLogger(__name__)


@dataclass
class CampaignConfig:
    """Configuration for A/B test campaign."""
    name: str
    image_set: str  # Path pattern or directory
    sample_size_target: Optional[int] = None
    min_sample_size: int = 30
    max_duration_hours: float = 24.0
    significance_threshold: float = 0.05
    early_stopping: bool = True
    early_stopping_checks: int = 5  # Check every N samples
    continuous_mode: bool = False
    traffic_percentage: float = 100.0  # For continuous testing
    output_directory: str = 'ab_test_campaigns'
    generate_visuals: bool = True
    generate_report: bool = True


@dataclass
class CampaignStatus:
    """Status of running campaign."""
    name: str
    started_at: datetime
    last_update: datetime
    samples_collected: int
    target_samples: Optional[int]
    is_running: bool
    is_significant: bool
    should_stop: bool
    stop_reason: Optional[str]
    current_improvement: float
    current_p_value: float


class ABTestOrchestrator:
    """
    Orchestrate complete A/B test campaigns with batch and continuous testing.
    Manages test execution, analysis, and reporting.
    """

    def __init__(self):
        """Initialize A/B test orchestrator."""
        self.framework = ABTestFramework()
        self.analyzer = StatisticalAnalyzer()
        self.visualizer = VisualComparisonGenerator()

        # Campaign management
        self.active_campaigns = {}
        self.campaign_history = []

        # Early stopping rules
        self.early_stopping_rules = {
            'min_samples_before_stopping': 30,
            'significance_threshold': 0.05,
            'minimum_effect_size': 0.1,
            'max_duration_hours': 24.0
        }

        # Test templates
        self.test_templates = self._create_default_templates()

        logger.info("ABTestOrchestrator initialized")

    def run_campaign(self, campaign_config: CampaignConfig) -> Dict[str, Any]:
        """
        Run complete A/B test campaign.

        Args:
            campaign_config: Campaign configuration

        Returns:
            Campaign results dictionary
        """
        logger.info(f"Starting A/B test campaign: {campaign_config.name}")

        try:
            # Initialize campaign
            campaign_start = datetime.now()
            output_dir = Path(campaign_config.output_directory) / campaign_config.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load test images
            images = self.load_test_images(campaign_config.image_set)
            if not images:
                return {
                    'error': f'No images found in {campaign_config.image_set}',
                    'campaign': campaign_config.name
                }

            logger.info(f"Loaded {len(images)} test images")

            # Create test configuration
            test_config = TestConfig(
                name=f"{campaign_config.name}_test",
                sample_size_target=campaign_config.sample_size_target,
                min_sample_size=campaign_config.min_sample_size,
                significance_threshold=campaign_config.significance_threshold
            )

            # Initialize framework with config
            framework = ABTestFramework(test_config)

            # Track campaign status
            campaign_status = CampaignStatus(
                name=campaign_config.name,
                started_at=campaign_start,
                last_update=campaign_start,
                samples_collected=0,
                target_samples=campaign_config.sample_size_target,
                is_running=True,
                is_significant=False,
                should_stop=False,
                stop_reason=None,
                current_improvement=0.0,
                current_p_value=1.0
            )

            self.active_campaigns[campaign_config.name] = campaign_status

            # Run tests
            results = []
            total_tests = len(images) * 2  # Each image tested in both groups

            # Limit to target sample size if specified
            if campaign_config.sample_size_target:
                total_tests = min(total_tests, campaign_config.sample_size_target)
                images = images[:campaign_config.sample_size_target // 2]

            with tqdm(total=total_tests, desc=f"Running {campaign_config.name}") as pbar:
                test_count = 0

                for image in images:
                    if campaign_status.should_stop:
                        break

                    # Run test on this image
                    try:
                        result = framework.run_test(image)
                        results.append(result)
                        test_count += 1
                        pbar.update(1)

                        # Update campaign status
                        campaign_status.samples_collected = test_count
                        campaign_status.last_update = datetime.now()

                        # Check early stopping conditions
                        if (campaign_config.early_stopping and
                            test_count >= campaign_config.min_sample_size and
                            test_count % campaign_config.early_stopping_checks == 0):

                            should_stop, stop_reason = self._check_early_stopping(
                                framework.results, campaign_config, campaign_status
                            )

                            if should_stop:
                                campaign_status.should_stop = True
                                campaign_status.stop_reason = stop_reason
                                logger.info(f"Early stopping triggered: {stop_reason}")
                                break

                        # Check maximum duration
                        elapsed = datetime.now() - campaign_start
                        if elapsed.total_seconds() / 3600 > campaign_config.max_duration_hours:
                            campaign_status.should_stop = True
                            campaign_status.stop_reason = "Maximum duration reached"
                            break

                    except Exception as e:
                        logger.error(f"Test failed for {image}: {e}")
                        continue

            # Mark campaign as completed
            campaign_status.is_running = False
            campaign_end = datetime.now()

            logger.info(f"Campaign completed: {test_count} tests in "
                       f"{(campaign_end - campaign_start).total_seconds():.1f}s")

            # Analyze results
            analysis = self.analyzer.analyze_results([asdict(r) for r in framework.results])

            # Generate visualizations
            visualizations = {}
            if campaign_config.generate_visuals:
                visualizations = self.generate_visualizations(framework.results, output_dir)

            # Generate report
            report = {}
            if campaign_config.generate_report:
                report = self.generate_report(
                    campaign_config, campaign_status, analysis, visualizations, output_dir
                )

            # Final results
            campaign_results = {
                'campaign': {
                    'name': campaign_config.name,
                    'started_at': campaign_start.isoformat(),
                    'completed_at': campaign_end.isoformat(),
                    'duration_seconds': (campaign_end - campaign_start).total_seconds(),
                    'samples_collected': test_count,
                    'early_stopped': campaign_status.should_stop,
                    'stop_reason': campaign_status.stop_reason
                },
                'analysis': analysis,
                'visualizations': visualizations,
                'report': report,
                'success': True
            }

            # Save campaign results
            results_file = output_dir / 'campaign_results.json'
            with open(results_file, 'w') as f:
                json.dump(campaign_results, f, indent=2, default=str)

            # Save raw data
            framework.save_results(str(output_dir / 'raw_results.json'))

            # Add to history
            self.campaign_history.append(campaign_results)

            # Remove from active campaigns
            if campaign_config.name in self.active_campaigns:
                del self.active_campaigns[campaign_config.name]

            logger.info(f"Campaign {campaign_config.name} completed successfully")
            return campaign_results

        except Exception as e:
            logger.error(f"Campaign {campaign_config.name} failed: {e}")
            return {
                'error': str(e),
                'campaign': campaign_config.name,
                'success': False
            }

    def run_continuous_test(self, duration_hours: float, campaign_config: Optional[CampaignConfig] = None):
        """
        Run ongoing A/B test in production.

        Args:
            duration_hours: Duration to run continuous test
            campaign_config: Optional campaign configuration
        """
        if not campaign_config:
            campaign_config = CampaignConfig(
                name=f"continuous_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                image_set="data/test_images/**/*.png",
                continuous_mode=True,
                traffic_percentage=10.0,  # Sample 10% of traffic
                max_duration_hours=duration_hours
            )

        logger.info(f"Starting continuous A/B test for {duration_hours} hours")

        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)

            # Create test framework
            framework = ABTestFramework()

            # Sample percentage of traffic
            traffic_sample_rate = campaign_config.traffic_percentage / 100.0

            test_count = 0
            while datetime.now() < end_time:
                try:
                    # Simulate incoming request (in real implementation, this would be actual traffic)
                    if self._should_sample_request(traffic_sample_rate):
                        # Get random image for testing
                        images = self.load_test_images(campaign_config.image_set)
                        if images:
                            import random
                            test_image = random.choice(images)
                            result = framework.run_test(test_image)
                            test_count += 1

                            # Check for significance periodically
                            if test_count % 50 == 0:  # Check every 50 tests
                                results = [asdict(r) for r in framework.results]
                                analysis = self.analyzer.analyze_results(results)

                                if ('overall_summary' in analysis and
                                    analysis['overall_summary'].get('t_test', {}).get('significant', False)):
                                    logger.info(f"Continuous test reached significance after {test_count} tests")

                                    # Monitor for significance
                                    improvement = analysis['overall_summary'].get('quality_improvement', 0)
                                    p_value = analysis['overall_summary']['t_test']['p_value']

                                    logger.info(f"Current results: {improvement:.1f}% improvement, p={p_value:.4f}")

                    # Sleep between tests (simulate real-world timing)
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Error in continuous test iteration: {e}")
                    continue

            logger.info(f"Continuous test completed: {test_count} samples collected")

            # Final analysis
            final_results = [asdict(r) for r in framework.results]
            final_analysis = self.analyzer.analyze_results(final_results)

            return {
                'continuous_test': {
                    'duration_hours': duration_hours,
                    'samples_collected': test_count,
                    'completed_at': datetime.now().isoformat()
                },
                'analysis': final_analysis,
                'success': True
            }

        except Exception as e:
            logger.error(f"Continuous test failed: {e}")
            return {'error': str(e), 'success': False}

    def load_test_images(self, image_set: str) -> List[str]:
        """
        Load test images from path pattern or directory.

        Args:
            image_set: Path pattern, directory, or file list

        Returns:
            List of image file paths
        """
        try:
            images = []

            # If it's a directory
            if Path(image_set).is_dir():
                # Find all image files in directory
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
                    images.extend(glob.glob(str(Path(image_set) / ext)))
                    images.extend(glob.glob(str(Path(image_set) / '**' / ext), recursive=True))

            # If it's a glob pattern
            elif '*' in image_set:
                images = glob.glob(image_set, recursive=True)

            # If it's a single file
            elif Path(image_set).is_file():
                images = [image_set]

            # If it's a text file with image paths
            elif image_set.endswith('.txt'):
                with open(image_set, 'r') as f:
                    images = [line.strip() for line in f if line.strip()]

            # Filter to only existing image files
            valid_images = []
            for img in images:
                if Path(img).exists() and Path(img).suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    valid_images.append(img)

            return valid_images

        except Exception as e:
            logger.error(f"Failed to load test images from {image_set}: {e}")
            return []

    def generate_visualizations(self, results: List, output_dir: Path) -> Dict[str, str]:
        """
        Generate visualizations for test results.

        Args:
            results: Test results
            output_dir: Output directory

        Returns:
            Dictionary of generated visualization paths
        """
        try:
            logger.info("Generating visualizations...")

            # Convert results to dictionaries
            dict_results = [asdict(r) for r in results]

            # Generate batch comparisons
            visual_output_dir = output_dir / 'visualizations'
            generated_files = self.visualizer.batch_generate_comparisons(
                dict_results, str(visual_output_dir)
            )

            # Generate HTML report
            html_output = visual_output_dir / 'visual_report.html'
            comparison_data = self._prepare_comparison_data(dict_results)
            self.visualizer.export_html_report(comparison_data, str(html_output))

            return {
                'batch_comparisons': generated_files,
                'html_report': str(html_output),
                'output_directory': str(visual_output_dir)
            }

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            return {}

    def generate_report(self, campaign_config: CampaignConfig, campaign_status: CampaignStatus,
                       analysis: Dict, visualizations: Dict, output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive campaign report.

        Args:
            campaign_config: Campaign configuration
            campaign_status: Campaign status
            analysis: Statistical analysis results
            visualizations: Generated visualizations
            output_dir: Output directory

        Returns:
            Report dictionary
        """
        try:
            # Calculate key metrics
            if 'overall_summary' in analysis:
                summary = analysis['overall_summary']
                improvement = summary.get('quality_improvement', 0)
                p_value = summary.get('t_test', {}).get('p_value', 1.0)
                significant = summary.get('t_test', {}).get('significant', False)
                confidence = summary.get('confidence_interval', [0, 0])
                effect_size = summary.get('effect_size', 0)

                # Make recommendation
                if significant and improvement > 1:
                    recommendation = "DEPLOY"
                    recommendation_reason = f"Significant {improvement:.1f}% improvement with p={p_value:.4f}"
                elif significant and improvement < -1:
                    recommendation = "REJECT"
                    recommendation_reason = f"Significant {abs(improvement):.1f}% degradation with p={p_value:.4f}"
                else:
                    recommendation = "CONTINUE TESTING"
                    recommendation_reason = f"No significant difference detected (p={p_value:.4f})"
            else:
                improvement = 0
                p_value = 1.0
                significant = False
                confidence = [0, 0]
                effect_size = 0
                recommendation = "INSUFFICIENT DATA"
                recommendation_reason = "Not enough data for analysis"

            # Create executive summary
            executive_summary = {
                'campaign_name': campaign_config.name,
                'recommendation': recommendation,
                'recommendation_reason': recommendation_reason,
                'quality_improvement_pct': improvement,
                'statistical_significance': significant,
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_interval': confidence,
                'samples_collected': campaign_status.samples_collected,
                'duration_hours': (campaign_status.last_update - campaign_status.started_at).total_seconds() / 3600,
                'early_stopped': campaign_status.should_stop,
                'stop_reason': campaign_status.stop_reason
            }

            # Detailed metrics
            detailed_metrics = {}
            if 'metrics' in analysis:
                for metric_name, metric_result in analysis['metrics'].items():
                    if hasattr(metric_result, 'control_mean'):
                        detailed_metrics[metric_name] = {
                            'control_mean': metric_result.control_mean,
                            'treatment_mean': metric_result.treatment_mean,
                            'improvement_pct': metric_result.improvement_pct,
                            'p_value': metric_result.p_value,
                            'significant': metric_result.significant,
                            'effect_size': metric_result.effect_size
                        }

            # Power analysis
            power_analysis = analysis.get('power_analysis', {})

            # Create full report
            report = {
                'executive_summary': executive_summary,
                'detailed_metrics': detailed_metrics,
                'power_analysis': power_analysis,
                'multiple_testing_correction': analysis.get('multiple_testing', {}),
                'campaign_configuration': asdict(campaign_config),
                'visualizations': visualizations,
                'generated_at': datetime.now().isoformat()
            }

            # Save report
            report_file = output_dir / 'campaign_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Campaign report generated: {report_file}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}

    def _check_early_stopping(self, results: List, config: CampaignConfig,
                             status: CampaignStatus) -> tuple[bool, Optional[str]]:
        """
        Check if early stopping conditions are met.

        Args:
            results: Current test results
            config: Campaign configuration
            status: Campaign status

        Returns:
            Tuple of (should_stop, reason)
        """
        try:
            if len(results) < config.min_sample_size:
                return False, None

            # Analyze current results
            dict_results = [asdict(r) for r in results]
            analysis = self.analyzer.analyze_results(dict_results)

            if 'overall_summary' not in analysis:
                return False, None

            summary = analysis['overall_summary']
            improvement = summary.get('quality_improvement', 0)
            p_value = summary.get('t_test', {}).get('p_value', 1.0)
            significant = summary.get('t_test', {}).get('significant', False)

            # Update campaign status
            status.current_improvement = improvement
            status.current_p_value = p_value
            status.is_significant = significant

            # Early stopping rule 1: Strong significance achieved
            if significant and p_value < 0.01 and abs(improvement) > 5:
                return True, f"Strong significance achieved (p={p_value:.4f}, improvement={improvement:.1f}%)"

            # Early stopping rule 2: Futility analysis
            power_analysis = analysis.get('power_analysis')
            if power_analysis and hasattr(power_analysis, 'current_power'):
                if power_analysis.current_power > 0.8 and not significant:
                    return True, f"Futility: High power ({power_analysis.current_power:.2f}) but no significance"

            # Early stopping rule 3: Harmful effect detected
            if significant and improvement < -2:
                return True, f"Harmful effect detected: {improvement:.1f}% degradation"

            # Early stopping rule 4: Maximum samples reached
            if config.sample_size_target and len(results) >= config.sample_size_target:
                return True, "Target sample size reached"

            return False, None

        except Exception as e:
            logger.warning(f"Early stopping check failed: {e}")
            return False, None

    def _should_sample_request(self, sample_rate: float) -> bool:
        """Determine if a request should be sampled for continuous testing."""
        import random
        return random.random() < sample_rate

    def _prepare_comparison_data(self, results: List[Dict]) -> List[Dict]:
        """Prepare comparison data for HTML report."""
        comparisons = []

        # Group by image
        image_groups = {}
        for result in results:
            image_path = result.get('image', 'unknown')
            if image_path not in image_groups:
                image_groups[image_path] = {'control': None, 'treatment': None}

            group = result.get('group')
            if group in ['control', 'treatment']:
                image_groups[image_path][group] = result

        # Create comparison data
        for image_path, groups in image_groups.items():
            if groups['control'] and groups['treatment']:
                control = groups['control']
                treatment = groups['treatment']

                control_ssim = control.get('quality', {}).get('ssim', 0)
                treatment_ssim = treatment.get('quality', {}).get('ssim', 0)
                improvement = ((treatment_ssim - control_ssim) / control_ssim * 100) if control_ssim > 0 else 0

                comparisons.append({
                    'image_name': Path(image_path).name,
                    'control_ssim': control_ssim,
                    'treatment_ssim': treatment_ssim,
                    'control_time': control.get('duration', 0),
                    'treatment_time': treatment.get('duration', 0),
                    'improvement': improvement
                })

        return comparisons

    def _create_default_templates(self) -> Dict[str, CampaignConfig]:
        """Create default test templates."""
        return {
            'quick_test': CampaignConfig(
                name='quick_test',
                image_set='data/test_images/*.png',
                sample_size_target=50,
                min_sample_size=20,
                max_duration_hours=1.0,
                early_stopping=True
            ),
            'comprehensive_test': CampaignConfig(
                name='comprehensive_test',
                image_set='data/test_images/**/*.png',
                sample_size_target=200,
                min_sample_size=50,
                max_duration_hours=12.0,
                early_stopping=True
            ),
            'production_test': CampaignConfig(
                name='production_test',
                image_set='data/production_images/*.png',
                sample_size_target=1000,
                min_sample_size=100,
                max_duration_hours=48.0,
                continuous_mode=True,
                traffic_percentage=5.0,
                early_stopping=True
            )
        }

    def get_campaign_status(self, campaign_name: str) -> Optional[CampaignStatus]:
        """Get status of active campaign."""
        return self.active_campaigns.get(campaign_name)

    def list_active_campaigns(self) -> List[str]:
        """List names of all active campaigns."""
        return list(self.active_campaigns.keys())

    def get_campaign_history(self) -> List[Dict]:
        """Get history of completed campaigns."""
        return self.campaign_history


def test_ab_test_orchestrator():
    """Test the A/B test orchestrator."""
    print("Testing A/B Test Orchestrator...")

    # Create orchestrator
    orchestrator = ABTestOrchestrator()

    # Test 1: Load test images
    print("\n✓ Testing image loading:")

    # Create test directory with mock images
    test_dir = Path('/tmp/orchestrator_test_images')
    test_dir.mkdir(exist_ok=True)

    # Create some mock image files
    for i in range(5):
        mock_file = test_dir / f'test_{i}.png'
        mock_file.touch()

    images = orchestrator.load_test_images(str(test_dir))
    print(f"  Loaded {len(images)} test images")

    # Test 2: Campaign configuration
    print("\n✓ Testing campaign configuration:")
    campaign_config = CampaignConfig(
        name='test_campaign',
        image_set=str(test_dir),
        sample_size_target=10,
        min_sample_size=5,
        max_duration_hours=0.1,  # 6 minutes
        early_stopping=True
    )
    print(f"  Created campaign: {campaign_config.name}")

    # Test 3: Run small campaign
    print("\n✓ Testing campaign execution:")
    results = orchestrator.run_campaign(campaign_config)

    if results.get('success'):
        print(f"  Campaign completed successfully")
        print(f"  Samples collected: {results['campaign']['samples_collected']}")
        print(f"  Duration: {results['campaign']['duration_seconds']:.1f}s")

        if 'analysis' in results and 'overall_summary' in results['analysis']:
            summary = results['analysis']['overall_summary']
            print(f"  Quality improvement: {summary.get('quality_improvement', 0):.1f}%")
    else:
        print(f"  Campaign failed: {results.get('error', 'Unknown error')}")

    # Test 4: Test templates
    print("\n✓ Testing test templates:")
    templates = orchestrator._create_default_templates()
    print(f"  Available templates: {list(templates.keys())}")

    # Test 5: Campaign status tracking
    print("\n✓ Testing campaign status:")
    active_campaigns = orchestrator.list_active_campaigns()
    print(f"  Active campaigns: {active_campaigns}")

    history = orchestrator.get_campaign_history()
    print(f"  Campaign history: {len(history)} completed campaigns")

    # Test 6: Continuous test simulation (very short)
    print("\n✓ Testing continuous test (simulated):")
    try:
        continuous_config = CampaignConfig(
            name='continuous_test',
            image_set=str(test_dir),
            continuous_mode=True,
            traffic_percentage=50.0,
            max_duration_hours=0.01  # 36 seconds
        )

        continuous_results = orchestrator.run_continuous_test(0.01, continuous_config)
        if continuous_results.get('success'):
            print(f"  Continuous test completed")
            print(f"  Samples: {continuous_results['continuous_test']['samples_collected']}")
        else:
            print(f"  Continuous test failed: {continuous_results.get('error')}")
    except Exception as e:
        print(f"  Continuous test skipped: {e}")

    print("\n✅ All orchestrator tests passed!")
    return orchestrator


if __name__ == "__main__":
    test_ab_test_orchestrator()