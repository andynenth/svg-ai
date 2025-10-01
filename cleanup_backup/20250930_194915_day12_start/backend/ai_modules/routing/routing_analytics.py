"""
Routing Performance Analytics - Task 4 Implementation
Track and analyze routing decisions and performance metrics.
"""

import logging
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Container for a routing decision."""
    timestamp: float
    image_path: str
    complexity: float
    target_quality: float
    selected_tier: int
    original_tier: int
    time_budget: Optional[float]
    user_preference: Optional[str]
    confidence: float
    reasoning: str


@dataclass
class RoutingOutcome:
    """Container for routing outcome."""
    decision_id: str
    actual_tier: int
    processing_time: float
    achieved_quality: float
    success: bool
    error: Optional[str] = None


class RoutingAnalytics:
    """
    Analytics system for tracking and analyzing routing performance.
    Provides insights into routing effectiveness and optimization opportunities.
    """

    def __init__(self, history_size: int = 10000):
        """
        Initialize routing analytics.

        Args:
            history_size: Maximum size of history to maintain
        """
        # Routing decision history
        self.routing_decisions = deque(maxlen=history_size)
        self.routing_outcomes = {}  # decision_id -> outcome

        # Performance tracking by tier
        self.tier_performance = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0,
            'total_quality': 0,
            'quality_targets_met': 0,
            'time_targets_met': 0,
            'downgrades': 0,
            'upgrades': 0
        })

        # Quality achievement tracking
        self.quality_achievement = defaultdict(list)  # tier -> list of (target, achieved)

        # Time compliance tracking
        self.time_compliance = defaultdict(list)  # tier -> list of (budget, actual)

        # Routing patterns
        self.routing_patterns = defaultdict(int)  # (original_tier, selected_tier) -> count

        # Optimization opportunities
        self.optimization_opportunities = []

        # Real-time metrics
        self.current_metrics = {
            'total_decisions': 0,
            'avg_confidence': 0,
            'tier_distribution': {1: 0, 2: 0, 3: 0},
            'success_rate': 0,
            'avg_quality_achievement': 0,
            'avg_time_compliance': 0
        }

        # Thread safety
        self._lock = threading.RLock()

        logger.info("RoutingAnalytics initialized")

    def record_routing_decision(self,
                               image_path: str,
                               complexity: float,
                               target_quality: float,
                               selected_tier: int,
                               original_tier: int,
                               confidence: float,
                               reasoning: str,
                               time_budget: Optional[float] = None,
                               user_preference: Optional[str] = None) -> str:
        """
        Record a routing decision.

        Args:
            image_path: Path to image
            complexity: Complexity score
            target_quality: Target quality
            selected_tier: Selected tier
            original_tier: Originally requested tier
            confidence: Confidence in decision
            reasoning: Reasoning for decision
            time_budget: Time budget if any
            user_preference: User preference if any

        Returns:
            Decision ID for tracking outcome
        """
        with self._lock:
            decision = RoutingDecision(
                timestamp=time.time(),
                image_path=image_path,
                complexity=complexity,
                target_quality=target_quality,
                selected_tier=selected_tier,
                original_tier=original_tier,
                time_budget=time_budget,
                user_preference=user_preference,
                confidence=confidence,
                reasoning=reasoning
            )

            self.routing_decisions.append(decision)

            # Update routing patterns
            pattern = (original_tier, selected_tier)
            self.routing_patterns[pattern] += 1

            # Update tier statistics
            if selected_tier < original_tier:
                self.tier_performance[selected_tier]['downgrades'] += 1
            elif selected_tier > original_tier:
                self.tier_performance[selected_tier]['upgrades'] += 1

            # Generate decision ID
            decision_id = f"{int(decision.timestamp)}_{hash(image_path) % 10000}"

            # Update current metrics
            self._update_current_metrics()

            logger.debug(f"Recorded routing decision {decision_id}: Tier {selected_tier}")

            return decision_id

    def record_routing_outcome(self,
                             decision_id: str,
                             actual_tier: int,
                             processing_time: float,
                             achieved_quality: float,
                             success: bool,
                             error: Optional[str] = None):
        """
        Record the outcome of a routing decision.

        Args:
            decision_id: Decision ID from record_routing_decision
            actual_tier: Tier that actually processed the request
            processing_time: Actual processing time
            achieved_quality: Quality achieved
            success: Whether processing was successful
            error: Error message if failed
        """
        with self._lock:
            outcome = RoutingOutcome(
                decision_id=decision_id,
                actual_tier=actual_tier,
                processing_time=processing_time,
                achieved_quality=achieved_quality,
                success=success,
                error=error
            )

            self.routing_outcomes[decision_id] = outcome

            # Update tier performance
            perf = self.tier_performance[actual_tier]
            perf['total_requests'] += 1

            if success:
                perf['successful_requests'] += 1
                perf['total_processing_time'] += processing_time
                perf['total_quality'] += achieved_quality

                # Find corresponding decision
                decision = self._find_decision_by_id(decision_id)
                if decision:
                    # Check quality target
                    if achieved_quality >= decision.target_quality:
                        perf['quality_targets_met'] += 1

                    # Check time target
                    if decision.time_budget and processing_time <= decision.time_budget:
                        perf['time_targets_met'] += 1

                    # Track quality achievement
                    self.quality_achievement[actual_tier].append(
                        (decision.target_quality, achieved_quality)
                    )

                    # Track time compliance
                    if decision.time_budget:
                        self.time_compliance[actual_tier].append(
                            (decision.time_budget, processing_time)
                        )
            else:
                perf['failed_requests'] += 1

            # Identify optimization opportunities
            self._identify_optimization_opportunity(outcome)

            logger.debug(f"Recorded routing outcome {decision_id}: "
                       f"success={success}, quality={achieved_quality:.3f}")

    def _find_decision_by_id(self, decision_id: str) -> Optional[RoutingDecision]:
        """Find decision by ID based on timestamp match."""
        try:
            timestamp = int(decision_id.split('_')[0])
            for decision in self.routing_decisions:
                if int(decision.timestamp) == timestamp:
                    return decision
        except:
            pass
        return None

    def _identify_optimization_opportunity(self, outcome: RoutingOutcome):
        """Identify potential optimization opportunities from outcomes."""
        decision = self._find_decision_by_id(outcome.decision_id)
        if not decision:
            return

        opportunities = []

        # Check if tier was over-provisioned
        if outcome.success and decision.selected_tier > 1:
            if outcome.achieved_quality > decision.target_quality * 1.2:
                opportunities.append({
                    'type': 'over_provisioning',
                    'message': f"Tier {decision.selected_tier} achieved {outcome.achieved_quality:.2%} "
                             f"quality when only {decision.target_quality:.2%} was needed",
                    'recommendation': f"Consider using Tier {decision.selected_tier - 1}"
                })

        # Check if time budget was exceeded
        if decision.time_budget and outcome.processing_time > decision.time_budget:
            opportunities.append({
                'type': 'time_budget_exceeded',
                'message': f"Processing took {outcome.processing_time:.1f}s but budget was "
                         f"{decision.time_budget:.1f}s",
                'recommendation': "Consider stricter tier selection for time-critical requests"
            })

        # Check if quality target was missed
        if outcome.achieved_quality < decision.target_quality:
            opportunities.append({
                'type': 'quality_target_missed',
                'message': f"Achieved {outcome.achieved_quality:.2%} quality but target was "
                         f"{decision.target_quality:.2%}",
                'recommendation': f"Consider using Tier {min(3, decision.selected_tier + 1)}"
            })

        for opp in opportunities:
            opp['timestamp'] = time.time()
            opp['decision_id'] = outcome.decision_id
            self.optimization_opportunities.append(opp)

    def analyze_routing_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze overall routing effectiveness.

        Returns:
            Dictionary with effectiveness metrics
        """
        with self._lock:
            # Calculate tier accuracy
            tier_accuracy = self.calculate_tier_accuracy()

            # Calculate quality achievement
            quality_achievement = self.calculate_quality_achievement()

            # Calculate time compliance
            time_compliance = self.calculate_time_compliance()

            # Calculate recommendation quality
            recommendation_quality = self.calculate_recommendation_quality()

            return {
                'tier_accuracy': tier_accuracy,
                'quality_achievement': quality_achievement,
                'time_compliance': time_compliance,
                'recommendation_quality': recommendation_quality,
                'total_decisions': len(self.routing_decisions),
                'total_outcomes': len(self.routing_outcomes)
            }

    def calculate_tier_accuracy(self) -> Dict[str, float]:
        """
        Calculate how accurately tiers are selected.

        Returns:
            Accuracy metrics by tier
        """
        accuracy = {}

        for tier, perf in self.tier_performance.items():
            if perf['total_requests'] > 0:
                # Success rate
                success_rate = perf['successful_requests'] / perf['total_requests']

                # Quality target achievement
                quality_accuracy = (perf['quality_targets_met'] / perf['successful_requests']
                                  if perf['successful_requests'] > 0 else 0)

                # Time target achievement
                time_accuracy = (perf['time_targets_met'] / perf['successful_requests']
                               if perf['successful_requests'] > 0 else 0)

                accuracy[f"tier_{tier}"] = {
                    'success_rate': success_rate,
                    'quality_accuracy': quality_accuracy,
                    'time_accuracy': time_accuracy,
                    'overall_accuracy': (success_rate + quality_accuracy + time_accuracy) / 3
                }

        return accuracy

    def calculate_quality_achievement(self) -> Dict[str, float]:
        """
        Calculate quality achievement statistics.

        Returns:
            Quality achievement metrics
        """
        achievement = {}

        for tier, qualities in self.quality_achievement.items():
            if qualities:
                targets = [t for t, _ in qualities]
                achieved = [a for _, a in qualities]

                achievement[f"tier_{tier}"] = {
                    'avg_target': np.mean(targets),
                    'avg_achieved': np.mean(achieved),
                    'achievement_ratio': np.mean([a/t for t, a in qualities if t > 0]),
                    'targets_met': sum(1 for t, a in qualities if a >= t) / len(qualities)
                }

        # Overall achievement
        all_qualities = []
        for qualities in self.quality_achievement.values():
            all_qualities.extend(qualities)

        if all_qualities:
            achievement['overall'] = {
                'avg_achievement_ratio': np.mean([a/t for t, a in all_qualities if t > 0]),
                'targets_met_ratio': sum(1 for t, a in all_qualities if a >= t) / len(all_qualities)
            }

        return achievement

    def calculate_time_compliance(self) -> Dict[str, float]:
        """
        Calculate time budget compliance statistics.

        Returns:
            Time compliance metrics
        """
        compliance = {}

        for tier, times in self.time_compliance.items():
            if times:
                budgets = [b for b, _ in times]
                actuals = [a for _, a in times]

                compliance[f"tier_{tier}"] = {
                    'avg_budget': np.mean(budgets),
                    'avg_actual': np.mean(actuals),
                    'compliance_ratio': np.mean([b/a for b, a in times if a > 0]),
                    'budgets_met': sum(1 for b, a in times if a <= b) / len(times)
                }

        # Overall compliance
        all_times = []
        for times in self.time_compliance.values():
            all_times.extend(times)

        if all_times:
            compliance['overall'] = {
                'compliance_ratio': np.mean([b/a for b, a in all_times if a > 0]),
                'budgets_met_ratio': sum(1 for b, a in all_times if a <= b) / len(all_times)
            }

        return compliance

    def calculate_recommendation_quality(self) -> float:
        """
        Calculate quality of routing recommendations.

        Returns:
            Recommendation quality score (0-1)
        """
        if not self.routing_outcomes:
            return 0.0

        scores = []

        for decision_id, outcome in self.routing_outcomes.items():
            decision = self._find_decision_by_id(decision_id)
            if not decision or not outcome.success:
                continue

            # Score based on:
            # 1. Quality achievement
            quality_score = min(1.0, outcome.achieved_quality / decision.target_quality)

            # 2. Time compliance (if applicable)
            if decision.time_budget:
                time_score = min(1.0, decision.time_budget / outcome.processing_time)
            else:
                time_score = 1.0

            # 3. Tier selection efficiency (penalty for over-provisioning)
            tier_efficiency = 1.0 - (outcome.actual_tier - 1) * 0.2

            # Combined score
            score = (quality_score * 0.5 + time_score * 0.3 + tier_efficiency * 0.2)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def generate_routing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive routing analytics report.

        Returns:
            Complete analytics report
        """
        with self._lock:
            # Analyze effectiveness
            effectiveness = self.analyze_routing_effectiveness()

            # Tier usage distribution
            tier_distribution = self._calculate_tier_distribution()

            # Success rates by tier
            success_rates = self._calculate_success_rates()

            # Quality vs time tradeoffs
            tradeoffs = self._analyze_tradeoffs()

            # Top optimization opportunities
            top_opportunities = self._get_top_opportunities(10)

            # Routing pattern analysis
            pattern_analysis = self._analyze_routing_patterns()

            report = {
                'timestamp': time.time(),
                'period': {
                    'start': min(d.timestamp for d in self.routing_decisions) if self.routing_decisions else 0,
                    'end': max(d.timestamp for d in self.routing_decisions) if self.routing_decisions else 0,
                    'duration_hours': ((max(d.timestamp for d in self.routing_decisions) -
                                      min(d.timestamp for d in self.routing_decisions)) / 3600
                                     if len(self.routing_decisions) > 1 else 0)
                },
                'summary': {
                    'total_decisions': len(self.routing_decisions),
                    'total_outcomes': len(self.routing_outcomes),
                    'overall_success_rate': self._calculate_overall_success_rate(),
                    'avg_confidence': np.mean([d.confidence for d in self.routing_decisions])
                                     if self.routing_decisions else 0
                },
                'effectiveness': effectiveness,
                'tier_distribution': tier_distribution,
                'success_rates': success_rates,
                'quality_time_tradeoffs': tradeoffs,
                'optimization_opportunities': top_opportunities,
                'routing_patterns': pattern_analysis,
                'recommendations': self._generate_recommendations()
            }

            return report

    def _calculate_tier_distribution(self) -> Dict[str, float]:
        """Calculate distribution of requests across tiers."""
        distribution = {1: 0, 2: 0, 3: 0}

        for decision in self.routing_decisions:
            distribution[decision.selected_tier] += 1

        total = sum(distribution.values())
        if total > 0:
            return {f"tier_{k}": v/total for k, v in distribution.items()}
        return {f"tier_{k}": 0 for k in [1, 2, 3]}

    def _calculate_success_rates(self) -> Dict[str, float]:
        """Calculate success rates by tier."""
        rates = {}

        for tier, perf in self.tier_performance.items():
            if perf['total_requests'] > 0:
                rates[f"tier_{tier}"] = perf['successful_requests'] / perf['total_requests']
            else:
                rates[f"tier_{tier}"] = 0.0

        return rates

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_requests = sum(p['total_requests'] for p in self.tier_performance.values())
        total_successes = sum(p['successful_requests'] for p in self.tier_performance.values())

        return total_successes / total_requests if total_requests > 0 else 0.0

    def _analyze_tradeoffs(self) -> Dict[str, Any]:
        """Analyze quality vs time tradeoffs."""
        tradeoffs = {
            'tier_1': {'avg_quality': 0, 'avg_time': 0, 'samples': 0},
            'tier_2': {'avg_quality': 0, 'avg_time': 0, 'samples': 0},
            'tier_3': {'avg_quality': 0, 'avg_time': 0, 'samples': 0}
        }

        for tier, perf in self.tier_performance.items():
            if perf['successful_requests'] > 0:
                tradeoffs[f'tier_{tier}'] = {
                    'avg_quality': perf['total_quality'] / perf['successful_requests'],
                    'avg_time': perf['total_processing_time'] / perf['successful_requests'],
                    'samples': perf['successful_requests']
                }

        # Calculate efficiency scores (quality per second)
        for tier_key, metrics in tradeoffs.items():
            if metrics['avg_time'] > 0:
                metrics['efficiency'] = metrics['avg_quality'] / metrics['avg_time']
            else:
                metrics['efficiency'] = 0

        return tradeoffs

    def _get_top_opportunities(self, n: int) -> List[Dict[str, Any]]:
        """Get top N optimization opportunities."""
        # Sort by recency
        sorted_opportunities = sorted(
            self.optimization_opportunities,
            key=lambda x: x['timestamp'],
            reverse=True
        )

        # Group by type and count
        by_type = defaultdict(list)
        for opp in sorted_opportunities[:100]:  # Look at last 100
            by_type[opp['type']].append(opp)

        # Create summary
        top_opportunities = []
        for opp_type, opportunities in by_type.items():
            if opportunities:
                top_opportunities.append({
                    'type': opp_type,
                    'count': len(opportunities),
                    'recent_example': opportunities[0]['message'],
                    'recommendation': opportunities[0]['recommendation']
                })

        return sorted(top_opportunities, key=lambda x: x['count'], reverse=True)[:n]

    def _analyze_routing_patterns(self) -> Dict[str, Any]:
        """Analyze routing patterns."""
        patterns = {
            'upgrades': 0,
            'downgrades': 0,
            'unchanged': 0,
            'pattern_details': {}
        }

        for (original, selected), count in self.routing_patterns.items():
            if selected > original:
                patterns['upgrades'] += count
            elif selected < original:
                patterns['downgrades'] += count
            else:
                patterns['unchanged'] += count

            patterns['pattern_details'][f"{original}->{selected}"] = count

        total = patterns['upgrades'] + patterns['downgrades'] + patterns['unchanged']
        if total > 0:
            patterns['upgrade_rate'] = patterns['upgrades'] / total
            patterns['downgrade_rate'] = patterns['downgrades'] / total

        return patterns

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analytics."""
        recommendations = []

        # Check success rates
        success_rates = self._calculate_success_rates()
        for tier_key, rate in success_rates.items():
            if rate < 0.9:
                tier = tier_key.split('_')[1]
                recommendations.append(
                    f"Tier {tier} has low success rate ({rate:.1%}). "
                    f"Consider adjusting selection criteria."
                )

        # Check quality achievement
        quality_achievement = self.calculate_quality_achievement()
        if 'overall' in quality_achievement:
            if quality_achievement['overall']['targets_met_ratio'] < 0.8:
                recommendations.append(
                    "Quality targets are frequently missed. "
                    "Consider more conservative tier selection."
                )

        # Check time compliance
        time_compliance = self.calculate_time_compliance()
        if 'overall' in time_compliance:
            if time_compliance['overall']['budgets_met_ratio'] < 0.8:
                recommendations.append(
                    "Time budgets are frequently exceeded. "
                    "Consider stricter time-based routing."
                )

        # Check for over-provisioning
        for tier in [2, 3]:
            perf = self.tier_performance[tier]
            if perf['successful_requests'] > 10:
                avg_quality = perf['total_quality'] / perf['successful_requests']
                # If achieving much higher quality than needed
                qualities = self.quality_achievement.get(tier, [])
                if qualities:
                    avg_target = np.mean([t for t, _ in qualities])
                    if avg_quality > avg_target * 1.15:
                        recommendations.append(
                            f"Tier {tier} consistently exceeds quality targets. "
                            f"Consider using lower tiers for some requests."
                        )

        # Check routing patterns
        patterns = self._analyze_routing_patterns()
        if patterns.get('downgrade_rate', 0) > 0.2:
            recommendations.append(
                f"High downgrade rate ({patterns['downgrade_rate']:.1%}). "
                f"Initial tier selection may be too optimistic."
            )

        return recommendations[:5]  # Return top 5 recommendations

    def _update_current_metrics(self):
        """Update current real-time metrics."""
        if not self.routing_decisions:
            return

        self.current_metrics['total_decisions'] = len(self.routing_decisions)

        # Average confidence
        confidences = [d.confidence for d in self.routing_decisions]
        self.current_metrics['avg_confidence'] = np.mean(confidences) if confidences else 0

        # Tier distribution
        for tier in [1, 2, 3]:
            count = sum(1 for d in self.routing_decisions if d.selected_tier == tier)
            self.current_metrics['tier_distribution'][tier] = count / len(self.routing_decisions)

        # Success rate
        self.current_metrics['success_rate'] = self._calculate_overall_success_rate()

        # Quality and time metrics
        quality_achievement = self.calculate_quality_achievement()
        if 'overall' in quality_achievement:
            self.current_metrics['avg_quality_achievement'] = quality_achievement['overall']['avg_achievement_ratio']

        time_compliance = self.calculate_time_compliance()
        if 'overall' in time_compliance:
            self.current_metrics['avg_time_compliance'] = time_compliance['overall']['compliance_ratio']

    def export_to_json(self, filepath: str):
        """Export analytics to JSON file."""
        report = self.generate_routing_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Exported routing analytics to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        with self._lock:
            return {
                'current_metrics': self.current_metrics,
                'tier_performance': dict(self.tier_performance),
                'optimization_opportunities_count': len(self.optimization_opportunities),
                'routing_patterns': dict(self.routing_patterns)
            }


def test_routing_analytics():
    """Test the routing analytics system."""
    print("Testing Routing Analytics...")

    # Initialize analytics
    analytics = RoutingAnalytics()

    # Simulate routing decisions and outcomes
    print("\n✓ Simulating routing decisions:")
    decision_ids = []

    for i in range(20):
        # Generate random decision
        complexity = np.random.random()
        target_quality = 0.7 + np.random.random() * 0.25
        original_tier = np.random.randint(1, 4)
        selected_tier = np.random.randint(1, 4)
        confidence = 0.6 + np.random.random() * 0.4
        time_budget = np.random.uniform(2, 10) if np.random.random() < 0.5 else None

        decision_id = analytics.record_routing_decision(
            image_path=f"test_{i}.png",
            complexity=complexity,
            target_quality=target_quality,
            selected_tier=selected_tier,
            original_tier=original_tier,
            confidence=confidence,
            reasoning=f"Test decision {i}",
            time_budget=time_budget
        )

        decision_ids.append((decision_id, selected_tier, target_quality, time_budget))

    print(f"  Recorded {len(decision_ids)} routing decisions")

    # Record outcomes
    print("\n✓ Recording routing outcomes:")
    for decision_id, tier, target_quality, time_budget in decision_ids:
        # Simulate outcome
        success = np.random.random() < 0.9  # 90% success rate
        processing_time = np.random.uniform(0.5, 15)
        achieved_quality = target_quality + np.random.uniform(-0.1, 0.1)
        achieved_quality = max(0.5, min(1.0, achieved_quality))

        analytics.record_routing_outcome(
            decision_id=decision_id,
            actual_tier=tier,
            processing_time=processing_time,
            achieved_quality=achieved_quality,
            success=success
        )

    print(f"  Recorded {len(decision_ids)} routing outcomes")

    # Analyze effectiveness
    print("\n✓ Analyzing routing effectiveness:")
    effectiveness = analytics.analyze_routing_effectiveness()
    print(f"  Tier accuracy: {len(effectiveness['tier_accuracy'])} tiers analyzed")
    print(f"  Quality achievement: {len(effectiveness['quality_achievement'])} metrics")
    print(f"  Recommendation quality: {effectiveness['recommendation_quality']:.2%}")

    # Generate report
    print("\n✓ Generating routing report:")
    report = analytics.generate_routing_report()
    print(f"  Total decisions: {report['summary']['total_decisions']}")
    print(f"  Success rate: {report['summary']['overall_success_rate']:.2%}")
    print(f"  Avg confidence: {report['summary']['avg_confidence']:.2f}")
    print(f"  Optimization opportunities: {len(report['optimization_opportunities'])}")
    print(f"  Recommendations: {len(report['recommendations'])}")

    if report['recommendations']:
        print("\n  Sample recommendations:")
        for rec in report['recommendations'][:2]:
            print(f"    - {rec}")

    # Export to JSON
    print("\n✓ Testing JSON export:")
    export_path = Path("/tmp/routing_analytics_test.json")
    analytics.export_to_json(str(export_path))
    assert export_path.exists(), "Failed to export JSON"
    print(f"  Exported to {export_path}")

    # Get statistics
    print("\n✓ Testing statistics:")
    stats = analytics.get_statistics()
    print(f"  Current metrics: {stats['current_metrics']['total_decisions']} decisions")
    print(f"  Tier performance: {len(stats['tier_performance'])} tiers tracked")

    print("\n✅ All routing analytics tests passed!")
    return analytics


if __name__ == "__main__":
    test_routing_analytics()