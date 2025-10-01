"""
Intelligent Tier Selection - Task 2 Implementation
Smart tier selection based on complexity, quality requirements, and constraints.
"""

import logging
import time
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import threading

logger = logging.getLogger(__name__)


@dataclass
class TierDecision:
    """Container for tier selection decision."""
    tier: int
    confidence: float
    reasoning: str
    factors: Dict[str, Any]
    estimated_processing_time: float
    estimated_quality: float


class IntelligentTierSelector:
    """
    Intelligently selects processing tier based on multiple factors including
    complexity, quality requirements, time budget, and user preferences.
    """

    def __init__(self,
                 enable_ml_predictor: bool = True,
                 history_size: int = 1000):
        """
        Initialize intelligent tier selector.

        Args:
            enable_ml_predictor: Whether to use ML for tier prediction
            history_size: Size of historical performance data to keep
        """
        self.enable_ml_predictor = enable_ml_predictor
        self.history_size = history_size

        # Decision rules for tier selection
        self.decision_rules = {
            'simple': {'complexity': (0, 0.3), 'tier': 1},
            'moderate': {'complexity': (0.3, 0.7), 'tier': 2},
            'complex': {'complexity': (0.7, 1.0), 'tier': 3}
        }

        # Tier capabilities and characteristics
        self.tier_capabilities = {
            1: {
                'name': 'Fast',
                'max_time': 2.0,
                'avg_time': 1.5,
                'min_quality': 0.7,
                'avg_quality': 0.75,
                'methods': ['statistical', 'formulas'],
                'description': 'Quick processing with basic optimization'
            },
            2: {
                'name': 'Balanced',
                'max_time': 5.0,
                'avg_time': 3.5,
                'min_quality': 0.8,
                'avg_quality': 0.85,
                'methods': ['statistical', 'learned', 'regression'],
                'description': 'Balanced processing with moderate optimization'
            },
            3: {
                'name': 'Quality',
                'max_time': 15.0,
                'avg_time': 8.0,
                'min_quality': 0.9,
                'avg_quality': 0.95,
                'methods': ['all'],
                'description': 'Comprehensive processing with full optimization'
            }
        }

        # Historical performance tracking
        self.performance_history = deque(maxlen=history_size)
        self.tier_statistics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'total_quality': 0,
            'successes': 0,
            'failures': 0
        })

        # ML predictor components
        self.ml_model = None
        self.scaler = None
        self.feature_names = [
            'complexity', 'target_quality', 'time_budget',
            'spatial_complexity', 'color_complexity', 'edge_complexity',
            'gradient_complexity', 'texture_complexity',
            'image_width', 'image_height', 'image_pixels'
        ]

        # Initialize ML predictor if enabled
        if self.enable_ml_predictor:
            self._initialize_ml_predictor()

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"IntelligentTierSelector initialized (ml_enabled={enable_ml_predictor})")

    def select_tier(self,
                   complexity: float,
                   target_quality: float,
                   time_budget: Optional[float] = None,
                   user_preference: Optional[str] = None,
                   detailed_complexity: Optional[Dict[str, float]] = None,
                   image_info: Optional[Dict[str, Any]] = None) -> int:
        """
        Select appropriate processing tier based on multiple factors.

        Args:
            complexity: Overall complexity score (0-1)
            target_quality: Target quality score (0-1)
            time_budget: Optional time budget in seconds
            user_preference: Optional user preference ('fast', 'balanced', 'quality')
            detailed_complexity: Optional detailed complexity scores
            image_info: Optional image information

        Returns:
            Selected tier (1, 2, or 3)
        """
        start_time = time.time()

        # Build decision context
        decision_context = {
            'complexity': complexity,
            'target_quality': target_quality,
            'time_budget': time_budget,
            'user_preference': user_preference,
            'detailed_complexity': detailed_complexity or {},
            'image_info': image_info or {}
        }

        # Try ML prediction first if available
        ml_tier = None
        ml_confidence = 0.0
        if self.enable_ml_predictor and self.ml_model is not None:
            ml_tier, ml_confidence = self._predict_with_ml(decision_context)

        # Apply rule-based decision
        rule_tier = self._apply_decision_rules(
            complexity, target_quality, time_budget, user_preference
        )

        # Combine ML and rule-based decisions
        if ml_tier is not None and ml_confidence > 0.7:
            # High confidence ML prediction
            final_tier = ml_tier
            decision_method = "ml_prediction"
        elif ml_tier is not None and ml_confidence > 0.5:
            # Medium confidence - average with rules
            final_tier = round((ml_tier + rule_tier) / 2)
            decision_method = "ml_rule_hybrid"
        else:
            # Low confidence or no ML - use rules
            final_tier = rule_tier
            decision_method = "rule_based"

        # Ensure tier is valid
        final_tier = max(1, min(3, final_tier))

        # Build reasoning explanation
        reasoning = self._generate_reasoning(
            final_tier, decision_context, decision_method, ml_confidence
        )

        # Log decision
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Tier {final_tier} selected in {processing_time:.2f}ms: {reasoning}")

        return final_tier

    def select_tier_with_details(self,
                                complexity: float,
                                target_quality: float,
                                time_budget: Optional[float] = None,
                                user_preference: Optional[str] = None,
                                detailed_complexity: Optional[Dict[str, float]] = None,
                                image_info: Optional[Dict[str, Any]] = None) -> TierDecision:
        """
        Select tier and return detailed decision information.

        Args:
            Same as select_tier

        Returns:
            TierDecision object with full decision details
        """
        start_time = time.time()

        # Get basic tier selection
        tier = self.select_tier(
            complexity, target_quality, time_budget,
            user_preference, detailed_complexity, image_info
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            tier, complexity, target_quality, time_budget
        )

        # Build detailed factors
        factors = {
            'complexity_score': complexity,
            'target_quality': target_quality,
            'time_budget': time_budget,
            'user_preference': user_preference,
            'complexity_category': self._get_complexity_category(complexity),
            'quality_requirement': self._get_quality_requirement(target_quality),
            'time_constraint': self._get_time_constraint(time_budget)
        }

        if detailed_complexity:
            factors['detailed_complexity'] = detailed_complexity

        # Estimate performance
        estimated_time = self._estimate_processing_time(tier, complexity)
        estimated_quality = self._estimate_quality(tier, complexity)

        # Generate reasoning
        reasoning = self._generate_detailed_reasoning(tier, factors, confidence)

        processing_time = (time.time() - start_time) * 1000

        return TierDecision(
            tier=tier,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors,
            estimated_processing_time=estimated_time,
            estimated_quality=estimated_quality
        )

    def _apply_decision_rules(self,
                             complexity: float,
                             target_quality: float,
                             time_budget: Optional[float],
                             user_preference: Optional[str]) -> int:
        """
        Apply rule-based decision logic.

        Args:
            complexity: Complexity score (0-1)
            target_quality: Target quality (0-1)
            time_budget: Time budget in seconds
            user_preference: User preference

        Returns:
            Selected tier
        """
        # Factor 1: Complexity-based tier
        base_tier = self.get_complexity_tier(complexity)

        # Factor 2: Quality requirement adjustment
        if target_quality > 0.95:
            base_tier = min(base_tier + 1, 3)
        elif target_quality > 0.9:
            base_tier = min(base_tier + 1, 3) if base_tier == 1 else base_tier
        elif target_quality < 0.7:
            base_tier = max(base_tier - 1, 1)

        # Factor 3: Time budget constraint
        if time_budget is not None:
            if time_budget <= 2:
                base_tier = 1  # Must use fast tier
            elif time_budget < 4:
                base_tier = min(base_tier, 2)  # Can't use quality tier
            elif time_budget < 8:
                # Prefer lower tier if close to limit
                if base_tier == 3 and complexity < 0.8:
                    base_tier = 2

        # Factor 4: User preference override
        if user_preference:
            if user_preference == 'fast':
                base_tier = 1
            elif user_preference == 'quality':
                base_tier = 3
            elif user_preference == 'balanced':
                base_tier = 2

        return base_tier

    def get_complexity_tier(self, complexity: float) -> int:
        """
        Get tier based on complexity score.

        Args:
            complexity: Complexity score (0-1)

        Returns:
            Recommended tier
        """
        for category, rule in self.decision_rules.items():
            min_comp, max_comp = rule['complexity']
            if min_comp <= complexity < max_comp:
                return rule['tier']
        return 2  # Default to balanced

    def _initialize_ml_predictor(self):
        """Initialize ML predictor model."""
        try:
            # Try to load existing model
            model_path = Path(__file__).parent / "tier_predictor_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data['model']
                    self.scaler = model_data['scaler']
                logger.info("Loaded existing ML tier predictor model")
            else:
                # Create and train new model with synthetic data
                self._train_ml_predictor()

        except Exception as e:
            logger.warning(f"Failed to initialize ML predictor: {e}")
            self.ml_model = None
            self.scaler = None

    def _train_ml_predictor(self):
        """Train ML predictor with synthetic or historical data."""
        try:
            # Generate synthetic training data
            X_train = []
            y_train = []

            # Generate samples for each tier
            np.random.seed(42)
            for tier in range(1, 4):
                for _ in range(100):
                    # Generate features that correlate with tier
                    if tier == 1:
                        complexity = np.random.uniform(0, 0.4)
                        target_quality = np.random.uniform(0.6, 0.8)
                        time_budget = np.random.uniform(0.5, 3)
                    elif tier == 2:
                        complexity = np.random.uniform(0.3, 0.7)
                        target_quality = np.random.uniform(0.75, 0.9)
                        time_budget = np.random.uniform(2, 6)
                    else:
                        complexity = np.random.uniform(0.6, 1.0)
                        target_quality = np.random.uniform(0.85, 1.0)
                        time_budget = np.random.uniform(5, 20)

                    # Add noise to other features
                    features = [
                        complexity,
                        target_quality,
                        time_budget,
                        complexity * np.random.uniform(0.8, 1.2),  # spatial
                        complexity * np.random.uniform(0.7, 1.3),  # color
                        complexity * np.random.uniform(0.8, 1.2),  # edge
                        complexity * np.random.uniform(0.8, 1.2),  # gradient
                        complexity * np.random.uniform(0.6, 1.4),  # texture
                        np.random.uniform(100, 1000),  # width
                        np.random.uniform(100, 1000),  # height
                        np.random.uniform(10000, 1000000)  # pixels
                    ]

                    X_train.append(features)
                    y_train.append(tier)

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Train scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)

            # Train model
            self.ml_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y_train)

            # Evaluate
            train_score = self.ml_model.score(X_scaled, y_train)
            logger.info(f"Trained ML tier predictor with accuracy: {train_score:.3f}")

        except Exception as e:
            logger.error(f"Failed to train ML predictor: {e}")
            self.ml_model = None
            self.scaler = None

    def _predict_with_ml(self, decision_context: Dict[str, Any]) -> Tuple[Optional[int], float]:
        """
        Predict tier using ML model.

        Args:
            decision_context: Decision context with features

        Returns:
            Tuple of (predicted_tier, confidence)
        """
        if self.ml_model is None or self.scaler is None:
            return None, 0.0

        try:
            # Extract features
            features = []
            features.append(decision_context['complexity'])
            features.append(decision_context['target_quality'])
            features.append(decision_context.get('time_budget', 10.0))

            # Detailed complexity features
            detailed = decision_context.get('detailed_complexity', {})
            features.append(detailed.get('spatial_complexity', decision_context['complexity']))
            features.append(detailed.get('color_complexity', decision_context['complexity']))
            features.append(detailed.get('edge_complexity', decision_context['complexity']))
            features.append(detailed.get('gradient_complexity', decision_context['complexity']))
            features.append(detailed.get('texture_complexity', decision_context['complexity']))

            # Image info features
            info = decision_context.get('image_info', {})
            features.append(info.get('width', 500))
            features.append(info.get('height', 500))
            features.append(info.get('width', 500) * info.get('height', 500))

            # Scale features
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)

            # Predict with probabilities
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            return int(prediction), float(confidence)

        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None, 0.0

    def _calculate_confidence(self,
                             tier: int,
                             complexity: float,
                             target_quality: float,
                             time_budget: Optional[float]) -> float:
        """
        Calculate confidence in tier selection.

        Args:
            tier: Selected tier
            complexity: Complexity score
            target_quality: Target quality
            time_budget: Time budget

        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0

        # Check if tier matches complexity
        expected_tier = self.get_complexity_tier(complexity)
        if tier != expected_tier:
            confidence *= 0.8

        # Check if tier can meet quality requirement
        tier_quality = self.tier_capabilities[tier]['avg_quality']
        if target_quality > tier_quality:
            confidence *= (tier_quality / target_quality)

        # Check if tier can meet time budget
        if time_budget:
            tier_time = self.tier_capabilities[tier]['avg_time']
            if tier_time > time_budget:
                confidence *= (time_budget / tier_time)

        # Consider historical performance
        with self._lock:
            stats = self.tier_statistics[tier]
            if stats['count'] > 0:
                success_rate = stats['successes'] / stats['count']
                confidence *= (0.5 + 0.5 * success_rate)

        return max(0.1, min(1.0, confidence))

    def _estimate_processing_time(self, tier: int, complexity: float) -> float:
        """
        Estimate processing time for given tier and complexity.

        Args:
            tier: Selected tier
            complexity: Complexity score

        Returns:
            Estimated processing time in seconds
        """
        base_time = self.tier_capabilities[tier]['avg_time']

        # Adjust based on complexity
        complexity_factor = 0.5 + complexity  # 0.5x to 1.5x
        estimated_time = base_time * complexity_factor

        # Consider historical performance
        with self._lock:
            stats = self.tier_statistics[tier]
            if stats['count'] > 10:
                avg_historical = stats['total_time'] / stats['count']
                # Blend with historical average
                estimated_time = 0.7 * estimated_time + 0.3 * avg_historical

        return estimated_time

    def _estimate_quality(self, tier: int, complexity: float) -> float:
        """
        Estimate achievable quality for given tier and complexity.

        Args:
            tier: Selected tier
            complexity: Complexity score

        Returns:
            Estimated quality score (0-1)
        """
        base_quality = self.tier_capabilities[tier]['avg_quality']

        # Adjust based on complexity (higher complexity = slightly lower quality)
        complexity_penalty = complexity * 0.1  # Up to 10% penalty
        estimated_quality = base_quality - complexity_penalty

        # Consider historical performance
        with self._lock:
            stats = self.tier_statistics[tier]
            if stats['count'] > 10:
                avg_historical = stats['total_quality'] / stats['count']
                # Blend with historical average
                estimated_quality = 0.7 * estimated_quality + 0.3 * avg_historical

        return max(0.5, min(1.0, estimated_quality))

    def _get_complexity_category(self, complexity: float) -> str:
        """Get complexity category name."""
        if complexity < 0.3:
            return "simple"
        elif complexity < 0.7:
            return "moderate"
        else:
            return "complex"

    def _get_quality_requirement(self, target_quality: float) -> str:
        """Get quality requirement category."""
        if target_quality >= 0.95:
            return "very_high"
        elif target_quality >= 0.85:
            return "high"
        elif target_quality >= 0.75:
            return "moderate"
        else:
            return "basic"

    def _get_time_constraint(self, time_budget: Optional[float]) -> str:
        """Get time constraint category."""
        if time_budget is None:
            return "none"
        elif time_budget < 2:
            return "strict"
        elif time_budget < 5:
            return "moderate"
        else:
            return "relaxed"

    def _generate_reasoning(self,
                           tier: int,
                           context: Dict[str, Any],
                           method: str,
                           ml_confidence: float) -> str:
        """Generate brief reasoning for tier selection."""
        tier_name = self.tier_capabilities[tier]['name']
        complexity_cat = self._get_complexity_category(context['complexity'])
        quality_req = self._get_quality_requirement(context['target_quality'])

        reasoning = f"{tier_name} tier selected for {complexity_cat} complexity, {quality_req} quality requirement"

        if context['time_budget']:
            time_constraint = self._get_time_constraint(context['time_budget'])
            reasoning += f", {time_constraint} time constraint"

        if context['user_preference']:
            reasoning += f", user prefers {context['user_preference']}"

        if method == "ml_prediction":
            reasoning += f" (ML confidence: {ml_confidence:.2f})"

        return reasoning

    def _generate_detailed_reasoning(self,
                                    tier: int,
                                    factors: Dict[str, Any],
                                    confidence: float) -> str:
        """Generate detailed reasoning explanation."""
        tier_info = self.tier_capabilities[tier]
        reasons = []

        # Complexity-based reasoning
        complexity_cat = factors['complexity_category']
        reasons.append(f"Image has {complexity_cat} complexity ({factors['complexity_score']:.2f})")

        # Quality-based reasoning
        quality_req = factors['quality_requirement']
        reasons.append(f"Quality requirement is {quality_req} ({factors['target_quality']:.2f})")

        # Time-based reasoning
        if factors['time_budget']:
            time_constraint = factors['time_constraint']
            reasons.append(f"Time constraint is {time_constraint} ({factors['time_budget']:.1f}s)")

        # User preference reasoning
        if factors['user_preference']:
            reasons.append(f"User preference is '{factors['user_preference']}'")

        # Tier capabilities
        reasons.append(
            f"{tier_info['name']} tier provides {tier_info['description']} "
            f"with {tier_info['avg_quality']:.0%} quality in ~{tier_info['avg_time']:.1f}s"
        )

        # Confidence statement
        confidence_level = "high" if confidence > 0.8 else ("moderate" if confidence > 0.6 else "low")
        reasons.append(f"Decision confidence is {confidence_level} ({confidence:.2f})")

        return " | ".join(reasons)

    def record_performance(self,
                          tier: int,
                          actual_time: float,
                          actual_quality: float,
                          success: bool):
        """
        Record actual performance for a tier selection.

        Args:
            tier: Tier that was used
            actual_time: Actual processing time
            actual_quality: Actual quality achieved
            success: Whether processing was successful
        """
        with self._lock:
            # Update tier statistics
            stats = self.tier_statistics[tier]
            stats['count'] += 1
            stats['total_time'] += actual_time
            stats['total_quality'] += actual_quality
            if success:
                stats['successes'] += 1
            else:
                stats['failures'] += 1

            # Add to history
            self.performance_history.append({
                'tier': tier,
                'time': actual_time,
                'quality': actual_quality,
                'success': success,
                'timestamp': time.time()
            })

        logger.debug(f"Recorded performance: tier={tier}, time={actual_time:.2f}s, "
                    f"quality={actual_quality:.3f}, success={success}")

    def get_tier_recommendation(self,
                               complexity: float,
                               target_quality: float,
                               time_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Get tier recommendation with explanation.

        Args:
            complexity: Complexity score
            target_quality: Target quality
            time_budget: Time budget

        Returns:
            Dictionary with recommendation and explanation
        """
        decision = self.select_tier_with_details(
            complexity, target_quality, time_budget
        )

        return {
            'recommended_tier': decision.tier,
            'tier_name': self.tier_capabilities[decision.tier]['name'],
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'estimated_time': decision.estimated_processing_time,
            'estimated_quality': decision.estimated_quality,
            'alternatives': self._get_alternative_tiers(decision)
        }

    def _get_alternative_tiers(self, decision: TierDecision) -> List[Dict[str, Any]]:
        """Get alternative tier options."""
        alternatives = []
        for tier in [1, 2, 3]:
            if tier != decision.tier:
                alternatives.append({
                    'tier': tier,
                    'name': self.tier_capabilities[tier]['name'],
                    'estimated_time': self._estimate_processing_time(tier, decision.factors['complexity_score']),
                    'estimated_quality': self._estimate_quality(tier, decision.factors['complexity_score'])
                })
        return alternatives

    def save_ml_model(self, path: Optional[str] = None):
        """Save ML model to disk."""
        if self.ml_model is None:
            logger.warning("No ML model to save")
            return

        save_path = path or (Path(__file__).parent / "tier_predictor_model.pkl")

        try:
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': self.ml_model,
                    'scaler': self.scaler
                }, f)
            logger.info(f"Saved ML model to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get tier selection statistics."""
        with self._lock:
            stats = {}
            for tier, tier_stats in self.tier_statistics.items():
                if tier_stats['count'] > 0:
                    stats[f"tier_{tier}"] = {
                        'count': tier_stats['count'],
                        'avg_time': tier_stats['total_time'] / tier_stats['count'],
                        'avg_quality': tier_stats['total_quality'] / tier_stats['count'],
                        'success_rate': tier_stats['successes'] / tier_stats['count']
                    }
            return stats


def test_intelligent_tier_selector():
    """Test the intelligent tier selector."""
    print("Testing Intelligent Tier Selector...")

    # Initialize selector
    selector = IntelligentTierSelector(enable_ml_predictor=True)

    # Test cases
    test_cases = [
        # Simple image, basic quality, no time constraint
        {
            'complexity': 0.2,
            'target_quality': 0.75,
            'time_budget': None,
            'expected_tier': 1,
            'description': "Simple image with basic quality"
        },
        # Complex image, high quality requirement
        {
            'complexity': 0.8,
            'target_quality': 0.95,
            'time_budget': None,
            'expected_tier': 3,
            'description': "Complex image with high quality"
        },
        # Simple complexity, strict time constraint
        {
            'complexity': 0.2,
            'target_quality': 0.75,
            'time_budget': 1.5,
            'expected_tier': 1,
            'description': "Simple with very strict time"
        },
        # User preference override
        {
            'complexity': 0.4,
            'target_quality': 0.8,
            'time_budget': None,
            'user_preference': 'quality',
            'expected_tier': 3,
            'description': "User prefers quality"
        }
    ]

    print("\n✓ Testing tier selection:")
    for i, test in enumerate(test_cases, 1):
        tier = selector.select_tier(
            test['complexity'],
            test['target_quality'],
            test.get('time_budget'),
            test.get('user_preference')
        )

        print(f"  Test {i}: {test['description']}")
        print(f"    Selected tier: {tier} (expected: {test['expected_tier']})")
        assert tier == test['expected_tier'], f"Unexpected tier for test {i}"

    # Test detailed decision
    print("\n✓ Testing detailed decision:")
    decision = selector.select_tier_with_details(
        complexity=0.5,
        target_quality=0.9,
        time_budget=5.0,
        detailed_complexity={
            'spatial_complexity': 0.6,
            'color_complexity': 0.4,
            'edge_complexity': 0.5,
            'gradient_complexity': 0.5,
            'texture_complexity': 0.3
        }
    )

    print(f"  Tier: {decision.tier}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Estimated time: {decision.estimated_processing_time:.1f}s")
    print(f"  Estimated quality: {decision.estimated_quality:.2%}")
    print(f"  Reasoning: {decision.reasoning[:100]}...")

    # Test performance recording
    print("\n✓ Testing performance recording:")
    for tier in [1, 2, 3]:
        for _ in range(5):
            selector.record_performance(
                tier=tier,
                actual_time=np.random.uniform(1, 10),
                actual_quality=np.random.uniform(0.7, 0.95),
                success=np.random.choice([True, False], p=[0.9, 0.1])
            )

    stats = selector.get_statistics()
    for tier_key, tier_stats in stats.items():
        print(f"  {tier_key}: {tier_stats['count']} selections, "
              f"{tier_stats['success_rate']:.0%} success")

    # Test recommendation
    print("\n✓ Testing recommendation:")
    recommendation = selector.get_tier_recommendation(
        complexity=0.6,
        target_quality=0.88,
        time_budget=4.0
    )

    print(f"  Recommended: Tier {recommendation['recommended_tier']} ({recommendation['tier_name']})")
    print(f"  Confidence: {recommendation['confidence']:.2f}")
    print(f"  Alternatives: {len(recommendation['alternatives'])} options")

    print("\n✅ All tier selector tests passed!")
    return selector


if __name__ == "__main__":
    test_intelligent_tier_selector()