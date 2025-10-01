# Day 8: Intelligent Routing Enhancement

## Objective
Improve the tier selection logic to make smarter decisions based on image complexity, quality requirements, and time constraints.

## Prerequisites
- [x] Unified pipeline from Day 7
- [x] Working classifiers from Day 2
- [x] Quality prediction from Day 3
- [x] Feature extraction working

## Tasks

### Task 1: Complexity Analysis System (2 hours)
**File**: `backend/ai_modules/routing/complexity_analyzer.py`

- [x] Implement multi-dimensional complexity scoring:
  ```python
  class ComplexityAnalyzer:
      def analyze(self, image_path: str) -> Dict:
          return {
              'spatial_complexity': self.calculate_spatial_complexity(),
              'color_complexity': self.calculate_color_complexity(),
              'edge_complexity': self.calculate_edge_complexity(),
              'gradient_complexity': self.calculate_gradient_complexity(),
              'texture_complexity': self.calculate_texture_complexity(),
              'overall_score': self.calculate_overall_score()
          }

      def calculate_spatial_complexity(self, image):
          # Measure detail distribution
          # Check for fine details
          # Analyze region variations
          pass

      def calculate_color_complexity(self, image):
          # Count unique colors
          # Analyze color gradients
          # Check transparency
          pass
  ```
- [x] Add region-based analysis
- [x] Implement frequency domain analysis (FFT)
- [x] Create complexity visualization
- [x] Cache complexity scores

**Acceptance Criteria**:
- Analyzes image in <500ms
- Scores correlate with conversion difficulty
- Handles all image types
- Produces normalized scores (0-1)

### Task 2: Intelligent Tier Selection (2.5 hours)
**File**: `backend/ai_modules/routing/intelligent_tier_selector.py`

- [x] Build decision matrix:
  ```python
  class IntelligentTierSelector:
      def __init__(self):
          self.decision_rules = {
              'simple': {'complexity': (0, 0.3), 'tier': 1},
              'moderate': {'complexity': (0.3, 0.7), 'tier': 2},
              'complex': {'complexity': (0.7, 1.0), 'tier': 3}
          }

      def select_tier(self,
                     complexity: float,
                     target_quality: float,
                     time_budget: Optional[float],
                     user_preference: Optional[str]) -> int:

          # Factor 1: Complexity
          base_tier = self.get_complexity_tier(complexity)

          # Factor 2: Quality requirement
          if target_quality > 0.95:
              base_tier = min(base_tier + 1, 3)

          # Factor 3: Time budget
          if time_budget and time_budget < 3:
              base_tier = 1
          elif time_budget and time_budget < 8:
              base_tier = min(base_tier, 2)

          # Factor 4: User preference
          if user_preference == 'fast':
              base_tier = 1
          elif user_preference == 'quality':
              base_tier = 3

          return base_tier
  ```
- [x] Add machine learning tier predictor
- [x] Implement historical performance consideration
- [x] Create tier recommendation explanation
- [x] Add confidence scoring

**Acceptance Criteria**:
- Selects appropriate tier 90% of time
- Respects time budgets
- Provides clear reasoning
- Handles edge cases

### Task 3: Adaptive Routing Based on Load (1.5 hours)
**File**: `backend/ai_modules/routing/adaptive_router.py`

- [x] Implement load-aware routing:
  ```python
  class AdaptiveRouter:
      def __init__(self):
          self.current_load = 0
          self.processing_times = deque(maxlen=100)
          self.tier_capabilities = {
              1: {'capacity': 10, 'avg_time': 1.5},
              2: {'capacity': 5, 'avg_time': 4.0},
              3: {'capacity': 2, 'avg_time': 12.0}
          }

      def route_with_load_balancing(self, request):
          # Check current system load
          if self.current_load > 0.8:
              # Downgrade tier for new requests
              return self.downgrade_tier(request.tier)

          # Predict processing time
          estimated_time = self.estimate_processing_time(request)

          # Route to tier with capacity
          return self.find_available_tier(request.tier, estimated_time)
  ```
- [x] Track processing queue
- [x] Implement tier downgrade logic
- [x] Add priority queue support
- [x] Monitor resource usage

**Acceptance Criteria**:
- Maintains <5s average response time
- Prevents system overload
- Fair request distribution
- Graceful degradation under load

### Task 4: Routing Performance Analytics (1.5 hours)
**File**: `backend/ai_modules/routing/routing_analytics.py`

- [x] Build analytics system:
  ```python
  class RoutingAnalytics:
      def __init__(self):
          self.tier_performance = defaultdict(list)
          self.routing_decisions = []

      def analyze_routing_effectiveness(self):
          return {
              'tier_accuracy': self.calculate_tier_accuracy(),
              'quality_achievement': self.calculate_quality_achievement(),
              'time_compliance': self.calculate_time_compliance(),
              'recommendation_quality': self.calculate_recommendation_quality()
          }

      def generate_routing_report(self):
          # Tier usage distribution
          # Success rate by tier
          # Quality vs time tradeoffs
          # Optimization opportunities
          pass
  ```
- [x] Track routing decisions and outcomes
- [x] Calculate success metrics
- [x] Identify routing patterns
- [x] Generate improvement recommendations

**Acceptance Criteria**:
- Tracks all routing decisions
- Identifies suboptimal routing
- Generates actionable insights
- Exports analytics data

### Task 5: Routing Configuration & Tuning (30 minutes)
**File**: `backend/ai_modules/routing/routing_config.py`

- [x] Create tunable routing configuration:
  ```yaml
  routing:
    complexity_weights:
      spatial: 0.3
      color: 0.2
      edge: 0.3
      gradient: 0.15
      texture: 0.05

    tier_thresholds:
      tier1_max: 0.3
      tier2_max: 0.7

    quality_boost:
      high_quality: 1  # Upgrade tier for >0.95 target
      medium_quality: 0

    time_constraints:
      strict: true
      tier1_max_time: 2.0
      tier2_max_time: 5.0
      tier3_max_time: 15.0
  ```
- [x] Support hot configuration reload
- [x] Add A/B testing support
- [x] Implement configuration validation
- [x] Create tuning interface

**Acceptance Criteria**:
- Configuration loadable from file
- Changes apply without restart
- Validates constraints
- Supports experimentation

## Deliverables
1. **Complexity Analyzer**: Multi-dimensional complexity scoring
2. **Tier Selector**: Intelligent tier selection logic
3. **Adaptive Router**: Load-aware routing system
4. **Analytics**: Routing performance tracking
5. **Configuration**: Tunable routing parameters

## Testing Commands
```bash
# Test complexity analysis
python -c "from backend.ai_modules.routing.complexity_analyzer import ComplexityAnalyzer; ca = ComplexityAnalyzer(); print(ca.analyze('test.png'))"

# Test tier selection
python -c "from backend.ai_modules.routing.intelligent_tier_selector import IntelligentTierSelector; ts = IntelligentTierSelector(); print(ts.select_tier(0.5, 0.9, 5.0))"

# Test adaptive routing
python -m backend.ai_modules.routing.adaptive_router --simulate-load

# Generate routing analytics
python -c "from backend.ai_modules.routing.routing_analytics import RoutingAnalytics; ra = RoutingAnalytics(); ra.generate_routing_report()"

# Test with different configurations
python scripts/test_routing_configs.py --config routing_aggressive.yaml
```

## Routing Decision Flow
```
Image Input
    ↓
Complexity Analysis
    ↓
┌─────────────────────────────┐
│  Tier Selection Factors:     │
│  • Complexity Score          │
│  • Target Quality            │
│  • Time Budget               │
│  • System Load               │
│  • User Preference           │
└─────────────┬───────────────┘
              ↓
        Tier Decision
              ↓
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Tier 1   Tier 2   Tier 3
  (Fast)  (Balanced) (Quality)
```

## Success Metrics
- [x] Correct tier selection >90% of time
- [x] Time budget compliance >95%
- [x] Quality target achievement >85%
- [x] Load balancing effective (no timeouts)

## Common Issues & Solutions

### Issue: Complexity score doesn't correlate with difficulty
**Solution**:
- Add more complexity dimensions
- Weight factors based on historical data
- Use ML model trained on outcomes

### Issue: Always selecting highest tier
**Solution**:
- Adjust tier thresholds
- Add stronger time constraints
- Consider system load

### Issue: Poor performance under load
**Solution**:
- Implement request queuing
- Add tier downgrade logic
- Cache complexity scores

## Notes
- Routing is critical for user experience
- Balance quality vs speed carefully
- Monitor and adjust continuously
- Consider user preferences strongly

## Next Day Preview
Day 9 will implement a comprehensive A/B testing framework to validate that our AI improvements actually deliver better results than the baseline system.