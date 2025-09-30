# DAY 1: Production Model Integration - Monday

**Date**: Week 5, Day 1
**Duration**: 8 hours
**Focus**: Optimize exported model loading and management for production deployment
**Lead Developer**: Backend Engineer (Primary)
**Support**: DevOps Engineer (Infrastructure)

---

## 🎯 **Daily Objectives**

**Primary Goal**: Implement production-optimized model loading system with <3 second startup and <500MB memory usage

**Key Deliverables**:
1. ProductionModelManager with optimized loading
2. OptimizedQualityPredictor with batched inference
3. Model warmup and caching system
4. Memory usage monitoring and optimization

---

## ⏰ **Hour-by-Hour Schedule**

### **Hour 1-2 (9:00-11:00): ProductionModelManager Implementation**

#### **Task 1.1: Core Model Manager Structure** (90 minutes)
```python
# backend/ai_modules/management/production_model_manager.py
class ProductionModelManager:
    def __init__(self, model_dir: str = "backend/ai_modules/models/exported"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_metadata = {}
        self.loading_lock = threading.Lock()

    def _load_all_exported_models(self) -> Dict[str, Any]:
        """Load all exported models with error handling"""
        models = {}

        # Quality Predictor (TorchScript)
        try:
            models['quality_predictor'] = torch.jit.load(
                str(self.model_dir / 'quality_predictor.torchscript')
            )
            models['quality_predictor'].eval()
            logging.info("✅ Quality predictor loaded")
        except Exception as e:
            logging.warning(f"⚠️ Quality predictor unavailable: {e}")
            models['quality_predictor'] = None

        # Logo Classifier (ONNX)
        try:
            import onnxruntime as ort
            models['logo_classifier'] = ort.InferenceSession(
                str(self.model_dir / 'logo_classifier.onnx')
            )
            logging.info("✅ Logo classifier loaded")
        except Exception as e:
            logging.warning(f"⚠️ Logo classifier unavailable: {e}")
            models['logo_classifier'] = None

        # Correlation Models (Pickle)
        try:
            models['correlation_models'] = joblib.load(
                str(self.model_dir / 'correlation_models.pkl')
            )
            logging.info("✅ Correlation models loaded")
        except Exception as e:
            logging.warning(f"⚠️ Correlation models unavailable: {e}")
            models['correlation_models'] = None

        return models
```

**Checklist**:
- [ ] Create `backend/ai_modules/management/` directory
- [ ] Implement `ProductionModelManager` class structure
- [ ] Add model loading with error handling
- [ ] Implement logging for model loading status
- [ ] Test with mock model files

**Dependencies**: None
**Estimated Time**: 1.5 hours
**Success Criteria**: Model manager loads available models without errors

---

#### **Task 1.2: Model Loading Optimization** (30 minutes)
```python
def _optimize_for_production(self):
    """Optimize models for production inference"""
    for model_name, model in self.models.items():
        if model is None:
            continue

        if hasattr(model, 'eval'):
            model.eval()  # Set to evaluation mode

        # Warmup models with dummy input
        self._warmup_model(model_name, model)

        # Track memory usage
        self._track_model_memory(model_name)

def _warmup_model(self, model_name: str, model):
    """Warmup model with dummy inference"""
    try:
        if model_name == 'quality_predictor':
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_params = torch.randn(1, 8)
            with torch.no_grad():
                _ = model(dummy_input, dummy_params)
        elif model_name == 'logo_classifier':
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            _ = model.run(None, {model.get_inputs()[0].name: dummy_input})

        logging.info(f"✅ {model_name} warmed up")
    except Exception as e:
        logging.warning(f"⚠️ {model_name} warmup failed: {e}")
```

**Checklist**:
- [ ] Implement model optimization methods
- [ ] Add model warmup functionality
- [ ] Create memory tracking utilities
- [ ] Test warmup with actual model files
- [ ] Verify performance improvement

**Dependencies**: Task 1.1 completion
**Estimated Time**: 30 minutes
**Success Criteria**: Models warm up successfully with dummy inputs

---

### **Hour 3-4 (11:00-13:00): OptimizedQualityPredictor Implementation**

#### **Task 2.1: Quality Predictor Optimization** (90 minutes)
```python
# backend/ai_modules/inference/optimized_quality_predictor.py
class OptimizedQualityPredictor:
    def __init__(self, model_manager: ProductionModelManager):
        self.model_manager = model_manager
        self.model = None
        self.preprocessor = None
        self._initialize()

    def _initialize(self):
        """Initialize predictor with error handling"""
        try:
            self.model = self.model_manager.models.get('quality_predictor')
            self.preprocessor = self.model_manager.models.get('feature_preprocessor')

            if self.model is None:
                logging.warning("Quality predictor model not available")
                return False

            # Test inference capability
            self._test_inference()
            return True

        except Exception as e:
            logging.error(f"Quality predictor initialization failed: {e}")
            return False

    def predict_quality(self, image_path: str, params: Dict[str, float]) -> float:
        """Predict SSIM quality for given image and parameters"""
        if self.model is None:
            # Fallback to simple heuristic
            return self._heuristic_quality_estimate(params)

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            param_tensor = self._encode_parameters(params)

            # Run inference
            with torch.no_grad():
                predicted_ssim = self.model(image_tensor, param_tensor).item()

            # Ensure valid range [0, 1]
            return max(0.0, min(1.0, predicted_ssim))

        except Exception as e:
            logging.warning(f"Quality prediction failed: {e}")
            return self._heuristic_quality_estimate(params)
```

**Checklist**:
- [ ] Create OptimizedQualityPredictor class
- [ ] Implement model initialization with fallbacks
- [ ] Add image preprocessing methods
- [ ] Create parameter encoding system
- [ ] Implement heuristic fallback for model failures
- [ ] Test with real image files

**Dependencies**: Task 1.1 completion
**Estimated Time**: 1.5 hours
**Success Criteria**: Quality predictor works with exported models and fallbacks gracefully

---

#### **Task 2.2: Batch Inference Capabilities** (30 minutes)
```python
def predict_batch(self, image_paths: List[str], params_list: List[Dict]) -> List[float]:
    """Batched inference for efficiency"""
    if len(image_paths) != len(params_list):
        raise ValueError("Image paths and params lists must have same length")

    if self.model is None:
        return [self._heuristic_quality_estimate(p) for p in params_list]

    try:
        # Batch preprocessing
        image_batch = torch.stack([
            self._preprocess_image(path) for path in image_paths
        ])
        param_batch = torch.stack([
            self._encode_parameters(params) for params in params_list
        ])

        # Batch inference
        with torch.no_grad():
            predictions = self.model(image_batch, param_batch)

        # Convert to list and ensure valid range
        return [max(0.0, min(1.0, pred.item())) for pred in predictions]

    except Exception as e:
        logging.warning(f"Batch prediction failed: {e}")
        return [self._heuristic_quality_estimate(p) for p in params_list]

def _estimate_processing_time(self, num_images: int) -> float:
    """Estimate processing time for batch"""
    if self.model is None:
        return num_images * 0.01  # Heuristic is very fast
    return num_images * 0.05  # Model inference per image
```

**Checklist**:
- [ ] Implement batch processing methods
- [ ] Add processing time estimation
- [ ] Create batch preprocessing utilities
- [ ] Test with multiple images
- [ ] Verify performance improvement over individual calls

**Dependencies**: Task 2.1 completion
**Estimated Time**: 30 minutes
**Success Criteria**: Batch inference processes multiple images efficiently

---

### **Hour 5-6 (14:00-16:00): Model Caching & Memory Optimization**

#### **Task 3.1: Memory Usage Monitoring** (60 minutes)
```python
# backend/ai_modules/management/memory_monitor.py
class ModelMemoryMonitor:
    def __init__(self):
        self.memory_stats = {}
        self.peak_usage = 0

    def track_model_memory(self, model_name: str, model) -> Dict[str, float]:
        """Track memory usage for a specific model"""
        import psutil
        import sys

        # Memory before model
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Estimate model size
        model_size = 0
        if hasattr(model, 'parameters'):
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        elif hasattr(model, 'get_session_config'):
            # ONNX model - estimate from file size
            model_size = 50  # Approximate for quality predictor

        # Memory after model loading
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        self.memory_stats[model_name] = {
            'estimated_size_mb': model_size,
            'actual_memory_delta_mb': memory_delta,
            'total_memory_mb': memory_after
        }

        self.peak_usage = max(self.peak_usage, memory_after)

        logging.info(f"📊 {model_name}: {model_size:.1f}MB estimated, {memory_delta:.1f}MB actual")
        return self.memory_stats[model_name]

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        import psutil

        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_usage,
            'model_breakdown': self.memory_stats,
            'memory_limit_mb': 500,  # Target limit
            'within_limits': current_memory < 500
        }
```

**Checklist**:
- [ ] Create ModelMemoryMonitor class
- [ ] Implement memory tracking per model
- [ ] Add peak usage monitoring
- [ ] Create memory reporting system
- [ ] Test with actual models
- [ ] Verify memory stays within 500MB limit

**Dependencies**: Tasks 1.1, 2.1 completion
**Estimated Time**: 1 hour
**Success Criteria**: Memory monitoring tracks usage and stays within limits

---

#### **Task 3.2: Model Caching System** (60 minutes)
```python
# backend/ai_modules/management/model_cache.py
class ModelCache:
    def __init__(self, max_memory_mb: int = 400):
        self.cache = {}
        self.access_times = {}
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0

    def get_model(self, model_name: str, loader_func=None):
        """Get model from cache or load if needed"""
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            logging.debug(f"📋 Cache hit: {model_name}")
            return self.cache[model_name]

        if loader_func is None:
            return None

        # Check memory before loading
        if self._estimate_memory_after_load() > self.max_memory_mb:
            self._evict_least_used()

        # Load model
        model = loader_func()
        if model is not None:
            self.cache[model_name] = model
            self.access_times[model_name] = time.time()
            self._update_memory_usage()
            logging.info(f"📥 Cached: {model_name}")

        return model

    def _evict_least_used(self):
        """Remove least recently used model"""
        if not self.cache:
            return

        lru_model = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.remove_model(lru_model)

    def remove_model(self, model_name: str):
        """Remove model from cache"""
        if model_name in self.cache:
            del self.cache[model_name]
            del self.access_times[model_name]
            self._update_memory_usage()
            logging.info(f"🗑️ Evicted: {model_name}")

    def clear_cache(self):
        """Clear all cached models"""
        self.cache.clear()
        self.access_times.clear()
        self.current_memory_mb = 0
        logging.info("🧹 Cache cleared")
```

**Checklist**:
- [ ] Create ModelCache class with LRU eviction
- [ ] Implement memory-aware caching
- [ ] Add cache hit/miss tracking
- [ ] Create cache statistics methods
- [ ] Test cache behavior under memory pressure
- [ ] Verify models are properly evicted when needed

**Dependencies**: Task 3.1 completion
**Estimated Time**: 1 hour
**Success Criteria**: Cache manages memory efficiently and improves model access speed

---

### **Hour 7-8 (16:00-18:00): Integration & Testing**

#### **Task 4.1: Integration Testing** (60 minutes)
```python
# tests/test_production_model_integration.py
class TestProductionModelIntegration:
    def setup_method(self):
        """Setup test environment"""
        self.model_manager = ProductionModelManager()
        self.quality_predictor = OptimizedQualityPredictor(self.model_manager)
        self.memory_monitor = ModelMemoryMonitor()

    def test_model_loading_performance(self):
        """Test model loading time meets requirements"""
        start_time = time.time()

        # Load all models
        models = self.model_manager._load_all_exported_models()

        loading_time = time.time() - start_time

        # Requirement: <3 seconds loading time
        assert loading_time < 3.0, f"Model loading took {loading_time:.2f}s, exceeds 3s limit"

        # Verify at least some models loaded
        available_models = [name for name, model in models.items() if model is not None]
        assert len(available_models) > 0, "No models loaded successfully"

    def test_quality_prediction_performance(self):
        """Test quality prediction speed"""
        test_image = "data/test/simple_logo.png"
        test_params = {"color_precision": 4, "corner_threshold": 30}

        # Warm up
        self.quality_predictor.predict_quality(test_image, test_params)

        # Time multiple predictions
        start_time = time.time()
        for _ in range(10):
            quality = self.quality_predictor.predict_quality(test_image, test_params)
            assert 0.0 <= quality <= 1.0, f"Invalid quality value: {quality}"

        avg_time = (time.time() - start_time) / 10

        # Requirement: <100ms per prediction
        assert avg_time < 0.1, f"Quality prediction took {avg_time:.3f}s, exceeds 0.1s limit"

    def test_memory_usage_limits(self):
        """Test memory usage stays within limits"""
        # Load all models
        self.model_manager._load_all_exported_models()

        # Generate memory report
        memory_report = self.memory_monitor.get_memory_report()

        # Requirement: <500MB total memory
        assert memory_report['current_memory_mb'] < 500, \
            f"Memory usage {memory_report['current_memory_mb']:.1f}MB exceeds 500MB limit"

        assert memory_report['within_limits'], "Memory usage exceeds configured limits"
```

**Checklist**:
- [ ] Create integration test suite
- [ ] Test model loading performance (<3 seconds)
- [ ] Test quality prediction speed (<100ms)
- [ ] Test memory usage limits (<500MB)
- [ ] Test graceful fallbacks when models unavailable
- [ ] Verify all components work together

**Dependencies**: All previous tasks completion
**Estimated Time**: 1 hour
**Success Criteria**: All tests pass and performance requirements met

---

#### **Task 4.2: Performance Benchmarking & Validation** (60 minutes)
```python
# scripts/benchmark_day1_implementation.py
class Day1PerformanceBenchmark:
    def __init__(self):
        self.results = {}

    def run_all_benchmarks(self):
        """Run comprehensive performance benchmarks"""
        self.results['model_loading'] = self.benchmark_model_loading()
        self.results['quality_prediction'] = self.benchmark_quality_prediction()
        self.results['memory_efficiency'] = self.benchmark_memory_usage()
        self.results['concurrent_performance'] = self.benchmark_concurrent_usage()

        return self.results

    def benchmark_model_loading(self):
        """Benchmark model loading performance"""
        times = []

        for i in range(5):
            # Clear any cached models
            model_manager = ProductionModelManager()

            start_time = time.time()
            models = model_manager._load_all_exported_models()
            model_manager._optimize_for_production()
            loading_time = time.time() - start_time

            times.append(loading_time)

        avg_time = sum(times) / len(times)

        return {
            'average_loading_time': avg_time,
            'max_loading_time': max(times),
            'meets_requirement': avg_time < 3.0,
            'requirement': '<3 seconds'
        }

    def benchmark_concurrent_usage(self):
        """Test concurrent model usage"""
        import threading
        import concurrent.futures

        model_manager = ProductionModelManager()
        quality_predictor = OptimizedQualityPredictor(model_manager)

        def worker_task():
            test_image = "data/test/simple_logo.png"
            test_params = {"color_precision": 4, "corner_threshold": 30}

            start_time = time.time()
            quality = quality_predictor.predict_quality(test_image, test_params)
            processing_time = time.time() - start_time

            return {
                'quality': quality,
                'processing_time': processing_time,
                'success': True
            }

        # Test with 10 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task) for _ in range(10)]
            results = [future.result() for future in futures]

        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        success_rate = sum(r['success'] for r in results) / len(results)

        return {
            'concurrent_workers': 10,
            'average_processing_time': avg_processing_time,
            'success_rate': success_rate,
            'meets_requirement': avg_processing_time < 0.2 and success_rate > 0.95
        }
```

**Checklist**:
- [ ] Create comprehensive benchmark suite
- [ ] Test model loading performance across multiple runs
- [ ] Benchmark quality prediction speed
- [ ] Test memory efficiency under load
- [ ] Test concurrent usage (10+ workers)
- [ ] Generate performance report
- [ ] Validate all requirements met

**Dependencies**: Task 4.1 completion
**Estimated Time**: 1 hour
**Success Criteria**: All benchmarks pass and requirements validated

---

## 📊 **Day 1 Success Criteria**

### **Performance Requirements**
- [ ] **Model Loading**: <3 seconds (cold start)
- [ ] **Quality Prediction**: <100ms per inference
- [ ] **Memory Usage**: <500MB total (all models loaded)
- [ ] **Concurrent Support**: 10+ requests without degradation

### **Functionality Requirements**
- [ ] **ProductionModelManager**: Loads all available models
- [ ] **OptimizedQualityPredictor**: Works with exported models
- [ ] **Memory Monitoring**: Tracks usage within limits
- [ ] **Graceful Fallbacks**: System works when models unavailable

### **Quality Requirements**
- [ ] **Error Handling**: Comprehensive error recovery
- [ ] **Logging**: Detailed logging for debugging
- [ ] **Testing**: Integration tests pass
- [ ] **Documentation**: Code properly documented

---

## 🔄 **Handoff to Day 2**

### **Completed Deliverables**
- ProductionModelManager with optimized loading
- OptimizedQualityPredictor with batch capabilities
- Memory monitoring and caching system
- Performance benchmarks and validation

### **Available for Day 2**
- Model loading system ready for API integration
- Quality prediction service operational
- Memory usage optimized and monitored
- Performance baselines established

### **Integration Points**
- `ProductionModelManager` → Use in Flask app initialization
- `OptimizedQualityPredictor` → Integrate in AI endpoints
- Memory monitoring → Add to health check endpoints
- Performance metrics → Include in API responses

**Status**: ✅ Day 1 foundation ready for API enhancement