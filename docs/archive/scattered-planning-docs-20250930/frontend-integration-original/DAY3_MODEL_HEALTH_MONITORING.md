# Day 3: Model Health Monitoring & Status Dashboard

## Overview
Implement comprehensive model health monitoring with real-time status indicators, performance metrics, and intelligent fallback mechanisms for graceful degradation.

## Daily Objectives
- ✅ Build real-time model health dashboard
- ✅ Implement intelligent fallback UI mechanisms
- ✅ Create performance monitoring visualizations
- ✅ Add proactive error handling and recovery

## Schedule (8 hours)

### Morning Session (4 hours)

#### 🎯 Task 1: Model Health Dashboard Core (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Day 2 AI Insights Panel

**Deliverables**:
- Real-time health status indicators for all AI models
- Performance metrics dashboard with historical data
- Alert system for model degradation or failures
- Health score calculation and visualization

**Implementation**:
```javascript
// frontend/js/modules/modelHealthDashboard.js
class ModelHealthDashboard {
    constructor() {
        this.container = null;
        this.healthData = new Map();
        this.alertSystem = null;
        this.refreshInterval = 3000; // 3 seconds
        this.performanceHistory = new Map();
        this.thresholds = {
            responseTime: { warning: 1000, critical: 3000 },
            accuracy: { warning: 0.8, critical: 0.7 },
            availability: { warning: 95, critical: 90 },
            memoryUsage: { warning: 80, critical: 90 }
        };
    }

    initialize() {
        this.createDashboard();
        this.setupAlertSystem();
        this.startHealthMonitoring();
        this.setupEventListeners();
    }

    createDashboard() {
        const container = document.createElement('div');
        container.className = 'model-health-dashboard';
        container.innerHTML = `
            <div class="dashboard-header">
                <h3>Model Health Monitor</h3>
                <div class="dashboard-controls">
                    <button class="refresh-btn" title="Refresh Now">🔄</button>
                    <button class="settings-btn" title="Settings">⚙️</button>
                    <button class="alerts-btn" title="View Alerts">🚨</button>
                    <div class="auto-refresh-status">
                        <span class="status-dot active"></span>
                        <span class="status-text">Auto-refresh: ON</span>
                    </div>
                </div>
            </div>

            <div class="dashboard-content">
                <!-- Overall System Health -->
                <div class="system-health-overview">
                    <div class="health-score-card">
                        <div class="score-circle">
                            <svg viewBox="0 0 100 100" class="score-svg">
                                <circle cx="50" cy="50" r="45" class="score-bg"/>
                                <circle cx="50" cy="50" r="45" class="score-fill"/>
                            </svg>
                            <div class="score-text">
                                <span class="score-value">-</span>
                                <span class="score-label">Health</span>
                            </div>
                        </div>
                        <div class="health-summary">
                            <div class="summary-item">
                                <span class="label">Models Online:</span>
                                <span class="value online-count">-</span>
                            </div>
                            <div class="summary-item">
                                <span class="label">Avg Response:</span>
                                <span class="value avg-response">-</span>
                            </div>
                            <div class="summary-item">
                                <span class="label">Last Update:</span>
                                <span class="value last-update">-</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Individual Model Status -->
                <div class="models-grid">
                    <div class="model-card" data-model="classifier">
                        <div class="model-header">
                            <div class="model-info">
                                <h4 class="model-name">Logo Classifier</h4>
                                <span class="model-version">v2.1</span>
                            </div>
                            <div class="model-status">
                                <span class="status-indicator"></span>
                                <span class="status-text">-</span>
                            </div>
                        </div>
                        <div class="model-metrics">
                            <div class="metric-item">
                                <span class="metric-label">Response Time</span>
                                <span class="metric-value" data-metric="responseTime">-</span>
                                <div class="metric-trend" data-trend="responseTime"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Accuracy</span>
                                <span class="metric-value" data-metric="accuracy">-</span>
                                <div class="metric-trend" data-trend="accuracy"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Load</span>
                                <span class="metric-value" data-metric="load">-</span>
                                <div class="metric-bar">
                                    <div class="bar-fill" data-metric="load"></div>
                                </div>
                            </div>
                        </div>
                        <div class="model-actions">
                            <button class="action-btn details-btn">Details</button>
                            <button class="action-btn restart-btn">Restart</button>
                        </div>
                    </div>

                    <div class="model-card" data-model="predictor">
                        <div class="model-header">
                            <div class="model-info">
                                <h4 class="model-name">Quality Predictor</h4>
                                <span class="model-version">v1.8</span>
                            </div>
                            <div class="model-status">
                                <span class="status-indicator"></span>
                                <span class="status-text">-</span>
                            </div>
                        </div>
                        <div class="model-metrics">
                            <div class="metric-item">
                                <span class="metric-label">Response Time</span>
                                <span class="metric-value" data-metric="responseTime">-</span>
                                <div class="metric-trend" data-trend="responseTime"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Accuracy</span>
                                <span class="metric-value" data-metric="accuracy">-</span>
                                <div class="metric-trend" data-trend="accuracy"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Load</span>
                                <span class="metric-value" data-metric="load">-</span>
                                <div class="metric-bar">
                                    <div class="bar-fill" data-metric="load"></div>
                                </div>
                            </div>
                        </div>
                        <div class="model-actions">
                            <button class="action-btn details-btn">Details</button>
                            <button class="action-btn restart-btn">Restart</button>
                        </div>
                    </div>

                    <div class="model-card" data-model="optimizer">
                        <div class="model-header">
                            <div class="model-info">
                                <h4 class="model-name">Parameter Optimizer</h4>
                                <span class="model-version">v3.0</span>
                            </div>
                            <div class="model-status">
                                <span class="status-indicator"></span>
                                <span class="status-text">-</span>
                            </div>
                        </div>
                        <div class="model-metrics">
                            <div class="metric-item">
                                <span class="metric-label">Response Time</span>
                                <span class="metric-value" data-metric="responseTime">-</span>
                                <div class="metric-trend" data-trend="responseTime"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Success Rate</span>
                                <span class="metric-value" data-metric="successRate">-</span>
                                <div class="metric-trend" data-trend="successRate"></div>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Queue</span>
                                <span class="metric-value" data-metric="queue">-</span>
                                <div class="metric-bar">
                                    <div class="bar-fill" data-metric="queue"></div>
                                </div>
                            </div>
                        </div>
                        <div class="model-actions">
                            <button class="action-btn details-btn">Details</button>
                            <button class="action-btn restart-btn">Restart</button>
                        </div>
                    </div>
                </div>

                <!-- Performance Charts -->
                <div class="performance-charts">
                    <div class="chart-container">
                        <h4>Response Time Trends</h4>
                        <canvas id="responseTimeChart" width="400" height="200"></canvas>
                    </div>
                    <div class="chart-container">
                        <h4>Model Accuracy Trends</h4>
                        <canvas id="accuracyChart" width="400" height="200"></canvas>
                    </div>
                </div>

                <!-- Alert Panel -->
                <div class="alerts-panel hidden">
                    <div class="alerts-header">
                        <h4>System Alerts</h4>
                        <button class="close-alerts-btn">✕</button>
                    </div>
                    <div class="alerts-list"></div>
                </div>
            </div>
        `;

        // Insert into page
        const mainContent = document.getElementById('mainContent');
        if (mainContent) {
            this.container = container;
            const existingDashboard = mainContent.querySelector('.model-health-dashboard');
            if (existingDashboard) {
                existingDashboard.replaceWith(container);
            } else {
                mainContent.appendChild(container);
            }
        }
    }

    setupAlertSystem() {
        this.alertSystem = new ModelAlertSystem(this.container);
        this.alertSystem.initialize();

        // Subscribe to health data changes
        this.alertSystem.addThresholdRule('responseTime', (value, model) => {
            if (value > this.thresholds.responseTime.critical) {
                return {
                    level: 'critical',
                    message: `${model} response time critical: ${value}ms`,
                    action: 'Consider restarting the model service'
                };
            } else if (value > this.thresholds.responseTime.warning) {
                return {
                    level: 'warning',
                    message: `${model} response time elevated: ${value}ms`,
                    action: 'Monitor performance closely'
                };
            }
            return null;
        });

        this.alertSystem.addThresholdRule('accuracy', (value, model) => {
            if (value < this.thresholds.accuracy.critical) {
                return {
                    level: 'critical',
                    message: `${model} accuracy below threshold: ${(value * 100).toFixed(1)}%`,
                    action: 'Model may need retraining or restart'
                };
            } else if (value < this.thresholds.accuracy.warning) {
                return {
                    level: 'warning',
                    message: `${model} accuracy declining: ${(value * 100).toFixed(1)}%`,
                    action: 'Investigate model performance'
                };
            }
            return null;
        });
    }

    async startHealthMonitoring() {
        // Initial health check
        await this.refreshHealthData();

        // Set up regular monitoring
        this.monitoringInterval = setInterval(async () => {
            await this.refreshHealthData();
        }, this.refreshInterval);

        console.log('[Health] Monitoring started');
    }

    async refreshHealthData() {
        try {
            const response = await fetch('/api/model-health/detailed');
            const healthData = await response.json();

            this.updateHealthDisplay(healthData);
            this.updatePerformanceCharts(healthData);
            this.checkAlertConditions(healthData);

        } catch (error) {
            console.error('[Health] Failed to fetch health data:', error);
            this.handleHealthCheckFailure();
        }
    }

    updateHealthDisplay(healthData) {
        // Update overall system health score
        const overallScore = this.calculateOverallHealth(healthData);
        this.updateHealthScore(overallScore);

        // Update individual model status
        Object.entries(healthData.models || {}).forEach(([modelName, data]) => {
            this.updateModelCard(modelName, data);
            this.updatePerformanceHistory(modelName, data);
        });

        // Update summary stats
        this.updateSummaryStats(healthData);
    }

    calculateOverallHealth(healthData) {
        const models = Object.values(healthData.models || {});
        if (models.length === 0) return 0;

        const weights = {
            availability: 0.4,
            responseTime: 0.3,
            accuracy: 0.2,
            load: 0.1
        };

        let totalScore = 0;
        models.forEach(model => {
            let modelScore = 0;

            // Availability score (0-100)
            modelScore += (model.status === 'healthy' ? 100 : 0) * weights.availability;

            // Response time score (inverse, normalized)
            const responseScore = Math.max(0, 100 - (model.responseTime / 10));
            modelScore += responseScore * weights.responseTime;

            // Accuracy score (0-100)
            modelScore += (model.accuracy * 100) * weights.accuracy;

            // Load score (inverse)
            modelScore += (100 - model.load) * weights.load;

            totalScore += modelScore;
        });

        return Math.round(totalScore / models.length);
    }
}
```

**Testing Criteria**:
- [ ] Dashboard displays real-time health data
- [ ] Health score calculation is accurate
- [ ] Model status indicators update correctly
- [ ] Performance metrics show current values

#### 🎯 Task 2: Advanced Health Visualizations (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1

**Deliverables**:
- Real-time performance charts using Chart.js
- Historical trend analysis visualizations
- Interactive health timeline
- Predictive health indicators

**Implementation**:
```javascript
// frontend/js/modules/healthVisualizations.js
class HealthVisualizationEngine {
    constructor(container) {
        this.container = container;
        this.charts = new Map();
        this.chartConfig = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: { display: true, title: { display: true, text: 'Time' } },
                y: { display: true, title: { display: true, text: 'Value' } }
            }
        };
    }

    initialize() {
        this.loadChartLibrary().then(() => {
            this.createResponseTimeChart();
            this.createAccuracyChart();
            this.createLoadChart();
            this.createHealthTimelineChart();
        });
    }

    async loadChartLibrary() {
        if (window.Chart) return;

        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(script);

        return new Promise((resolve) => {
            script.onload = resolve;
        });
    }

    createResponseTimeChart() {
        const canvas = this.container.querySelector('#responseTimeChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Classifier',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Predictor',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Optimizer',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...this.chartConfig,
                scales: {
                    ...this.chartConfig.scales,
                    y: {
                        ...this.chartConfig.scales.y,
                        title: { display: true, text: 'Response Time (ms)' },
                        beginAtZero: true
                    }
                },
                plugins: {
                    ...this.chartConfig.plugins,
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: ${context.parsed.y}ms`;
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('responseTime', chart);
    }

    createAccuracyChart() {
        const canvas = this.container.querySelector('#accuracyChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Classifier Accuracy',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Predictor Accuracy',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...this.chartConfig,
                scales: {
                    ...this.chartConfig.scales,
                    y: {
                        ...this.chartConfig.scales.y,
                        title: { display: true, text: 'Accuracy (%)' },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    ...this.chartConfig.plugins,
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('accuracy', chart);
    }

    createLoadChart() {
        const canvas = document.createElement('canvas');
        canvas.id = 'loadChart';
        canvas.width = 400;
        canvas.height = 200;

        const chartsContainer = this.container.querySelector('.performance-charts');
        if (chartsContainer) {
            const loadContainer = document.createElement('div');
            loadContainer.className = 'chart-container';
            loadContainer.innerHTML = '<h4>System Load Distribution</h4>';
            loadContainer.appendChild(canvas);
            chartsContainer.appendChild(loadContainer);

            const ctx = canvas.getContext('2d');
            const chart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Classifier', 'Predictor', 'Optimizer', 'Available'],
                    datasets: [{
                        data: [0, 0, 0, 100],
                        backgroundColor: [
                            '#ef4444',
                            '#f97316',
                            '#eab308',
                            '#22c55e'
                        ],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    return `${context.label}: ${context.parsed}%`;
                                }
                            }
                        }
                    }
                }
            });

            this.charts.set('load', chart);
        }
    }

    createHealthTimelineChart() {
        const canvas = document.createElement('canvas');
        canvas.id = 'healthTimelineChart';
        canvas.width = 800;
        canvas.height = 150;

        const chartsContainer = this.container.querySelector('.performance-charts');
        if (chartsContainer) {
            const timelineContainer = document.createElement('div');
            timelineContainer.className = 'chart-container timeline-chart';
            timelineContainer.innerHTML = '<h4>Health Timeline (24h)</h4>';
            timelineContainer.appendChild(canvas);
            chartsContainer.appendChild(timelineContainer);

            const ctx = canvas.getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Overall Health Score',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Time (24h)' }
                        },
                        y: {
                            display: true,
                            title: { display: true, text: 'Health Score' },
                            min: 0,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    return `Health Score: ${context.parsed.y}%`;
                                }
                            }
                        }
                    }
                }
            });

            this.charts.set('healthTimeline', chart);
        }
    }

    updateChart(chartName, data) {
        const chart = this.charts.get(chartName);
        if (!chart) return;

        switch (chartName) {
            case 'responseTime':
                this.updateResponseTimeChart(chart, data);
                break;
            case 'accuracy':
                this.updateAccuracyChart(chart, data);
                break;
            case 'load':
                this.updateLoadChart(chart, data);
                break;
            case 'healthTimeline':
                this.updateHealthTimelineChart(chart, data);
                break;
        }
    }

    updateResponseTimeChart(chart, data) {
        const now = new Date().toLocaleTimeString();

        // Add new data point
        chart.data.labels.push(now);

        // Update each model's response time
        chart.data.datasets.forEach((dataset, index) => {
            const modelNames = ['classifier', 'predictor', 'optimizer'];
            const modelData = data.models[modelNames[index]];

            if (modelData) {
                dataset.data.push(modelData.responseTime || 0);
            }
        });

        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        chart.update('none'); // No animation for real-time updates
    }

    updateAccuracyChart(chart, data) {
        const now = new Date().toLocaleTimeString();

        chart.data.labels.push(now);

        // Update classifier and predictor accuracy
        const models = ['classifier', 'predictor'];
        chart.data.datasets.forEach((dataset, index) => {
            const modelData = data.models[models[index]];
            if (modelData) {
                dataset.data.push((modelData.accuracy || 0) * 100);
            }
        });

        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        chart.update('none');
    }

    updateLoadChart(chart, data) {
        const loads = [];
        const models = ['classifier', 'predictor', 'optimizer'];

        models.forEach(model => {
            const modelData = data.models[model];
            loads.push(modelData ? modelData.load || 0 : 0);
        });

        const totalLoad = loads.reduce((sum, load) => sum + load, 0);
        const available = Math.max(0, 100 - totalLoad);

        chart.data.datasets[0].data = [...loads, available];
        chart.update();
    }

    updateHealthTimelineChart(chart, healthScore) {
        const now = new Date().toLocaleTimeString();

        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(healthScore);

        // Keep 24 hours of data (assuming updates every 5 minutes = 288 points)
        if (chart.data.labels.length > 288) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update('none');
    }
}
```

**Testing Criteria**:
- [ ] Charts render correctly with real data
- [ ] Real-time updates animate smoothly
- [ ] Historical data displays accurately
- [ ] Performance impact is minimal

### Afternoon Session (4 hours)

#### 🎯 Task 3: Intelligent Fallback Mechanisms (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1, Task 2

**Deliverables**:
- Graceful degradation when AI models unavailable
- Fallback UI components with reduced functionality
- User notification system for service disruptions
- Automatic recovery detection and re-enablement

**Implementation**:
```javascript
// frontend/js/modules/fallbackManager.js
class FallbackManager {
    constructor() {
        this.fallbackState = {
            aiEnabled: true,
            availableModels: new Set(),
            fallbackReason: null,
            fallbackStartTime: null,
            recoveryAttempts: 0
        };

        this.fallbackStrategies = new Map();
        this.recoveryCallbacks = new Set();
        this.userNotified = false;
    }

    initialize() {
        this.setupFallbackStrategies();
        this.setupRecoveryMonitoring();
        this.setupUIFallbacks();
    }

    setupFallbackStrategies() {
        // Classifier fallback: Use rule-based classification
        this.fallbackStrategies.set('classifier', {
            name: 'Rule-based Classification',
            implementation: async (imageData) => {
                return await this.performRuleBasedClassification(imageData);
            },
            capabilities: ['basic_classification'],
            limitations: ['reduced_accuracy', 'no_confidence_scores']
        });

        // Quality predictor fallback: Use historical averages
        this.fallbackStrategies.set('predictor', {
            name: 'Historical Quality Estimation',
            implementation: async (logoType, parameters) => {
                return await this.estimateQualityFromHistory(logoType, parameters);
            },
            capabilities: ['quality_estimation'],
            limitations: ['reduced_accuracy', 'no_real_time_prediction']
        });

        // Optimizer fallback: Use predefined parameter sets
        this.fallbackStrategies.set('optimizer', {
            name: 'Preset Parameter Optimization',
            implementation: async (logoType, qualityTarget) => {
                return await this.getPresetParameters(logoType, qualityTarget);
            },
            capabilities: ['parameter_selection'],
            limitations: ['no_adaptive_optimization', 'limited_personalization']
        });
    }

    async handleModelFailure(modelName, error) {
        console.warn(`[Fallback] Model ${modelName} failed:`, error);

        // Remove failed model from available set
        this.fallbackState.availableModels.delete(modelName);

        // Check if we need to enter fallback mode
        if (this.shouldEnterFallbackMode()) {
            await this.enterFallbackMode(modelName, error);
        }

        // Attempt specific model recovery
        this.attemptModelRecovery(modelName);
    }

    shouldEnterFallbackMode() {
        // Enter fallback mode if any critical model is unavailable
        const criticalModels = ['classifier', 'predictor'];
        return criticalModels.some(model => !this.fallbackState.availableModels.has(model));
    }

    async enterFallbackMode(triggerModel, error) {
        if (!this.fallbackState.aiEnabled) return; // Already in fallback

        this.fallbackState.aiEnabled = false;
        this.fallbackState.fallbackReason = `${triggerModel} unavailable`;
        this.fallbackState.fallbackStartTime = Date.now();

        // Update UI to fallback mode
        this.updateUIForFallback();

        // Notify user
        this.notifyUserOfFallback(triggerModel, error);

        // Log fallback event
        this.logFallbackEvent(triggerModel, error);

        console.log('[Fallback] Entered fallback mode due to:', triggerModel);
    }

    updateUIForFallback() {
        // Show fallback indicator in header
        this.showFallbackIndicator();

        // Update AI insights panel
        this.updateAIInsightsPanelForFallback();

        // Modify conversion interface
        this.updateConversionInterfaceForFallback();

        // Disable AI-dependent features
        this.disableAIDependentFeatures();
    }

    showFallbackIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'fallback-indicator';
        indicator.innerHTML = `
            <div class="fallback-content">
                <span class="fallback-icon">⚠️</span>
                <div class="fallback-text">
                    <span class="fallback-title">Limited AI Features</span>
                    <span class="fallback-subtitle">Using basic conversion mode</span>
                </div>
                <button class="fallback-details-btn">Details</button>
            </div>
        `;

        // Insert at top of page
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(indicator, container.firstChild);
        }

        // Setup details button
        indicator.querySelector('.fallback-details-btn').addEventListener('click', () => {
            this.showFallbackDetailsModal();
        });
    }

    updateAIInsightsPanelForFallback() {
        const aiPanel = document.querySelector('.ai-insights-panel');
        if (!aiPanel) return;

        // Add fallback overlay
        const overlay = document.createElement('div');
        overlay.className = 'fallback-overlay';
        overlay.innerHTML = `
            <div class="fallback-message">
                <h4>AI Features Temporarily Unavailable</h4>
                <p>Using fallback processing methods:</p>
                <ul class="fallback-capabilities">
                    ${this.generateFallbackCapabilitiesList()}
                </ul>
                <div class="fallback-actions">
                    <button class="retry-ai-btn">Retry AI Connection</button>
                    <button class="learn-more-btn">Learn More</button>
                </div>
            </div>
        `;

        aiPanel.appendChild(overlay);

        // Setup action buttons
        overlay.querySelector('.retry-ai-btn').addEventListener('click', () => {
            this.attemptFullRecovery();
        });

        overlay.querySelector('.learn-more-btn').addEventListener('click', () => {
            this.showFallbackExplanationModal();
        });
    }

    generateFallbackCapabilitiesList() {
        const capabilities = [];

        this.fallbackStrategies.forEach((strategy, model) => {
            if (!this.fallbackState.availableModels.has(model)) {
                capabilities.push(`
                    <li>
                        <strong>${strategy.name}:</strong>
                        ${strategy.capabilities.join(', ')}
                    </li>
                `);
            }
        });

        return capabilities.join('');
    }

    async attemptModelRecovery(modelName) {
        const maxAttempts = 3;
        const baseDelay = 5000; // 5 seconds

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const delay = baseDelay * Math.pow(2, attempt - 1); // Exponential backoff

            await new Promise(resolve => setTimeout(resolve, delay));

            try {
                const isHealthy = await this.checkModelHealth(modelName);

                if (isHealthy) {
                    await this.recoverModel(modelName);
                    return;
                }
            } catch (error) {
                console.warn(`[Recovery] Attempt ${attempt} failed for ${modelName}:`, error);
            }
        }

        console.error(`[Recovery] Failed to recover ${modelName} after ${maxAttempts} attempts`);
    }

    async checkModelHealth(modelName) {
        try {
            const response = await fetch(`/api/model-health/${modelName}`);
            const health = await response.json();
            return health.status === 'healthy';
        } catch (error) {
            return false;
        }
    }

    async recoverModel(modelName) {
        this.fallbackState.availableModels.add(modelName);

        // Check if we can exit fallback mode
        if (this.canExitFallbackMode()) {
            await this.exitFallbackMode();
        } else {
            // Partial recovery - update UI accordingly
            this.updateUIForPartialRecovery(modelName);
        }

        console.log(`[Recovery] Model ${modelName} recovered successfully`);
    }

    canExitFallbackMode() {
        const criticalModels = ['classifier', 'predictor'];
        return criticalModels.every(model => this.fallbackState.availableModels.has(model));
    }

    async exitFallbackMode() {
        this.fallbackState.aiEnabled = true;
        this.fallbackState.fallbackReason = null;

        const fallbackDuration = Date.now() - this.fallbackState.fallbackStartTime;
        this.fallbackState.fallbackStartTime = null;

        // Remove fallback UI elements
        this.removeFallbackUI();

        // Restore AI features
        this.restoreAIFeatures();

        // Notify user of recovery
        this.notifyUserOfRecovery(fallbackDuration);

        console.log('[Recovery] Successfully exited fallback mode');
    }

    removeFallbackUI() {
        // Remove fallback indicator
        const indicator = document.querySelector('.fallback-indicator');
        if (indicator) {
            indicator.remove();
        }

        // Remove fallback overlays
        document.querySelectorAll('.fallback-overlay').forEach(overlay => {
            overlay.remove();
        });
    }

    restoreAIFeatures() {
        // Re-enable AI-dependent UI elements
        document.querySelectorAll('[data-ai-dependent]').forEach(element => {
            element.classList.remove('ai-disabled');
            element.removeAttribute('disabled');
        });

        // Refresh AI insights panel
        const aiPanel = document.querySelector('.ai-insights-panel');
        if (aiPanel) {
            // Trigger refresh of AI data
            window.dispatchEvent(new CustomEvent('aiRecovered'));
        }
    }

    // Fallback implementations
    async performRuleBasedClassification(imageData) {
        // Simple rule-based classification fallback
        const features = await this.extractBasicFeatures(imageData);

        if (features.colorCount <= 2) {
            return { type: 'simple', confidence: 0.7 };
        } else if (features.hasText) {
            return { type: 'text', confidence: 0.6 };
        } else if (features.colorCount > 10) {
            return { type: 'complex', confidence: 0.5 };
        } else {
            return { type: 'gradient', confidence: 0.5 };
        }
    }

    async estimateQualityFromHistory(logoType, parameters) {
        // Use stored historical averages
        const historyKey = `${logoType}_${JSON.stringify(parameters)}`;
        const stored = localStorage.getItem(`quality_history_${historyKey}`);

        if (stored) {
            const history = JSON.parse(stored);
            const average = history.reduce((sum, val) => sum + val, 0) / history.length;
            return { quality: average, confidence: 0.6 };
        }

        // Default estimates based on logo type
        const defaults = {
            simple: 0.95,
            text: 0.92,
            gradient: 0.85,
            complex: 0.78
        };

        return {
            quality: defaults[logoType] || 0.8,
            confidence: 0.4
        };
    }

    async getPresetParameters(logoType, qualityTarget) {
        // Return preset parameter combinations
        const presets = {
            simple: {
                color_precision: 3,
                corner_threshold: 30,
                path_precision: 5,
                layer_difference: 8
            },
            text: {
                color_precision: 2,
                corner_threshold: 20,
                path_precision: 8,
                layer_difference: 4
            },
            gradient: {
                color_precision: 8,
                corner_threshold: 45,
                path_precision: 6,
                layer_difference: 12
            },
            complex: {
                color_precision: 6,
                corner_threshold: 60,
                path_precision: 4,
                layer_difference: 16
            }
        };

        return presets[logoType] || presets.simple;
    }

    showFallbackDetailsModal() {
        const modal = document.createElement('div');
        modal.className = 'fallback-details-modal';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>AI Service Status</h3>
                    <button class="modal-close">✕</button>
                </div>
                <div class="modal-body">
                    <div class="service-status">
                        <h4>Current Status</h4>
                        <p><strong>Mode:</strong> Fallback Processing</p>
                        <p><strong>Reason:</strong> ${this.fallbackState.fallbackReason}</p>
                        <p><strong>Duration:</strong> ${this.getFallbackDuration()}</p>
                    </div>
                    <div class="available-features">
                        <h4>Available Features</h4>
                        ${this.generateFallbackCapabilitiesList()}
                    </div>
                    <div class="recovery-status">
                        <h4>Recovery Attempts</h4>
                        <p>Automatic recovery is in progress. You can continue using the converter with reduced AI features.</p>
                        <button class="manual-retry-btn">Retry Now</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Setup modal interactions
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.querySelector('.modal-backdrop').addEventListener('click', () => {
            modal.remove();
        });

        modal.querySelector('.manual-retry-btn').addEventListener('click', () => {
            this.attemptFullRecovery();
            modal.remove();
        });
    }
}
```

**Testing Criteria**:
- [ ] Fallback mode activates when models fail
- [ ] UI gracefully degrades functionality
- [ ] Recovery mechanisms work automatically
- [ ] User notifications are clear and helpful

#### 🎯 Task 4: Error Handling & Recovery (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1, Task 2, Task 3

**Deliverables**:
- Comprehensive error handling for all AI operations
- User-friendly error messages with actionable guidance
- Automatic retry mechanisms with exponential backoff
- Error reporting and analytics collection

**Implementation**:
```javascript
// frontend/js/modules/aiErrorHandler.js
class AIErrorHandler {
    constructor() {
        this.errorLog = [];
        this.retryStrategies = new Map();
        this.userNotifications = new Map();
        this.errorThresholds = {
            consecutiveErrors: 3,
            errorRate: 0.3, // 30% error rate threshold
            timeWindow: 300000 // 5 minutes
        };
    }

    initialize() {
        this.setupRetryStrategies();
        this.setupGlobalErrorHandling();
        this.setupErrorReporting();
    }

    setupRetryStrategies() {
        // Network errors - retry with exponential backoff
        this.retryStrategies.set('NetworkError', {
            maxRetries: 3,
            baseDelay: 1000,
            exponential: true,
            shouldRetry: (error, attempt) => attempt < 3
        });

        // Timeout errors - retry with increased timeout
        this.retryStrategies.set('TimeoutError', {
            maxRetries: 2,
            baseDelay: 2000,
            exponential: false,
            shouldRetry: (error, attempt) => attempt < 2
        });

        // Service unavailable - longer delays
        this.retryStrategies.set('ServiceUnavailable', {
            maxRetries: 5,
            baseDelay: 5000,
            exponential: true,
            shouldRetry: (error, attempt) => attempt < 5
        });

        // Invalid input - no retry
        this.retryStrategies.set('ValidationError', {
            maxRetries: 0,
            shouldRetry: () => false
        });
    }

    async handleAIError(error, context = {}) {
        const errorInfo = this.categorizeError(error);
        const errorId = this.logError(errorInfo, context);

        // Check if we should trigger fallback mode
        if (this.shouldTriggerFallback(errorInfo)) {
            await this.triggerFallbackMode(errorInfo);
            return { handled: true, fallback: true };
        }

        // Attempt retry if appropriate
        const retryResult = await this.attemptRetry(errorInfo, context);
        if (retryResult.success) {
            return { handled: true, retried: true, result: retryResult.data };
        }

        // Show user-friendly error message
        this.showUserError(errorInfo, context);

        // Report error for analytics
        this.reportError(errorId, errorInfo, context);

        return { handled: true, error: errorInfo };
    }

    categorizeError(error) {
        const errorInfo = {
            type: 'UnknownError',
            message: error.message || 'An unknown error occurred',
            original: error,
            timestamp: Date.now(),
            severity: 'medium'
        };

        // Network-related errors
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorInfo.type = 'NetworkError';
            errorInfo.severity = 'high';
            errorInfo.userMessage = 'Connection to AI services failed. Please check your internet connection.';
            errorInfo.actionable = 'Retry in a moment or continue with basic conversion.';
        }

        // Timeout errors
        else if (error.name === 'AbortError' || error.message.includes('timeout')) {
            errorInfo.type = 'TimeoutError';
            errorInfo.severity = 'medium';
            errorInfo.userMessage = 'AI processing took too long and was cancelled.';
            errorInfo.actionable = 'Try again with a simpler image or use basic conversion.';
        }

        // Service unavailable
        else if (error.status === 503 || error.message.includes('Service Unavailable')) {
            errorInfo.type = 'ServiceUnavailable';
            errorInfo.severity = 'high';
            errorInfo.userMessage = 'AI services are temporarily unavailable.';
            errorInfo.actionable = 'Using fallback conversion methods automatically.';
        }

        // Rate limiting
        else if (error.status === 429) {
            errorInfo.type = 'RateLimited';
            errorInfo.severity = 'medium';
            errorInfo.userMessage = 'Too many requests. Please wait a moment.';
            errorInfo.actionable = 'Retry in 30 seconds or use basic conversion.';
        }

        // Validation errors
        else if (error.status === 400) {
            errorInfo.type = 'ValidationError';
            errorInfo.severity = 'low';
            errorInfo.userMessage = 'Invalid input provided to AI service.';
            errorInfo.actionable = 'Please try with a different image.';
        }

        // Model errors
        else if (error.message.includes('model') || error.message.includes('prediction')) {
            errorInfo.type = 'ModelError';
            errorInfo.severity = 'high';
            errorInfo.userMessage = 'AI model encountered an error.';
            errorInfo.actionable = 'Switching to alternative processing method.';
        }

        return errorInfo;
    }

    async attemptRetry(errorInfo, context) {
        const strategy = this.retryStrategies.get(errorInfo.type);
        if (!strategy || !strategy.shouldRetry(errorInfo, context.attempt || 0)) {
            return { success: false, reason: 'No retry strategy' };
        }

        const attempt = (context.attempt || 0) + 1;
        const delay = strategy.exponential
            ? strategy.baseDelay * Math.pow(2, attempt - 1)
            : strategy.baseDelay;

        console.log(`[AI Error] Retrying ${errorInfo.type} (attempt ${attempt}) after ${delay}ms`);

        // Show retry notification to user
        this.showRetryNotification(attempt, delay);

        await new Promise(resolve => setTimeout(resolve, delay));

        try {
            // Re-execute the original operation
            if (context.operation && typeof context.operation === 'function') {
                const result = await context.operation({ ...context, attempt });
                this.hideRetryNotification();
                return { success: true, data: result };
            }
        } catch (retryError) {
            console.warn(`[AI Error] Retry ${attempt} failed:`, retryError);
            return await this.attemptRetry(
                this.categorizeError(retryError),
                { ...context, attempt }
            );
        }

        return { success: false, reason: 'Retry failed' };
    }

    showUserError(errorInfo, context) {
        const notification = document.createElement('div');
        notification.className = `ai-error-notification ${errorInfo.severity}`;
        notification.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <span class="error-icon">${this.getErrorIcon(errorInfo.severity)}</span>
                    <span class="error-title">${errorInfo.userMessage}</span>
                    <button class="error-dismiss">✕</button>
                </div>
                ${errorInfo.actionable ? `
                    <div class="error-action">
                        <span class="action-text">${errorInfo.actionable}</span>
                    </div>
                ` : ''}
                <div class="error-actions">
                    ${this.generateErrorActions(errorInfo, context)}
                </div>
            </div>
        `;

        // Insert notification
        const container = this.getNotificationContainer();
        container.appendChild(notification);

        // Auto-dismiss after delay
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, this.getNotificationDuration(errorInfo.severity));

        // Setup dismiss handler
        notification.querySelector('.error-dismiss').addEventListener('click', () => {
            notification.remove();
        });
    }

    generateErrorActions(errorInfo, context) {
        const actions = [];

        // Retry action
        if (this.retryStrategies.has(errorInfo.type)) {
            actions.push(`
                <button class="error-action-btn retry-btn" data-action="retry">
                    🔄 Retry
                </button>
            `);
        }

        // Fallback action
        if (errorInfo.type !== 'ValidationError') {
            actions.push(`
                <button class="error-action-btn fallback-btn" data-action="fallback">
                    ⚡ Use Basic Mode
                </button>
            `);
        }

        // Report action
        actions.push(`
            <button class="error-action-btn report-btn" data-action="report">
                📝 Report Issue
            </button>
        `);

        return actions.join('');
    }

    getErrorIcon(severity) {
        const icons = {
            low: '⚠️',
            medium: '❌',
            high: '🚨'
        };
        return icons[severity] || '❌';
    }

    getNotificationDuration(severity) {
        const durations = {
            low: 5000,
            medium: 8000,
            high: 12000
        };
        return durations[severity] || 8000;
    }

    logError(errorInfo, context) {
        const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const logEntry = {
            id: errorId,
            ...errorInfo,
            context,
            userAgent: navigator.userAgent,
            url: window.location.href,
            timestamp: Date.now()
        };

        this.errorLog.push(logEntry);

        // Keep only last 100 errors in memory
        if (this.errorLog.length > 100) {
            this.errorLog.shift();
        }

        console.error('[AI Error Log]', logEntry);
        return errorId;
    }

    async reportError(errorId, errorInfo, context) {
        try {
            await fetch('/api/error-reporting', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    errorId,
                    type: errorInfo.type,
                    message: errorInfo.message,
                    severity: errorInfo.severity,
                    context: {
                        userAgent: navigator.userAgent,
                        timestamp: errorInfo.timestamp,
                        operation: context.operation?.name,
                        ...context
                    }
                })
            });
        } catch (reportingError) {
            console.warn('[Error Reporting] Failed to report error:', reportingError);
        }
    }

    shouldTriggerFallback(errorInfo) {
        // Check consecutive errors
        const recentErrors = this.errorLog.slice(-this.errorThresholds.consecutiveErrors);
        const consecutiveHighSeverity = recentErrors.every(e => e.severity === 'high');

        // Check error rate in time window
        const windowStart = Date.now() - this.errorThresholds.timeWindow;
        const errorsInWindow = this.errorLog.filter(e => e.timestamp > windowStart);
        const errorRate = errorsInWindow.length / (this.errorThresholds.timeWindow / 60000); // errors per minute

        return consecutiveHighSeverity ||
               errorRate > this.errorThresholds.errorRate ||
               errorInfo.type === 'ServiceUnavailable';
    }

    getNotificationContainer() {
        let container = document.querySelector('.ai-error-notifications');
        if (!container) {
            container = document.createElement('div');
            container.className = 'ai-error-notifications';
            document.body.appendChild(container);
        }
        return container;
    }
}
```

**Testing Criteria**:
- [ ] Error categorization works correctly
- [ ] Retry mechanisms function as expected
- [ ] User notifications are helpful and actionable
- [ ] Error reporting captures necessary information

## End of Day Validation

### Functionality Checklist
- [ ] Model health dashboard displays real-time status
- [ ] Performance visualizations update correctly
- [ ] Fallback mechanisms activate appropriately
- [ ] Error handling covers all scenarios
- [ ] Recovery processes work automatically

### Performance Targets
- [ ] Health monitoring: <500ms refresh time
- [ ] Chart updates: <100ms render time
- [ ] Fallback activation: <1 second transition
- [ ] Error recovery: <5 seconds average time

### User Experience
- [ ] Health status is immediately visible
- [ ] Fallback mode is clearly communicated
- [ ] Error messages are actionable
- [ ] Recovery notifications are informative

## Tomorrow's Preparation
- [ ] Test fallback scenarios thoroughly
- [ ] Prepare enhanced conversion interface design
- [ ] Plan quality prediction display integration
- [ ] Review accessibility requirements

## Success Metrics
- Model health monitoring provides complete visibility
- Intelligent fallback ensures continuous service availability
- Error handling guides users through problems effectively
- Recovery mechanisms minimize service disruption
- Performance monitoring enables proactive maintenance