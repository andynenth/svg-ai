# Day 2: AI Insights Panel Implementation

## Overview
Build the comprehensive AI Insights Panel that displays rich metadata, processing information, and AI model outputs in an intuitive, visually appealing interface.

## Daily Objectives
- ✅ Design and implement AI Insights Panel UI
- ✅ Create interactive data visualizations for AI metrics
- ✅ Implement real-time updates and live data streaming
- ✅ Add expandable sections for detailed analysis

## Schedule (8 hours)

### Morning Session (4 hours)

#### 🎯 Task 1: AI Insights Panel Structure (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Day 1 foundation

**Deliverables**:
- Main panel container with collapsible sections
- Header with model status indicators
- Tab-based navigation for different insight types
- Responsive layout with smooth animations

**Implementation**:
```javascript
// frontend/js/modules/aiInsights.js
class AIInsightsPanel {
    constructor() {
        this.container = null;
        this.activeTab = 'overview';
        this.isCollapsed = false;
        this.currentData = null;
        this.refreshInterval = null;
    }

    initialize() {
        this.createPanelStructure();
        this.setupEventListeners();
        this.setupAutoRefresh();
    }

    createPanelStructure() {
        const container = document.createElement('div');
        container.className = 'ai-insights-panel';
        container.innerHTML = `
            <div class="ai-panel-header">
                <div class="ai-panel-title">
                    <h3>AI Insights</h3>
                    <div class="model-status-indicators">
                        <div class="status-indicator" data-model="classifier">
                            <span class="status-dot"></span>
                            <span class="status-label">Classifier</span>
                        </div>
                        <div class="status-indicator" data-model="predictor">
                            <span class="status-dot"></span>
                            <span class="status-label">Predictor</span>
                        </div>
                        <div class="status-indicator" data-model="optimizer">
                            <span class="status-dot"></span>
                            <span class="status-label">Optimizer</span>
                        </div>
                    </div>
                </div>
                <div class="ai-panel-controls">
                    <button class="refresh-btn" title="Refresh Data">🔄</button>
                    <button class="collapse-btn" title="Collapse Panel">➖</button>
                </div>
            </div>

            <div class="ai-panel-content">
                <div class="ai-panel-tabs">
                    <button class="tab-btn active" data-tab="overview">Overview</button>
                    <button class="tab-btn" data-tab="processing">Processing</button>
                    <button class="tab-btn" data-tab="quality">Quality</button>
                    <button class="tab-btn" data-tab="optimization">Optimization</button>
                </div>

                <div class="ai-panel-body">
                    <div class="tab-content active" data-tab="overview">
                        <div class="overview-grid">
                            <div class="insight-card" data-insight="logo-type">
                                <div class="card-header">
                                    <h4>Logo Type</h4>
                                    <span class="confidence-badge"></span>
                                </div>
                                <div class="card-content">
                                    <div class="logo-type-display">
                                        <span class="type-name">-</span>
                                        <div class="type-visualization"></div>
                                    </div>
                                </div>
                            </div>

                            <div class="insight-card" data-insight="quality-prediction">
                                <div class="card-header">
                                    <h4>Quality Prediction</h4>
                                    <span class="accuracy-badge"></span>
                                </div>
                                <div class="card-content">
                                    <div class="quality-meter-container">
                                        <div class="quality-meter">
                                            <div class="meter-fill"></div>
                                        </div>
                                        <div class="quality-values">
                                            <span class="predicted">Predicted: -</span>
                                            <span class="actual hidden">Actual: -</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="insight-card" data-insight="processing-tier">
                                <div class="card-header">
                                    <h4>Processing Tier</h4>
                                    <span class="tier-badge"></span>
                                </div>
                                <div class="card-content">
                                    <div class="tier-display">
                                        <div class="tier-selector">
                                            <div class="tier-option" data-tier="1">Basic</div>
                                            <div class="tier-option" data-tier="2">Enhanced</div>
                                            <div class="tier-option" data-tier="3">Premium</div>
                                        </div>
                                        <div class="tier-benefits"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" data-tab="processing">
                        <div class="processing-timeline">
                            <div class="timeline-header">
                                <h4>Processing Pipeline</h4>
                                <div class="processing-time">
                                    <span class="time-label">Total Time:</span>
                                    <span class="time-value">-</span>
                                </div>
                            </div>
                            <div class="timeline-steps"></div>
                        </div>
                    </div>

                    <div class="tab-content" data-tab="quality">
                        <div class="quality-analysis">
                            <div class="metrics-grid">
                                <div class="metric-item">
                                    <span class="metric-label">SSIM Score</span>
                                    <span class="metric-value" data-metric="ssim">-</span>
                                    <div class="metric-bar">
                                        <div class="bar-fill" data-metric="ssim"></div>
                                    </div>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">MSE</span>
                                    <span class="metric-value" data-metric="mse">-</span>
                                    <div class="metric-bar">
                                        <div class="bar-fill" data-metric="mse"></div>
                                    </div>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">PSNR</span>
                                    <span class="metric-value" data-metric="psnr">-</span>
                                    <div class="metric-bar">
                                        <div class="bar-fill" data-metric="psnr"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="quality-improvements">
                                <h4>AI Improvements</h4>
                                <div class="improvement-list"></div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-content" data-tab="optimization">
                        <div class="optimization-details">
                            <div class="method-display">
                                <h4>Optimization Method</h4>
                                <div class="method-info">
                                    <span class="method-name">-</span>
                                    <span class="method-description">-</span>
                                </div>
                            </div>
                            <div class="parameter-adjustments">
                                <h4>Parameter Adjustments</h4>
                                <div class="parameter-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert into page
        const mainContent = document.getElementById('mainContent');
        if (mainContent) {
            this.container = container;
            mainContent.appendChild(container);
        }
    }

    setupEventListeners() {
        if (!this.container) return;

        // Tab switching
        this.container.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Panel controls
        const refreshBtn = this.container.querySelector('.refresh-btn');
        const collapseBtn = this.container.querySelector('.collapse-btn');

        refreshBtn?.addEventListener('click', () => this.refreshData());
        collapseBtn?.addEventListener('click', () => this.toggleCollapse());

        // Tier selection
        this.container.querySelectorAll('.tier-option').forEach(option => {
            option.addEventListener('click', (e) => {
                this.selectTier(e.target.dataset.tier);
            });
        });
    }

    switchTab(tabName) {
        // Update tab buttons
        this.container.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update tab content
        this.container.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.tab === tabName);
        });

        this.activeTab = tabName;
    }

    updateInsights(data) {
        if (!data || !this.container) return;

        this.currentData = data;
        this.updateModelStatus(data.modelHealth);
        this.updateOverviewTab(data);
        this.updateProcessingTab(data);
        this.updateQualityTab(data);
        this.updateOptimizationTab(data);
    }
}
```

**Testing Criteria**:
- [ ] Panel structure renders correctly
- [ ] Tab switching works smoothly
- [ ] Responsive design adapts to screen size
- [ ] Animations are smooth and performant

#### 🎯 Task 2: Data Visualization Components (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1

**Deliverables**:
- Quality meter with animated progress
- Processing timeline visualization
- Parameter adjustment charts
- Confidence indicators and badges

**Implementation**:
```javascript
// frontend/js/modules/aiVisualizations.js
class AIVisualizationComponents {
    static createQualityMeter(predicted, actual = null) {
        const container = document.createElement('div');
        container.className = 'quality-meter-enhanced';

        const percentage = Math.round(predicted * 100);
        const color = this.getQualityColor(predicted);

        container.innerHTML = `
            <div class="meter-circle">
                <svg viewBox="0 0 100 100" class="meter-svg">
                    <circle cx="50" cy="50" r="45" class="meter-bg"/>
                    <circle cx="50" cy="50" r="45" class="meter-fill"
                            style="--percentage: ${percentage}; --color: ${color}"/>
                </svg>
                <div class="meter-text">
                    <span class="percentage">${percentage}%</span>
                    <span class="label">Quality</span>
                </div>
            </div>
            <div class="meter-details">
                <div class="predicted-value">
                    <span class="label">Predicted:</span>
                    <span class="value">${predicted.toFixed(3)}</span>
                </div>
                ${actual ? `
                    <div class="actual-value">
                        <span class="label">Actual:</span>
                        <span class="value">${actual.toFixed(3)}</span>
                    </div>
                    <div class="accuracy">
                        <span class="label">Accuracy:</span>
                        <span class="value">${this.calculateAccuracy(predicted, actual)}%</span>
                    </div>
                ` : ''}
            </div>
        `;

        return container;
    }

    static createProcessingTimeline(steps) {
        const container = document.createElement('div');
        container.className = 'processing-timeline-viz';

        const totalTime = steps.reduce((sum, step) => sum + (step.duration || 0), 0);

        container.innerHTML = `
            <div class="timeline-track">
                ${steps.map((step, index) => `
                    <div class="timeline-step ${step.completed ? 'completed' : 'pending'}"
                         style="--duration: ${step.duration || 0}ms">
                        <div class="step-marker">
                            <div class="step-icon">${this.getStepIcon(step.type)}</div>
                            <div class="step-number">${index + 1}</div>
                        </div>
                        <div class="step-content">
                            <div class="step-title">${step.name}</div>
                            <div class="step-details">
                                ${step.duration ? `<span class="duration">${step.duration}ms</span>` : ''}
                                ${step.status ? `<span class="status ${step.status}">${step.status}</span>` : ''}
                            </div>
                        </div>
                        <div class="step-progress" style="--progress: ${step.progress || 0}%"></div>
                    </div>
                `).join('')}
            </div>
            <div class="timeline-summary">
                <span class="total-time">Total: ${totalTime}ms</span>
                <span class="completed-steps">${steps.filter(s => s.completed).length}/${steps.length} steps</span>
            </div>
        `;

        return container;
    }

    static createParameterChart(parameters) {
        const container = document.createElement('div');
        container.className = 'parameter-chart';

        container.innerHTML = `
            <div class="chart-header">
                <h5>Parameter Optimizations</h5>
                <div class="chart-legend">
                    <span class="legend-item original">Original</span>
                    <span class="legend-item optimized">Optimized</span>
                </div>
            </div>
            <div class="chart-body">
                ${Object.entries(parameters).map(([param, data]) => `
                    <div class="parameter-row">
                        <div class="param-label">${param}</div>
                        <div class="param-bars">
                            <div class="param-bar original"
                                 style="--value: ${data.original}; --max: ${data.max}">
                                <span class="bar-value">${data.original}</span>
                            </div>
                            <div class="param-bar optimized"
                                 style="--value: ${data.optimized}; --max: ${data.max}">
                                <span class="bar-value">${data.optimized}</span>
                            </div>
                        </div>
                        <div class="param-improvement">
                            ${this.calculateImprovement(data.original, data.optimized)}%
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        return container;
    }

    static createConfidenceBadge(confidence, context = '') {
        const percentage = Math.round(confidence * 100);
        const level = this.getConfidenceLevel(confidence);

        const badge = document.createElement('span');
        badge.className = `confidence-badge ${level}`;
        badge.innerHTML = `
            <span class="confidence-icon">${this.getConfidenceIcon(level)}</span>
            <span class="confidence-text">${percentage}%</span>
            ${context ? `<span class="confidence-context">${context}</span>` : ''}
        `;

        return badge;
    }

    static getQualityColor(value) {
        if (value >= 0.9) return '#22c55e'; // Green
        if (value >= 0.8) return '#eab308'; // Yellow
        if (value >= 0.7) return '#f97316'; // Orange
        return '#ef4444'; // Red
    }

    static getStepIcon(type) {
        const icons = {
            upload: '📤',
            classify: '🔍',
            predict: '🎯',
            optimize: '⚙️',
            convert: '🔄',
            validate: '✅'
        };
        return icons[type] || '●';
    }

    static getConfidenceLevel(confidence) {
        if (confidence >= 0.9) return 'high';
        if (confidence >= 0.7) return 'medium';
        return 'low';
    }

    static getConfidenceIcon(level) {
        const icons = { high: '🟢', medium: '🟡', low: '🔴' };
        return icons[level] || '⚪';
    }

    static calculateAccuracy(predicted, actual) {
        const error = Math.abs(predicted - actual);
        const accuracy = Math.max(0, 100 - (error * 100));
        return Math.round(accuracy);
    }

    static calculateImprovement(original, optimized) {
        const improvement = ((optimized - original) / original) * 100;
        return improvement > 0 ? `+${improvement.toFixed(1)}` : improvement.toFixed(1);
    }
}
```

**Testing Criteria**:
- [ ] Quality meter animates smoothly
- [ ] Timeline shows processing steps correctly
- [ ] Parameter charts display comparisons accurately
- [ ] Confidence badges update in real-time

### Afternoon Session (4 hours)

#### 🎯 Task 3: Real-time Data Integration (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1, Task 2

**Deliverables**:
- WebSocket connection for live updates
- Real-time model health monitoring
- Progressive data loading for large insights
- Automatic refresh capabilities

**Implementation**:
```javascript
// frontend/js/modules/aiDataService.js
class AIDataService {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.subscribers = new Map();
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 5;
        this.reconnectAttempts = 0;
    }

    async initialize() {
        this.setupWebSocket();
        this.setupPollingFallback();
        this.startModelHealthMonitoring();
    }

    setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/ai-insights`;

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('[AI Data] WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.emit('connection', { status: 'connected' });
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleRealtimeUpdate(data);
                } catch (error) {
                    console.error('[AI Data] Failed to parse WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('[AI Data] WebSocket disconnected');
                this.isConnected = false;
                this.emit('connection', { status: 'disconnected' });
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('[AI Data] WebSocket error:', error);
                this.emit('error', { error, source: 'websocket' });
            };

        } catch (error) {
            console.warn('[AI Data] WebSocket setup failed, using polling fallback');
            this.setupPollingFallback();
        }
    }

    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'model_health':
                this.emit('modelHealth', data.payload);
                break;
            case 'processing_update':
                this.emit('processingUpdate', data.payload);
                break;
            case 'quality_update':
                this.emit('qualityUpdate', data.payload);
                break;
            case 'insights_update':
                this.emit('insightsUpdate', data.payload);
                break;
            default:
                console.warn('[AI Data] Unknown message type:', data.type);
        }
    }

    setupPollingFallback() {
        // Fallback to HTTP polling if WebSocket fails
        setInterval(async () => {
            if (!this.isConnected) {
                await this.pollModelHealth();
                await this.pollActiveProcessing();
            }
        }, 5000);
    }

    async pollModelHealth() {
        try {
            const response = await fetch('/api/model-health');
            const health = await response.json();
            this.emit('modelHealth', health);
        } catch (error) {
            console.warn('[AI Data] Model health polling failed:', error);
        }
    }

    async pollActiveProcessing() {
        try {
            const response = await fetch('/api/processing-status');
            const status = await response.json();
            this.emit('processingUpdate', status);
        } catch (error) {
            console.warn('[AI Data] Processing status polling failed:', error);
        }
    }

    async getHistoricalInsights(fileId, options = {}) {
        try {
            const params = new URLSearchParams({
                file_id: fileId,
                include_predictions: options.includePredictions || false,
                include_optimization: options.includeOptimization || false,
                ...options
            });

            const response = await fetch(`/api/insights/historical?${params}`);
            const insights = await response.json();

            return insights;
        } catch (error) {
            console.error('[AI Data] Failed to fetch historical insights:', error);
            throw error;
        }
    }

    subscribe(event, callback) {
        if (!this.subscribers.has(event)) {
            this.subscribers.set(event, new Set());
        }
        this.subscribers.get(event).add(callback);

        return () => {
            this.subscribers.get(event).delete(callback);
        };
    }

    emit(event, data) {
        if (this.subscribers.has(event)) {
            this.subscribers.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`[AI Data] Subscriber error for ${event}:`, error);
                }
            });
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`[AI Data] Reconnection attempt ${this.reconnectAttempts}`);
                this.setupWebSocket();
            }, this.reconnectInterval * this.reconnectAttempts);
        }
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
    }
}
```

**Testing Criteria**:
- [ ] WebSocket connection establishes successfully
- [ ] Real-time updates display without delay
- [ ] Fallback to polling works when WebSocket fails
- [ ] No memory leaks from event listeners

#### 🎯 Task 4: Interactive Features (2 hours)
**Status**: Pending
**Estimated**: 2 hours
**Dependencies**: Task 1, Task 2, Task 3

**Deliverables**:
- Expandable sections with smooth animations
- Interactive tooltips with detailed explanations
- Manual tier selection interface
- Export functionality for insights data

**Implementation**:
```javascript
// frontend/js/modules/aiInteractions.js
class AIInteractionController {
    constructor(insightsPanel, dataService) {
        this.panel = insightsPanel;
        this.dataService = dataService;
        this.tooltips = new Map();
        this.expandedSections = new Set();
    }

    initialize() {
        this.setupTooltips();
        this.setupExpandableSections();
        this.setupTierSelection();
        this.setupExportFunctionality();
        this.setupKeyboardNavigation();
    }

    setupTooltips() {
        // Create tooltip system for AI explanations
        const tooltipContainer = document.createElement('div');
        tooltipContainer.className = 'ai-tooltip-container';
        document.body.appendChild(tooltipContainer);

        this.panel.container.addEventListener('mouseenter', (e) => {
            if (e.target.hasAttribute('data-tooltip')) {
                this.showTooltip(e.target, e.target.dataset.tooltip);
            }
        }, true);

        this.panel.container.addEventListener('mouseleave', (e) => {
            if (e.target.hasAttribute('data-tooltip')) {
                this.hideTooltip();
            }
        }, true);
    }

    showTooltip(element, content) {
        const tooltip = document.createElement('div');
        tooltip.className = 'ai-tooltip';

        // Enhanced tooltip content based on context
        const enhancedContent = this.enhanceTooltipContent(content, element);
        tooltip.innerHTML = enhancedContent;

        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + (rect.width / 2)}px`;
        tooltip.style.top = `${rect.bottom + 8}px`;

        document.querySelector('.ai-tooltip-container').appendChild(tooltip);

        // Animate in
        requestAnimationFrame(() => {
            tooltip.classList.add('visible');
        });

        this.currentTooltip = tooltip;
    }

    enhanceTooltipContent(content, element) {
        const type = element.dataset.tooltipType || 'info';
        const context = element.dataset.context || '';

        const enhancements = {
            confidence: (content) => `
                <div class="tooltip-header">
                    <span class="tooltip-icon">🎯</span>
                    <span class="tooltip-title">Confidence Score</span>
                </div>
                <div class="tooltip-content">
                    <p>${content}</p>
                    <div class="confidence-scale">
                        <span class="scale-label">Low</span>
                        <div class="scale-bar">
                            <div class="scale-markers">
                                <span>0%</span><span>50%</span><span>100%</span>
                            </div>
                        </div>
                        <span class="scale-label">High</span>
                    </div>
                </div>
            `,
            quality: (content) => `
                <div class="tooltip-header">
                    <span class="tooltip-icon">📊</span>
                    <span class="tooltip-title">Quality Metrics</span>
                </div>
                <div class="tooltip-content">
                    <p>${content}</p>
                    <div class="metric-explanations">
                        <div class="metric-explain">
                            <strong>SSIM:</strong> Structural similarity (0-1, higher is better)
                        </div>
                        <div class="metric-explain">
                            <strong>MSE:</strong> Mean squared error (lower is better)
                        </div>
                        <div class="metric-explain">
                            <strong>PSNR:</strong> Peak signal-to-noise ratio (higher is better)
                        </div>
                    </div>
                </div>
            `,
            processing: (content) => `
                <div class="tooltip-header">
                    <span class="tooltip-icon">⚙️</span>
                    <span class="tooltip-title">Processing Details</span>
                </div>
                <div class="tooltip-content">
                    <p>${content}</p>
                    <div class="processing-tips">
                        <h5>Optimization Tips:</h5>
                        <ul>
                            <li>Higher tiers provide better quality but take longer</li>
                            <li>AI automatically selects optimal parameters</li>
                            <li>Manual override available in Advanced mode</li>
                        </ul>
                    </div>
                </div>
            `
        };

        return enhancements[type] ? enhancements[type](content) : `
            <div class="tooltip-content">
                <p>${content}</p>
            </div>
        `;
    }

    setupExpandableSections() {
        this.panel.container.querySelectorAll('[data-expandable]').forEach(section => {
            const header = section.querySelector('.section-header');
            if (header) {
                header.addEventListener('click', () => {
                    this.toggleSection(section);
                });
            }
        });
    }

    toggleSection(section) {
        const isExpanded = section.classList.contains('expanded');
        const sectionId = section.dataset.section;

        if (isExpanded) {
            section.classList.remove('expanded');
            this.expandedSections.delete(sectionId);
        } else {
            section.classList.add('expanded');
            this.expandedSections.add(sectionId);

            // Load additional data if needed
            this.loadSectionData(sectionId);
        }
    }

    async loadSectionData(sectionId) {
        const loaders = {
            'detailed-analysis': async () => {
                // Load detailed analysis data
                const data = await this.dataService.getDetailedAnalysis();
                this.updateDetailedAnalysisSection(data);
            },
            'parameter-history': async () => {
                // Load parameter optimization history
                const history = await this.dataService.getParameterHistory();
                this.updateParameterHistorySection(history);
            },
            'model-comparisons': async () => {
                // Load model comparison data
                const comparisons = await this.dataService.getModelComparisons();
                this.updateModelComparisonsSection(comparisons);
            }
        };

        if (loaders[sectionId]) {
            try {
                await loaders[sectionId]();
            } catch (error) {
                console.error(`Failed to load section data for ${sectionId}:`, error);
            }
        }
    }

    setupTierSelection() {
        const tierSelector = this.panel.container.querySelector('.tier-selector');
        if (!tierSelector) return;

        tierSelector.addEventListener('click', (e) => {
            if (e.target.classList.contains('tier-option')) {
                this.selectTier(e.target.dataset.tier);
            }
        });
    }

    async selectTier(tierLevel) {
        try {
            // Update UI immediately
            this.updateTierDisplay(tierLevel);

            // Send tier selection to backend
            const response = await fetch('/api/processing-tier', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tier: parseInt(tierLevel),
                    file_id: this.panel.currentData?.fileId
                })
            });

            if (!response.ok) {
                throw new Error('Failed to update processing tier');
            }

            const result = await response.json();
            this.panel.updateInsights(result);

        } catch (error) {
            console.error('Tier selection failed:', error);
            // Revert UI changes
            this.revertTierDisplay();
        }
    }

    setupExportFunctionality() {
        const exportBtn = document.createElement('button');
        exportBtn.className = 'export-insights-btn';
        exportBtn.innerHTML = '📊 Export Insights';
        exportBtn.title = 'Export AI insights as JSON';

        exportBtn.addEventListener('click', () => {
            this.exportInsights();
        });

        const controls = this.panel.container.querySelector('.ai-panel-controls');
        if (controls) {
            controls.appendChild(exportBtn);
        }
    }

    exportInsights() {
        if (!this.panel.currentData) {
            alert('No insights data available to export');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            fileId: this.panel.currentData.fileId,
            insights: this.panel.currentData,
            userPreferences: enhancedAppState.getState('preferences')
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-insights-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    setupKeyboardNavigation() {
        this.panel.container.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                // Handle tab navigation
                this.handleTabNavigation(e);
            } else if (e.key === 'Enter' || e.key === ' ') {
                // Handle activation
                this.handleActivation(e);
            } else if (e.key === 'Escape') {
                // Hide tooltips
                this.hideTooltip();
            }
        });
    }

    handleTabNavigation(e) {
        // Custom tab navigation for better accessibility
        const focusableElements = this.panel.container.querySelectorAll(
            'button, [tabindex="0"], .tab-btn, .tier-option'
        );

        const currentIndex = Array.from(focusableElements).indexOf(e.target);

        if (e.shiftKey) {
            // Previous element
            const prevIndex = currentIndex > 0 ? currentIndex - 1 : focusableElements.length - 1;
            focusableElements[prevIndex].focus();
        } else {
            // Next element
            const nextIndex = currentIndex < focusableElements.length - 1 ? currentIndex + 1 : 0;
            focusableElements[nextIndex].focus();
        }

        e.preventDefault();
    }
}
```

**Testing Criteria**:
- [ ] Tooltips display relevant information accurately
- [ ] Expandable sections animate smoothly
- [ ] Tier selection updates immediately
- [ ] Export functionality generates valid JSON
- [ ] Keyboard navigation works correctly

## End of Day Validation

### Functionality Checklist
- [ ] AI Insights Panel displays all required sections
- [ ] Data visualizations render correctly and update in real-time
- [ ] Interactive features respond to user input
- [ ] Real-time data integration works with WebSocket and HTTP fallback
- [ ] Export functionality generates comprehensive insights data

### Performance Targets
- [ ] Panel loads in <1 second
- [ ] Real-time updates display within 200ms
- [ ] Animations run at 60fps
- [ ] Memory usage remains stable during extended use

### Accessibility
- [ ] All interactive elements have proper ARIA labels
- [ ] Keyboard navigation works for all features
- [ ] Screen readers can access all information
- [ ] Color contrast meets WCAG 2.1 AA standards

## Tomorrow's Preparation
- [ ] Test panel with various data scenarios
- [ ] Prepare model health monitoring integration
- [ ] Plan fallback UI implementations
- [ ] Review Agent 1's API response formats

## Success Metrics
- AI Insights Panel provides comprehensive view of AI processing
- Real-time data updates enhance user experience
- Interactive features make AI insights accessible and actionable
- Export functionality enables data analysis and reporting
- Accessibility standards met for inclusive design