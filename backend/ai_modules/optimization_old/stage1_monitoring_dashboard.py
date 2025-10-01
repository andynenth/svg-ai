"""
Stage 1 Real-Time Monitoring Dashboard
Component for Task B7.2 - DAY7 PPO Agent Training

Provides specialized real-time monitoring dashboard for Stage 1 training with:
- Live training progress visualization
- Real-time quality metrics tracking
- Validation progress monitoring
- Quality assurance alerts display
- Milestone achievement notifications
- Training health indicators
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import websockets
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics for Stage 1 training"""
    timestamp: float
    episode: int

    # Training metrics
    current_reward: float
    avg_reward_100: float
    current_quality: float
    avg_quality_100: float
    success_rate_100: float

    # Validation metrics
    latest_validation_quality: float = 0.0
    latest_validation_success_rate: float = 0.0
    latest_validation_ssim: float = 0.0

    # Progress metrics
    episode_progress: float = 0.0
    estimated_completion_hours: float = 0.0
    milestones_achieved: int = 0

    # Health metrics
    training_stable: bool = True
    qa_alerts_count: int = 0
    critical_alerts_count: int = 0


class Stage1VisualizationEngine:
    """Real-time visualization engine for Stage 1 training"""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.plots_dir = save_dir / "live_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Data storage for plots
        self.training_data = deque(maxlen=1000)
        self.validation_data = deque(maxlen=100)
        self.milestone_data = []
        self.alert_data = deque(maxlen=50)

        # Plot configuration
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def update_training_data(self, metrics: DashboardMetrics):
        """Update training data for visualization"""
        self.training_data.append({
            'timestamp': metrics.timestamp,
            'episode': metrics.episode,
            'reward': metrics.current_reward,
            'quality': metrics.current_quality,
            'avg_reward': metrics.avg_reward_100,
            'avg_quality': metrics.avg_quality_100,
            'success_rate': metrics.success_rate_100
        })

    def update_validation_data(self, episode: int, validation_result: Dict[str, Any]):
        """Update validation data for visualization"""
        self.validation_data.append({
            'timestamp': time.time(),
            'episode': episode,
            'quality': validation_result.get('avg_quality', 0),
            'success_rate': validation_result.get('success_rate', 0),
            'ssim_improvement': validation_result.get('ssim_improvement', 0)
        })

    def add_milestone(self, milestone: Dict[str, Any]):
        """Add milestone to visualization"""
        self.milestone_data.append(milestone)

    def add_alert(self, alert: Dict[str, Any]):
        """Add alert to visualization"""
        self.alert_data.append(alert)

    def create_live_dashboard(self) -> str:
        """Create comprehensive live dashboard visualization"""
        if not self.training_data:
            return ""

        fig = plt.figure(figsize=(20, 12))

        # Create subplot grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Training Progress Overview (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_progress(ax1)

        # 2. Quality Metrics (top-right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_quality_metrics(ax2)

        # 3. Validation Performance (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_validation_performance(ax3)

        # 4. Success Rate Trend (middle-center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_success_rate_trend(ax4)

        # 5. Training Health (middle-center-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_training_health(ax5)

        # 6. Milestones & Alerts (middle-right)
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_milestones_and_alerts(ax6)

        # 7. Episode Rewards Distribution (bottom-left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_reward_distribution(ax7)

        # 8. Quality vs Success Rate Correlation (bottom-center-left)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_quality_success_correlation(ax8)

        # 9. Training Velocity (bottom-center-right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_training_velocity(ax9)

        # 10. System Health Summary (bottom-right)
        ax10 = fig.add_subplot(gs[2, 3])
        self._plot_system_health_summary(ax10)

        # Add overall title with current status
        latest_data = list(self.training_data)[-1]
        current_time = datetime.fromtimestamp(latest_data['timestamp']).strftime('%H:%M:%S')

        fig.suptitle(
            f'Stage 1 Training Dashboard - Episode {latest_data["episode"]} - {current_time}\n'
            f'Quality: {latest_data["quality"]:.4f} | Reward: {latest_data["reward"]:.2f} | '
            f'Success Rate: {latest_data["success_rate"]:.1%}',
            fontsize=16, fontweight='bold'
        )

        # Save dashboard
        dashboard_file = self.plots_dir / f"stage1_dashboard_{int(time.time())}.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(dashboard_file)

    def _plot_training_progress(self, ax):
        """Plot training progress over time"""
        if len(self.training_data) < 2:
            ax.text(0.5, 0.5, 'Waiting for training data...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.training_data)
        episodes = [d['episode'] for d in data]
        rewards = [d['reward'] for d in data]
        avg_rewards = [d['avg_reward'] for d in data]

        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        ax.plot(episodes, avg_rewards, color='darkblue', linewidth=2, label='100-Episode Average')

        ax.set_title('Training Progress - Rewards', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_quality_metrics(self, ax):
        """Plot quality metrics over time"""
        if len(self.training_data) < 2:
            ax.text(0.5, 0.5, 'Waiting for quality data...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.training_data)
        episodes = [d['episode'] for d in data]
        qualities = [d['quality'] for d in data]
        avg_qualities = [d['avg_quality'] for d in data]

        ax.plot(episodes, qualities, alpha=0.3, color='green', label='Episode Quality')
        ax.plot(episodes, avg_qualities, color='darkgreen', linewidth=2, label='100-Episode Average')

        # Add target line
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Target Quality (0.85)')

        ax.set_title('Quality Metrics - SSIM', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Quality (SSIM)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_validation_performance(self, ax):
        """Plot validation performance"""
        if not self.validation_data:
            ax.text(0.5, 0.5, 'Waiting for\nvalidation data...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.validation_data)
        episodes = [d['episode'] for d in data]
        qualities = [d['quality'] for d in data]

        ax.plot(episodes, qualities, 'o-', color='purple', linewidth=2, markersize=6)
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7)

        ax.set_title('Validation Quality', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Quality')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_success_rate_trend(self, ax):
        """Plot success rate trend"""
        if len(self.training_data) < 10:
            ax.text(0.5, 0.5, 'Calculating\nsuccess rate...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.training_data)[-50:]  # Last 50 episodes
        episodes = [d['episode'] for d in data]
        success_rates = [d['success_rate'] for d in data]

        ax.plot(episodes, success_rates, color='orange', linewidth=2)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (80%)')

        ax.set_title('Success Rate Trend', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_training_health(self, ax):
        """Plot training health indicators"""
        # Create health score based on recent performance
        if len(self.training_data) < 10:
            health_score = 0.5
            health_status = "Initializing"
            health_color = 'yellow'
        else:
            recent_data = list(self.training_data)[-10:]

            # Calculate stability metrics
            quality_stability = 1 - np.std([d['quality'] for d in recent_data])
            reward_trend = np.polyfit(range(len(recent_data)), [d['reward'] for d in recent_data], 1)[0]
            trend_health = min(1, max(0, reward_trend + 0.5))

            health_score = (quality_stability + trend_health) / 2

            if health_score > 0.8:
                health_status = "Excellent"
                health_color = 'green'
            elif health_score > 0.6:
                health_status = "Good"
                health_color = 'lightgreen'
            elif health_score > 0.4:
                health_status = "Fair"
                health_color = 'yellow'
            else:
                health_status = "Concerning"
                health_color = 'red'

        # Create health gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)

        ax.plot(theta, r, 'k-', linewidth=2)

        # Health indicator
        health_angle = health_score * np.pi
        ax.plot([health_angle], [0.8], 'o', markersize=15, color=health_color)

        ax.set_title(f'Training Health\n{health_status} ({health_score:.2f})', fontweight='bold')
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['Poor', 'Fair', 'Good', 'Great', 'Excellent'])
        ax.set_yticks([])

    def _plot_milestones_and_alerts(self, ax):
        """Plot milestones and alerts summary"""
        # Milestones
        milestone_count = len(self.milestone_data)
        alert_count = len(self.alert_data)

        categories = ['Milestones\nAchieved', 'Recent\nAlerts']
        values = [milestone_count, alert_count]
        colors = ['gold', 'red' if alert_count > 0 else 'lightgray']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(value), ha='center', va='bottom', fontweight='bold')

        ax.set_title('Status Summary', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_reward_distribution(self, ax):
        """Plot reward distribution"""
        if len(self.training_data) < 10:
            ax.text(0.5, 0.5, 'Collecting\nreward data...', ha='center', va='center', transform=ax.transAxes)
            return

        rewards = [d['reward'] for d in list(self.training_data)[-100:]]

        ax.hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')

        ax.set_title('Reward Distribution\n(Last 100 Episodes)', fontweight='bold')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_quality_success_correlation(self, ax):
        """Plot quality vs success rate correlation"""
        if len(self.training_data) < 20:
            ax.text(0.5, 0.5, 'Building\ncorrelation\nanalysis...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.training_data)[-50:]  # Last 50 episodes
        qualities = [d['quality'] for d in data]
        success_rates = [d['success_rate'] for d in data]

        ax.scatter(qualities, success_rates, alpha=0.6, color='purple')

        # Add trend line
        if len(qualities) > 5:
            z = np.polyfit(qualities, success_rates, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(qualities), max(qualities), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)

        ax.set_title('Quality vs Success\nCorrelation', fontweight='bold')
        ax.set_xlabel('Quality')
        ax.set_ylabel('Success Rate')
        ax.grid(True, alpha=0.3)

    def _plot_training_velocity(self, ax):
        """Plot training velocity (episodes per minute)"""
        if len(self.training_data) < 10:
            ax.text(0.5, 0.5, 'Calculating\ntraining\nvelocity...', ha='center', va='center', transform=ax.transAxes)
            return

        data = list(self.training_data)[-20:]  # Last 20 episodes

        if len(data) < 2:
            return

        time_diffs = []
        for i in range(1, len(data)):
            time_diff = data[i]['timestamp'] - data[i-1]['timestamp']
            if time_diff > 0:
                time_diffs.append(60 / time_diff)  # Episodes per minute

        if time_diffs:
            avg_velocity = np.mean(time_diffs)

            # Create velocity gauge
            max_velocity = 10  # Assume max 10 episodes per minute
            velocity_ratio = min(avg_velocity / max_velocity, 1.0)

            theta = np.linspace(0, 2*np.pi, 100)
            r = np.ones_like(theta)

            # Background circle
            ax.plot(theta, r, 'lightgray', linewidth=3)

            # Velocity indicator
            velocity_theta = velocity_ratio * 2 * np.pi
            velocity_arc = theta[theta <= velocity_theta]
            velocity_r = r[theta <= velocity_theta]
            ax.plot(velocity_arc, velocity_r, 'blue', linewidth=5)

            # Center text
            ax.text(0, 0, f'{avg_velocity:.1f}\nep/min', ha='center', va='center',
                   fontsize=12, fontweight='bold')

        ax.set_title('Training Velocity', fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_system_health_summary(self, ax):
        """Plot system health summary"""
        # Create system health indicators
        indicators = ['Memory', 'CPU', 'Training', 'Quality']

        # Simulate health scores (in real implementation, get from monitoring)
        health_scores = [0.8, 0.7, 0.9, 0.85]  # Mock data
        colors = ['green' if score > 0.8 else 'yellow' if score > 0.6 else 'red' for score in health_scores]

        y_pos = np.arange(len(indicators))
        bars = ax.barh(y_pos, health_scores, color=colors, alpha=0.7)

        # Add percentage labels
        for i, (bar, score) in enumerate(zip(bars, health_scores)):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.1%}', va='center', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(indicators)
        ax.set_xlim(0, 1)
        ax.set_title('System Health', fontweight='bold')
        ax.set_xlabel('Health Score')
        ax.grid(True, alpha=0.3, axis='x')


class Stage1MonitoringDashboard:
    """Real-time monitoring dashboard for Stage 1 training"""

    def __init__(self, save_dir: str = "models/stage1_training/monitoring",
                 websocket_port: int = 8768,
                 update_interval: float = 5.0):
        """
        Initialize Stage 1 monitoring dashboard

        Args:
            save_dir: Directory to save monitoring data and visualizations
            websocket_port: Port for WebSocket server
            update_interval: Dashboard update interval in seconds
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.websocket_port = websocket_port
        self.update_interval = update_interval

        # Components
        self.visualization_engine = Stage1VisualizationEngine(self.save_dir)

        # Data storage
        self.dashboard_metrics = deque(maxlen=1000)
        self.connected_clients = set()

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.websocket_server = None

        # Dashboard generation
        self.last_dashboard_update = 0
        self.dashboard_update_interval = 30  # Update dashboard every 30 seconds

        logger.info(f"Stage 1 Monitoring Dashboard initialized on port {websocket_port}")

    async def start_monitoring(self):
        """Start monitoring dashboard"""
        logger.info("🖥️ Starting Stage 1 Monitoring Dashboard")

        # Start WebSocket server
        self.websocket_server = await websockets.serve(
            self.handle_client_connection,
            "localhost",
            self.websocket_port
        )

        # Start monitoring loop
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self.monitoring_loop())

        logger.info(f"Dashboard available at: ws://localhost:{self.websocket_port}")
        logger.info("WebSocket clients can connect for real-time updates")

    async def stop_monitoring(self):
        """Stop monitoring dashboard"""
        logger.info("🛑 Stopping Stage 1 Monitoring Dashboard")

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

        # Save final dashboard
        self.generate_dashboard()

    async def handle_client_connection(self, websocket, path):
        """Handle WebSocket client connections"""
        self.connected_clients.add(websocket)
        logger.info(f"Dashboard client connected: {websocket.remote_address}")

        try:
            # Send initial data
            if self.dashboard_metrics:
                latest_metrics = list(self.dashboard_metrics)[-1]
                await websocket.send(json.dumps({
                    'type': 'initial_data',
                    'metrics': asdict(latest_metrics),
                    'timestamp': time.time()
                }))

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {websocket.remote_address}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Dashboard client disconnected: {websocket.remote_address}")

    async def handle_client_message(self, websocket, message: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        msg_type = message.get('type')

        if msg_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
        elif msg_type == 'request_dashboard':
            # Generate and send dashboard
            dashboard_path = self.generate_dashboard()
            await websocket.send(json.dumps({
                'type': 'dashboard_generated',
                'dashboard_path': dashboard_path,
                'timestamp': time.time()
            }))
        elif msg_type == 'request_metrics_history':
            # Send metrics history
            history = [asdict(m) for m in list(self.dashboard_metrics)[-100:]]
            await websocket.send(json.dumps({
                'type': 'metrics_history',
                'metrics': history,
                'timestamp': time.time()
            }))

    async def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check if dashboard needs update
                current_time = time.time()
                if current_time - self.last_dashboard_update > self.dashboard_update_interval:
                    dashboard_path = self.generate_dashboard()
                    self.last_dashboard_update = current_time

                    # Broadcast dashboard update
                    await self.broadcast_to_clients({
                        'type': 'dashboard_updated',
                        'dashboard_path': dashboard_path,
                        'timestamp': current_time
                    })

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.connected_clients:
            return

        json_message = json.dumps(message)
        disconnected = set()

        for client in self.connected_clients:
            try:
                await client.send(json_message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)

    def update_metrics(self, episode: int, training_metrics: Dict[str, Any],
                      validation_result: Optional[Dict[str, Any]] = None,
                      milestones_count: int = 0, alerts_count: int = 0):
        """Update dashboard metrics"""

        # Calculate moving averages
        recent_rewards = [m.current_reward for m in list(self.dashboard_metrics)[-100:]]
        recent_qualities = [m.current_quality for m in list(self.dashboard_metrics)[-100:]]
        recent_success = [m.success_rate_100 for m in list(self.dashboard_metrics)[-100:]]

        metrics = DashboardMetrics(
            timestamp=time.time(),
            episode=episode,
            current_reward=training_metrics.get('reward', 0),
            avg_reward_100=np.mean(recent_rewards) if recent_rewards else training_metrics.get('reward', 0),
            current_quality=training_metrics.get('quality', 0),
            avg_quality_100=np.mean(recent_qualities) if recent_qualities else training_metrics.get('quality', 0),
            success_rate_100=np.mean(recent_success) if recent_success else 0,
            latest_validation_quality=validation_result.get('avg_quality', 0) if validation_result else 0,
            latest_validation_success_rate=validation_result.get('success_rate', 0) if validation_result else 0,
            latest_validation_ssim=validation_result.get('ssim_improvement', 0) if validation_result else 0,
            episode_progress=episode / 5000,  # Assuming 5000 target episodes
            milestones_achieved=milestones_count,
            qa_alerts_count=alerts_count,
            training_stable=alerts_count == 0
        )

        self.dashboard_metrics.append(metrics)

        # Update visualization engine
        self.visualization_engine.update_training_data(metrics)

        if validation_result:
            self.visualization_engine.update_validation_data(episode, validation_result)

        # Broadcast real-time update
        if self.connected_clients:
            asyncio.create_task(self.broadcast_to_clients({
                'type': 'metrics_update',
                'metrics': asdict(metrics),
                'timestamp': time.time()
            }))

    def add_milestone(self, milestone: Dict[str, Any]):
        """Add milestone to dashboard"""
        self.visualization_engine.add_milestone(milestone)

        # Broadcast milestone
        if self.connected_clients:
            asyncio.create_task(self.broadcast_to_clients({
                'type': 'milestone_achieved',
                'milestone': milestone,
                'timestamp': time.time()
            }))

    def add_alert(self, alert: Dict[str, Any]):
        """Add alert to dashboard"""
        self.visualization_engine.add_alert(alert)

        # Broadcast alert
        if self.connected_clients:
            asyncio.create_task(self.broadcast_to_clients({
                'type': 'qa_alert',
                'alert': alert,
                'timestamp': time.time()
            }))

    def generate_dashboard(self) -> str:
        """Generate comprehensive dashboard visualization"""
        if not self.dashboard_metrics:
            logger.warning("No metrics available for dashboard generation")
            return ""

        try:
            dashboard_path = self.visualization_engine.create_live_dashboard()
            logger.info(f"📊 Dashboard generated: {dashboard_path}")
            return dashboard_path
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            return ""

    def save_monitoring_data(self):
        """Save monitoring data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_file = self.save_dir / f"dashboard_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.dashboard_metrics], f, indent=2)

        logger.info(f"Monitoring data saved to: {metrics_file}")

    def generate_monitoring_report(self) -> str:
        """Generate monitoring report"""
        if not self.dashboard_metrics:
            return "No monitoring data available"

        latest_metrics = list(self.dashboard_metrics)[-1]
        all_metrics = list(self.dashboard_metrics)

        # Calculate summary statistics
        all_rewards = [m.current_reward for m in all_metrics]
        all_qualities = [m.current_quality for m in all_metrics]

        report = [
            "# Stage 1 Monitoring Dashboard Report",
            "=" * 50,
            "",
            "## Current Status",
            f"- Episode: {latest_metrics.episode}",
            f"- Progress: {latest_metrics.episode_progress:.1%}",
            f"- Current Quality: {latest_metrics.current_quality:.4f}",
            f"- Current Reward: {latest_metrics.current_reward:.2f}",
            f"- Success Rate (100-ep avg): {latest_metrics.success_rate_100:.1%}",
            f"- Milestones Achieved: {latest_metrics.milestones_achieved}",
            f"- QA Alerts: {latest_metrics.qa_alerts_count}",
            "",
            "## Performance Summary",
            f"- Best Quality: {max(all_qualities):.4f}",
            f"- Average Quality: {np.mean(all_qualities):.4f}",
            f"- Best Reward: {max(all_rewards):.2f}",
            f"- Average Reward: {np.mean(all_rewards):.2f}",
            "",
            "## Validation Performance",
            f"- Latest Validation Quality: {latest_metrics.latest_validation_quality:.4f}",
            f"- Latest Validation Success Rate: {latest_metrics.latest_validation_success_rate:.1%}",
            f"- Latest SSIM Improvement: {latest_metrics.latest_validation_ssim:.4f}",
            "",
            f"Dashboard data saved to: {self.save_dir}",
            f"WebSocket clients connected: {len(self.connected_clients)}"
        ]

        return "\n".join(report)


# Factory function for easy usage
def create_stage1_dashboard(save_dir: str = "models/stage1_training/monitoring",
                           websocket_port: int = 8768) -> Stage1MonitoringDashboard:
    """
    Factory function to create Stage 1 monitoring dashboard

    Args:
        save_dir: Directory to save monitoring data
        websocket_port: Port for WebSocket server

    Returns:
        Configured Stage 1 monitoring dashboard
    """
    return Stage1MonitoringDashboard(
        save_dir=save_dir,
        websocket_port=websocket_port
    )


# Example usage and integration
async def main():
    """Example Stage 1 dashboard usage"""
    dashboard = create_stage1_dashboard(
        save_dir="models/stage1_demo_monitoring",
        websocket_port=8768
    )

    try:
        # Start dashboard
        await dashboard.start_monitoring()

        # Simulate training updates
        for episode in range(100):
            # Simulate training metrics
            training_metrics = {
                'reward': 5 + np.random.normal(0, 1),
                'quality': 0.7 + 0.002 * episode + np.random.normal(0, 0.05),
                'success': np.random.random() > 0.3
            }

            # Simulate validation every 10 episodes
            validation_result = None
            if episode % 10 == 0:
                validation_result = {
                    'avg_quality': 0.75 + 0.001 * episode + np.random.normal(0, 0.03),
                    'success_rate': min(0.9, 0.5 + 0.005 * episode),
                    'ssim_improvement': 0.7 + 0.003 * episode + np.random.normal(0, 0.05)
                }

            # Update dashboard
            dashboard.update_metrics(episode, training_metrics, validation_result)

            # Simulate milestones
            if episode in [25, 50, 75]:
                milestone = {
                    'milestone_type': f'episode_{episode}',
                    'episode': episode,
                    'description': f'Episode {episode} milestone reached',
                    'value': episode
                }
                dashboard.add_milestone(milestone)

            await asyncio.sleep(0.1)  # Fast simulation

        # Keep dashboard running for demo
        await asyncio.sleep(10)

    finally:
        await dashboard.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())