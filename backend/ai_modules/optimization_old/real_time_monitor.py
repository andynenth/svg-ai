"""
Real-time Monitoring Infrastructure for PPO Training Pipeline
Component 3 of Task B7.1 - DAY7 PPO Agent Training

Provides real-time monitoring, WebSocket support, live dashboard updates,
training callbacks, alerting system, and health monitoring for PPO training.
"""

import asyncio
import json
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import websockets
import psutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    timestamp: float
    episode: int
    reward: float
    episode_length: int
    quality: float
    success: bool

    # Loss metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None

    # Performance metrics
    fps: Optional[float] = None
    learning_rate: Optional[float] = None

    # Environment specific
    ssim_improvement: Optional[float] = None
    parameters_used: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System health metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    disk_usage: float = 0.0

    # Training specific
    training_active: bool = False
    episodes_completed: int = 0
    training_time: float = 0.0


@dataclass
class Alert:
    """Training alert"""
    timestamp: float
    level: str  # 'info', 'warning', 'error', 'critical'
    category: str  # 'training', 'system', 'performance'
    message: str
    details: Dict[str, Any]
    resolved: bool = False


class WebSocketServer:
    """WebSocket server for real-time data streaming"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.running = False

    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

        # Send initial connection confirmation
        await websocket.send(json.dumps({
            'type': 'connection',
            'status': 'connected',
            'timestamp': time.time()
        }))

    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return

        # Add timestamp to message
        message['timestamp'] = time.time()

        # Convert to JSON
        json_message = json.dumps(message, default=str)

        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(json_message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Handle incoming messages if needed
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from {websocket.remote_address}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def handle_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming message from client"""
        msg_type = message.get('type')

        if msg_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
        elif msg_type == 'subscribe':
            # Handle subscription to specific data streams
            pass
        elif msg_type == 'unsubscribe':
            # Handle unsubscription
            pass

    async def start_server(self):
        """Start WebSocket server"""
        self.running = True
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop_server(self):
        """Stop WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")


class AlertSystem:
    """Training alerting system"""

    def __init__(self, max_alerts: int = 1000):
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_counts = defaultdict(int)

        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'reward_stagnation_episodes': 100,
            'quality_drop_threshold': 0.1,
            'loss_explosion_threshold': 10.0,
            'training_failure_threshold': 5
        }

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)

    def create_alert(self, level: str, category: str, message: str, details: Dict[str, Any] = None):
        """Create and process new alert"""
        alert = Alert(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
            details=details or {}
        )

        self.alerts.append(alert)
        self.alert_counts[f"{level}_{category}"] += 1

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.log(
            logging.ERROR if level in ['error', 'critical'] else logging.WARNING,
            f"ALERT [{level.upper()}] {category}: {message}"
        )

        return alert

    def check_system_alerts(self, system_metrics: SystemMetrics):
        """Check for system-related alerts"""
        if system_metrics.cpu_percent > self.thresholds['cpu_percent']:
            self.create_alert(
                'warning', 'system',
                f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                {'cpu_percent': system_metrics.cpu_percent}
            )

        if system_metrics.memory_percent > self.thresholds['memory_percent']:
            self.create_alert(
                'warning', 'system',
                f"High memory usage: {system_metrics.memory_percent:.1f}%",
                {'memory_percent': system_metrics.memory_percent}
            )

    def check_training_alerts(self, training_metrics: List[TrainingMetrics]):
        """Check for training-related alerts"""
        if len(training_metrics) < 10:
            return

        recent_metrics = list(training_metrics)[-10:]

        # Check for reward stagnation
        recent_rewards = [m.reward for m in recent_metrics]
        if len(set([round(r, 2) for r in recent_rewards])) == 1:
            self.create_alert(
                'warning', 'training',
                "Reward stagnation detected",
                {'recent_rewards': recent_rewards}
            )

        # Check for quality drops
        if len(recent_metrics) >= 2:
            quality_drop = recent_metrics[-2].quality - recent_metrics[-1].quality
            if quality_drop > self.thresholds['quality_drop_threshold']:
                self.create_alert(
                    'warning', 'training',
                    f"Quality drop detected: {quality_drop:.3f}",
                    {'quality_drop': quality_drop}
                )

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get alerts from last N minutes"""
        cutoff = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]

    def get_alert_summary(self) -> Dict[str, int]:
        """Get alert count summary"""
        return dict(self.alert_counts)


class PerformanceMonitor:
    """System performance monitoring"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.metrics_history = deque(maxlen=1000)

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Try to get GPU metrics if available
        gpu_memory_used = None
        gpu_memory_total = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory_used = gpu.memoryUsed
                gpu_memory_total = gpu.memoryTotal
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            disk_usage=disk.percent
        )

    def start_monitoring(self):
        """Start system monitoring"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.update_interval)

    def get_recent_metrics(self, minutes: int = 10) -> List[SystemMetrics]:
        """Get system metrics from last N minutes"""
        cutoff = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp > cutoff]


class RealTimeTrainingCallback(BaseCallback):
    """Real-time training callback for PPO monitoring"""

    def __init__(self, monitor: 'RealTimeMonitor', verbose: int = 0):
        super().__init__(verbose)
        self.monitor = monitor
        self.episode_count = 0
        self.step_count = 0

    def _on_training_start(self) -> None:
        """Called when training starts"""
        self.monitor.on_training_start()

    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout"""
        pass

    def _on_step(self) -> bool:
        """Called after each environment step"""
        self.step_count += 1

        # Extract step information
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1

                    # Create training metrics
                    metrics = TrainingMetrics(
                        timestamp=time.time(),
                        episode=self.episode_count,
                        reward=info['episode']['r'],
                        episode_length=info['episode']['l'],
                        quality=info.get('quality', 0.0),
                        success=info.get('success', False),
                        ssim_improvement=info.get('ssim_improvement', 0.0),
                        parameters_used=info.get('parameters', {})
                    )

                    # Send to monitor
                    self.monitor.on_episode_complete(metrics)

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Extract training metrics
        if hasattr(self.model, 'logger') and self.model.logger:
            logger_data = self.model.logger.name_to_value

            # Update with loss information
            if hasattr(self.monitor, 'current_metrics'):
                self.monitor.current_metrics.policy_loss = logger_data.get('train/policy_loss')
                self.monitor.current_metrics.value_loss = logger_data.get('train/value_loss')
                self.monitor.current_metrics.entropy = logger_data.get('train/entropy_loss')
                self.monitor.current_metrics.learning_rate = logger_data.get('train/learning_rate')

    def _on_training_end(self) -> None:
        """Called when training ends"""
        self.monitor.on_training_end()


class RealTimeMonitor:
    """Main real-time monitoring system"""

    def __init__(self,
                 websocket_port: int = 8765,
                 update_interval: float = 1.0,
                 save_dir: str = "logs/real_time_monitoring"):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.websocket_server = WebSocketServer(port=websocket_port)
        self.alert_system = AlertSystem()
        self.performance_monitor = PerformanceMonitor(update_interval)

        # Data storage
        self.training_metrics = deque(maxlen=10000)
        self.system_metrics = deque(maxlen=1000)

        # State tracking
        self.training_active = False
        self.training_start_time = None
        self.current_metrics = None

        # Async event loop
        self.loop = None
        self.monitoring_task = None

        # Setup alert handlers
        self.alert_system.add_alert_handler(self._handle_alert)

        logger.info("Real-time monitor initialized")

    def _handle_alert(self, alert: Alert):
        """Handle generated alerts"""
        # Broadcast alert via WebSocket
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.websocket_server.broadcast({
                    'type': 'alert',
                    'alert': asdict(alert)
                }),
                self.loop
            )

    async def start_monitoring(self):
        """Start all monitoring components"""
        # Start WebSocket server
        await self.websocket_server.start_server()

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        # Start monitoring loop
        self.loop = asyncio.get_event_loop()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Real-time monitoring started")

    async def stop_monitoring(self):
        """Stop all monitoring components"""
        # Stop monitoring loop
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()

        # Stop WebSocket server
        await self.websocket_server.stop_server()

        logger.info("Real-time monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Get system metrics
                system_metrics = self.performance_monitor.get_system_metrics()
                system_metrics.training_active = self.training_active
                system_metrics.episodes_completed = len(self.training_metrics)

                if self.training_start_time:
                    system_metrics.training_time = time.time() - self.training_start_time

                self.system_metrics.append(system_metrics)

                # Check for alerts
                self.alert_system.check_system_alerts(system_metrics)
                if len(self.training_metrics) > 0:
                    self.alert_system.check_training_alerts(list(self.training_metrics))

                # Broadcast system metrics
                await self.websocket_server.broadcast({
                    'type': 'system_metrics',
                    'metrics': asdict(system_metrics)
                })

                # Broadcast training summary
                if self.training_metrics:
                    recent_metrics = list(self.training_metrics)[-10:]
                    summary = self._create_training_summary(recent_metrics)
                    await self.websocket_server.broadcast({
                        'type': 'training_summary',
                        'summary': summary
                    })

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)

    def _create_training_summary(self, metrics: List[TrainingMetrics]) -> Dict[str, Any]:
        """Create training summary from recent metrics"""
        if not metrics:
            return {}

        rewards = [m.reward for m in metrics]
        qualities = [m.quality for m in metrics if m.quality > 0]
        successes = [m.success for m in metrics]

        return {
            'total_episodes': len(self.training_metrics),
            'recent_episodes': len(metrics),
            'average_reward': np.mean(rewards) if rewards else 0.0,
            'reward_std': np.std(rewards) if rewards else 0.0,
            'average_quality': np.mean(qualities) if qualities else 0.0,
            'success_rate': np.mean(successes) if successes else 0.0,
            'latest_reward': metrics[-1].reward if metrics else 0.0,
            'latest_quality': metrics[-1].quality if metrics else 0.0,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0.0
        }

    def create_callback(self) -> RealTimeTrainingCallback:
        """Create training callback for PPO integration"""
        return RealTimeTrainingCallback(self)

    def on_training_start(self):
        """Called when training starts"""
        self.training_active = True
        self.training_start_time = time.time()

        # Broadcast training start
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.websocket_server.broadcast({
                    'type': 'training_event',
                    'event': 'training_started',
                    'timestamp': self.training_start_time
                }),
                self.loop
            )

        logger.info("Training started - real-time monitoring active")

    def on_training_end(self):
        """Called when training ends"""
        self.training_active = False

        # Broadcast training end
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.websocket_server.broadcast({
                    'type': 'training_event',
                    'event': 'training_ended',
                    'timestamp': time.time(),
                    'total_episodes': len(self.training_metrics),
                    'training_duration': time.time() - self.training_start_time if self.training_start_time else 0
                }),
                self.loop
            )

        logger.info("Training ended - monitoring continues")

    def on_episode_complete(self, metrics: TrainingMetrics):
        """Called when an episode completes"""
        self.training_metrics.append(metrics)
        self.current_metrics = metrics

        # Broadcast episode metrics
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.websocket_server.broadcast({
                    'type': 'episode_metrics',
                    'metrics': asdict(metrics)
                }),
                self.loop
            )

        # Log episode
        logger.info(
            f"Episode {metrics.episode}: "
            f"Reward={metrics.reward:.3f}, "
            f"Quality={metrics.quality:.3f}, "
            f"Success={metrics.success}"
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        recent_training = list(self.training_metrics)[-50:] if self.training_metrics else []
        recent_system = list(self.system_metrics)[-50:] if self.system_metrics else []
        recent_alerts = self.alert_system.get_recent_alerts(60)

        return {
            'training_metrics': [asdict(m) for m in recent_training],
            'system_metrics': [asdict(m) for m in recent_system],
            'alerts': [asdict(a) for a in recent_alerts],
            'alert_summary': self.alert_system.get_alert_summary(),
            'training_active': self.training_active,
            'total_episodes': len(self.training_metrics),
            'training_summary': self._create_training_summary(recent_training) if recent_training else {},
            'websocket_clients': len(self.websocket_server.clients)
        }

    def save_monitoring_data(self):
        """Save monitoring data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save training metrics
        training_file = self.save_dir / f"training_metrics_{timestamp}.json"
        with open(training_file, 'w') as f:
            json.dump([asdict(m) for m in self.training_metrics], f, indent=2)

        # Save system metrics
        system_file = self.save_dir / f"system_metrics_{timestamp}.json"
        with open(system_file, 'w') as f:
            json.dump([asdict(m) for m in self.system_metrics], f, indent=2)

        # Save alerts
        alerts_file = self.save_dir / f"alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump([asdict(a) for a in self.alert_system.alerts], f, indent=2)

        logger.info(f"Monitoring data saved to {self.save_dir}")

    def generate_monitoring_report(self) -> str:
        """Generate monitoring report"""
        if not self.training_metrics:
            return "No training data available"

        report = []
        report.append("# Real-Time Monitoring Report")
        report.append("=" * 50)
        report.append("")

        # Training summary
        all_metrics = list(self.training_metrics)
        rewards = [m.reward for m in all_metrics]
        qualities = [m.quality for m in all_metrics if m.quality > 0]
        successes = [m.success for m in all_metrics]

        report.append("## Training Performance")
        report.append(f"- Total Episodes: {len(all_metrics)}")
        report.append(f"- Average Reward: {np.mean(rewards):.4f}")
        report.append(f"- Best Reward: {max(rewards):.4f}")
        report.append(f"- Average Quality: {np.mean(qualities):.4f}" if qualities else "- Average Quality: N/A")
        report.append(f"- Success Rate: {np.mean(successes):.2%}")

        if self.training_start_time:
            training_duration = time.time() - self.training_start_time
            report.append(f"- Training Duration: {training_duration:.2f}s")
            report.append(f"- Episodes per Minute: {len(all_metrics) / (training_duration / 60):.1f}")

        report.append("")

        # System performance
        if self.system_metrics:
            recent_system = list(self.system_metrics)[-10:]
            avg_cpu = np.mean([m.cpu_percent for m in recent_system])
            avg_memory = np.mean([m.memory_percent for m in recent_system])

            report.append("## System Performance")
            report.append(f"- Average CPU Usage: {avg_cpu:.1f}%")
            report.append(f"- Average Memory Usage: {avg_memory:.1f}%")
            report.append("")

        # Alerts summary
        alert_summary = self.alert_system.get_alert_summary()
        if alert_summary:
            report.append("## Alerts Summary")
            for alert_type, count in alert_summary.items():
                report.append(f"- {alert_type}: {count}")
            report.append("")

        return "\n".join(report)


# Factory function for easy initialization
def create_real_time_monitor(websocket_port: int = 8765,
                           update_interval: float = 1.0,
                           save_dir: str = "logs/real_time_monitoring") -> RealTimeMonitor:
    """
    Factory function to create real-time monitor

    Args:
        websocket_port: Port for WebSocket server
        update_interval: System monitoring update interval
        save_dir: Directory to save monitoring data

    Returns:
        Configured real-time monitor
    """
    return RealTimeMonitor(
        websocket_port=websocket_port,
        update_interval=update_interval,
        save_dir=save_dir
    )


# Integration example
async def main():
    """Example usage of real-time monitor"""
    monitor = create_real_time_monitor()

    try:
        # Start monitoring
        await monitor.start_monitoring()

        # Simulate training for demo
        await asyncio.sleep(60)

    finally:
        # Stop monitoring
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())