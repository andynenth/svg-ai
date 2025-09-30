#!/usr/bin/env python3
"""
Performance Tuning Utilities
Automated performance optimization for the SVG-AI system
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import yaml
import psutil
import numpy as np
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceTuning:
    """Performance tuning recommendation"""
    component: str
    parameter: str
    current_value: Any
    recommended_value: Any
    improvement_estimate: str
    confidence: float
    justification: str
    implementation_complexity: str

@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    component: str
    optimization_type: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: float
    success: bool
    timestamp: datetime

class SystemProfiler:
    """Profiles system performance to identify optimization opportunities"""

    def __init__(self):
        self.metrics = {}

    def profile_system_resources(self) -> Dict[str, Any]:
        """Profile current system resource utilization"""
        profile = {
            'cpu': self._profile_cpu(),
            'memory': self._profile_memory(),
            'disk': self._profile_disk(),
            'network': self._profile_network(),
            'processes': self._profile_processes()
        }

        logger.info("System resource profiling completed")
        return profile

    def _profile_cpu(self) -> Dict[str, float]:
        """Profile CPU utilization and characteristics"""
        cpu_info = {
            'usage_percent': psutil.cpu_percent(interval=1),
            'core_count': psutil.cpu_count(logical=False),
            'thread_count': psutil.cpu_count(logical=True),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'load_1min': os.getloadavg()[0],
            'load_5min': os.getloadavg()[1],
            'load_15min': os.getloadavg()[2]
        }

        # Per-core utilization
        cpu_info['per_core_usage'] = psutil.cpu_percent(interval=1, percpu=True)
        cpu_info['max_core_usage'] = max(cpu_info['per_core_usage'])
        cpu_info['min_core_usage'] = min(cpu_info['per_core_usage'])
        cpu_info['core_usage_std'] = statistics.stdev(cpu_info['per_core_usage'])

        return cpu_info

    def _profile_memory(self) -> Dict[str, float]:
        """Profile memory utilization"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_usage_percent': swap.percent,
            'cached_gb': memory.cached / (1024**3),
            'buffers_gb': memory.buffers / (1024**3)
        }

    def _profile_disk(self) -> Dict[str, Any]:
        """Profile disk utilization and I/O"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        disk_info = {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'usage_percent': (disk_usage.used / disk_usage.total) * 100
        }

        if disk_io:
            disk_info.update({
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_time_ms': disk_io.read_time,
                'write_time_ms': disk_io.write_time
            })

        return disk_info

    def _profile_network(self) -> Dict[str, float]:
        """Profile network utilization"""
        net_io = psutil.net_io_counters()

        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors_in': net_io.errin,
            'errors_out': net_io.errout,
            'drops_in': net_io.dropin,
            'drops_out': net_io.dropout
        }

    def _profile_processes(self) -> List[Dict[str, Any]]:
        """Profile running processes"""
        processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 1.0 or proc_info['memory_percent'] > 1.0:
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent'],
                        'memory_percent': proc_info['memory_percent'],
                        'memory_mb': proc_info['memory_info'].rss / (1024**2) if proc_info['memory_info'] else 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by resource usage
        return sorted(processes, key=lambda x: x['cpu_percent'] + x['memory_percent'], reverse=True)[:20]

class DatabaseTuner:
    """Tunes database performance parameters"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config

    def analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database performance metrics"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'svgai_prod'),
                user=self.db_config.get('user', 'svgai_user'),
                password=self.db_config.get('password', '')
            )

            cursor = conn.cursor()
            analysis = {}

            # Connection statistics
            cursor.execute("""
                SELECT count(*) as total_connections,
                       count(*) FILTER (WHERE state = 'active') as active_connections,
                       count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity;
            """)
            connection_stats = cursor.fetchone()
            analysis['connections'] = {
                'total': connection_stats[0],
                'active': connection_stats[1],
                'idle': connection_stats[2]
            }

            # Database size and statistics
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(%s)) as db_size,
                       (SELECT count(*) FROM pg_stat_user_tables) as table_count;
            """, (self.db_config.get('database', 'svgai_prod'),))
            db_stats = cursor.fetchone()
            analysis['database'] = {
                'size': db_stats[0],
                'table_count': db_stats[1]
            }

            # Query performance
            cursor.execute("""
                SELECT query, calls, total_time, mean_time, rows
                FROM pg_stat_statements
                WHERE calls > 10
                ORDER BY mean_time DESC
                LIMIT 10;
            """)
            slow_queries = cursor.fetchall()
            analysis['slow_queries'] = [
                {
                    'query': row[0][:100] + '...' if len(row[0]) > 100 else row[0],
                    'calls': row[1],
                    'total_time_ms': row[2],
                    'mean_time_ms': row[3],
                    'rows': row[4]
                }
                for row in slow_queries
            ]

            # Index usage
            cursor.execute("""
                SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE idx_tup_read > 0
                ORDER BY idx_tup_read DESC
                LIMIT 10;
            """)
            index_usage = cursor.fetchall()
            analysis['index_usage'] = [
                {
                    'table': f"{row[0]}.{row[1]}",
                    'index': row[2],
                    'reads': row[3],
                    'fetches': row[4]
                }
                for row in index_usage
            ]

            conn.close()
            logger.info("Database performance analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return {}

    def get_database_tuning_recommendations(self, analysis: Dict[str, Any],
                                          system_profile: Dict[str, Any]) -> List[PerformanceTuning]:
        """Generate database tuning recommendations"""
        recommendations = []

        # Memory tuning
        total_memory_gb = system_profile['memory']['total_gb']
        shared_buffers_rec = min(total_memory_gb * 0.25, 8)  # 25% of RAM, max 8GB

        recommendations.append(PerformanceTuning(
            component='postgresql',
            parameter='shared_buffers',
            current_value='128MB',  # Default value
            recommended_value=f'{int(shared_buffers_rec * 1024)}MB',
            improvement_estimate='10-30% query performance',
            confidence=0.9,
            justification=f'Set to 25% of available RAM ({total_memory_gb:.1f}GB)',
            implementation_complexity='low'
        ))

        # Effective cache size
        effective_cache_rec = total_memory_gb * 0.75
        recommendations.append(PerformanceTuning(
            component='postgresql',
            parameter='effective_cache_size',
            current_value='4GB',  # Default value
            recommended_value=f'{int(effective_cache_rec)}GB',
            improvement_estimate='5-15% query planning',
            confidence=0.8,
            justification=f'Set to 75% of available RAM for query planner optimization',
            implementation_complexity='low'
        ))

        # Connection tuning
        if analysis.get('connections', {}).get('total', 0) > 100:
            recommendations.append(PerformanceTuning(
                component='postgresql',
                parameter='max_connections',
                current_value='100',
                recommended_value='200',
                improvement_estimate='Reduced connection errors',
                confidence=0.7,
                justification='High connection count detected',
                implementation_complexity='medium'
            ))

        # Work memory tuning
        work_mem_rec = max(4, min(total_memory_gb * 1024 / 200, 256))  # 4MB to 256MB
        recommendations.append(PerformanceTuning(
            component='postgresql',
            parameter='work_mem',
            current_value='4MB',
            recommended_value=f'{int(work_mem_rec)}MB',
            improvement_estimate='10-25% sort/hash operations',
            confidence=0.8,
            justification='Optimized for available memory and connection count',
            implementation_complexity='low'
        ))

        return recommendations

class ApplicationTuner:
    """Tunes application-specific performance parameters"""

    def __init__(self, project_root: str):
        self.project_root = project_root

    def analyze_application_performance(self) -> Dict[str, Any]:
        """Analyze application performance metrics"""
        analysis = {
            'python_processes': self._analyze_python_processes(),
            'web_server_config': self._analyze_web_server_config(),
            'optimization_workers': self._analyze_optimization_workers(),
            'cache_utilization': self._analyze_cache_utilization()
        }

        logger.info("Application performance analysis completed")
        return analysis

    def _analyze_python_processes(self) -> Dict[str, Any]:
        """Analyze Python process performance"""
        python_procs = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_procs.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(proc.info['cmdline'][:3]),
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / (1024**2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            'process_count': len(python_procs),
            'total_memory_mb': sum(p['memory_mb'] for p in python_procs),
            'avg_cpu_percent': statistics.mean([p['cpu_percent'] for p in python_procs]) if python_procs else 0,
            'processes': python_procs[:10]  # Top 10
        }

    def _analyze_web_server_config(self) -> Dict[str, Any]:
        """Analyze web server configuration"""
        config = {}

        # Check for gunicorn processes
        gunicorn_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'gunicorn' in proc.info['name'] or any('gunicorn' in arg for arg in proc.info['cmdline']):
                    gunicorn_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        config['gunicorn_workers'] = len(gunicorn_procs)

        # Recommended worker count
        cpu_cores = psutil.cpu_count(logical=False)
        config['recommended_workers'] = (cpu_cores * 2) + 1

        return config

    def _analyze_optimization_workers(self) -> Dict[str, Any]:
        """Analyze optimization worker performance"""
        # This would integrate with your specific worker queue system
        return {
            'active_workers': 2,  # Placeholder
            'queue_length': 0,    # Placeholder
            'avg_processing_time': 15.5  # Placeholder
        }

    def _analyze_cache_utilization(self) -> Dict[str, Any]:
        """Analyze cache performance"""
        # This would integrate with Redis or other caching systems
        return {
            'cache_hit_rate': 0.85,  # Placeholder
            'cache_size_mb': 128,    # Placeholder
            'eviction_rate': 0.02    # Placeholder
        }

    def get_application_tuning_recommendations(self, analysis: Dict[str, Any],
                                             system_profile: Dict[str, Any]) -> List[PerformanceTuning]:
        """Generate application tuning recommendations"""
        recommendations = []

        # Gunicorn worker tuning
        current_workers = analysis['web_server_config']['gunicorn_workers']
        recommended_workers = analysis['web_server_config']['recommended_workers']

        if current_workers != recommended_workers:
            recommendations.append(PerformanceTuning(
                component='gunicorn',
                parameter='workers',
                current_value=current_workers,
                recommended_value=recommended_workers,
                improvement_estimate='20-40% request throughput',
                confidence=0.9,
                justification=f'Optimal worker count for {psutil.cpu_count(logical=False)} CPU cores',
                implementation_complexity='low'
            ))

        # Memory tuning for workers
        avg_worker_memory = analysis['python_processes']['total_memory_mb'] / max(current_workers, 1)
        if avg_worker_memory > 512:  # High memory usage per worker
            recommendations.append(PerformanceTuning(
                component='application',
                parameter='memory_optimization',
                current_value=f'{avg_worker_memory:.0f}MB per worker',
                recommended_value='<512MB per worker',
                improvement_estimate='Reduced memory pressure',
                confidence=0.7,
                justification='High memory usage detected per worker process',
                implementation_complexity='medium'
            ))

        # Python GC tuning
        recommendations.append(PerformanceTuning(
            component='python',
            parameter='garbage_collection',
            current_value='default',
            recommended_value='tuned',
            improvement_estimate='5-15% latency reduction',
            confidence=0.6,
            justification='Optimize GC for long-running processes',
            implementation_complexity='medium'
        ))

        return recommendations

class KubernetesTuner:
    """Tunes Kubernetes deployment parameters"""

    def __init__(self, namespace: str = 'svg-ai-prod'):
        self.namespace = namespace

    def analyze_kubernetes_performance(self) -> Dict[str, Any]:
        """Analyze Kubernetes cluster performance"""
        try:
            analysis = {
                'pod_performance': self._analyze_pod_performance(),
                'resource_requests': self._analyze_resource_requests(),
                'autoscaling': self._analyze_autoscaling(),
                'node_utilization': self._analyze_node_utilization()
            }

            logger.info("Kubernetes performance analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Kubernetes analysis failed: {e}")
            return {}

    def _analyze_pod_performance(self) -> Dict[str, Any]:
        """Analyze pod performance metrics"""
        try:
            # Get pod metrics using kubectl
            result = subprocess.run(
                ['kubectl', 'top', 'pods', '-n', self.namespace],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return {'error': 'Failed to get pod metrics'}

            pods = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        pods.append({
                            'name': parts[0],
                            'cpu_millicores': parts[1].replace('m', ''),
                            'memory_mi': parts[2].replace('Mi', '')
                        })

            return {'pods': pods}

        except Exception as e:
            logger.error(f"Pod performance analysis failed: {e}")
            return {}

    def _analyze_resource_requests(self) -> Dict[str, Any]:
        """Analyze resource requests and limits"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.namespace, '-o', 'json'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return {'error': 'Failed to get pod specs'}

            pod_data = json.loads(result.stdout)
            resource_analysis = []

            for pod in pod_data.get('items', []):
                pod_name = pod['metadata']['name']

                for container in pod['spec'].get('containers', []):
                    resources = container.get('resources', {})
                    requests = resources.get('requests', {})
                    limits = resources.get('limits', {})

                    resource_analysis.append({
                        'pod': pod_name,
                        'container': container['name'],
                        'cpu_request': requests.get('cpu', 'none'),
                        'memory_request': requests.get('memory', 'none'),
                        'cpu_limit': limits.get('cpu', 'none'),
                        'memory_limit': limits.get('memory', 'none')
                    })

            return {'resources': resource_analysis}

        except Exception as e:
            logger.error(f"Resource analysis failed: {e}")
            return {}

    def _analyze_autoscaling(self) -> Dict[str, Any]:
        """Analyze HPA configuration"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'hpa', '-n', self.namespace, '-o', 'json'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return {'hpa_configured': False}

            hpa_data = json.loads(result.stdout)
            hpa_configs = []

            for hpa in hpa_data.get('items', []):
                hpa_configs.append({
                    'name': hpa['metadata']['name'],
                    'min_replicas': hpa['spec'].get('minReplicas', 1),
                    'max_replicas': hpa['spec'].get('maxReplicas', 10),
                    'target_cpu': hpa['spec'].get('targetCPUUtilizationPercentage', 50)
                })

            return {'hpa_configured': True, 'configurations': hpa_configs}

        except Exception as e:
            logger.error(f"HPA analysis failed: {e}")
            return {'hpa_configured': False}

    def _analyze_node_utilization(self) -> Dict[str, Any]:
        """Analyze node resource utilization"""
        try:
            result = subprocess.run(
                ['kubectl', 'top', 'nodes'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return {'error': 'Failed to get node metrics'}

            nodes = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        nodes.append({
                            'name': parts[0],
                            'cpu_cores_used': parts[1].replace('m', ''),
                            'cpu_percent': parts[2],
                            'memory_used': parts[3],
                            'memory_percent': parts[4]
                        })

            return {'nodes': nodes}

        except Exception as e:
            logger.error(f"Node utilization analysis failed: {e}")
            return {}

    def get_kubernetes_tuning_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceTuning]:
        """Generate Kubernetes tuning recommendations"""
        recommendations = []

        # Resource request recommendations
        resource_analysis = analysis.get('resource_requests', {}).get('resources', [])

        for resource in resource_analysis:
            if resource['cpu_request'] == 'none':
                recommendations.append(PerformanceTuning(
                    component='kubernetes',
                    parameter='cpu_requests',
                    current_value='none',
                    recommended_value='100m-500m',
                    improvement_estimate='Better scheduling and resource management',
                    confidence=0.8,
                    justification=f'Pod {resource["pod"]} missing CPU requests',
                    implementation_complexity='low'
                ))

            if resource['memory_request'] == 'none':
                recommendations.append(PerformanceTuning(
                    component='kubernetes',
                    parameter='memory_requests',
                    current_value='none',
                    recommended_value='256Mi-1Gi',
                    improvement_estimate='Better scheduling and resource management',
                    confidence=0.8,
                    justification=f'Pod {resource["pod"]} missing memory requests',
                    implementation_complexity='low'
                ))

        # HPA recommendations
        hpa_analysis = analysis.get('autoscaling', {})
        if not hpa_analysis.get('hpa_configured', False):
            recommendations.append(PerformanceTuning(
                component='kubernetes',
                parameter='horizontal_pod_autoscaler',
                current_value='not_configured',
                recommended_value='configured',
                improvement_estimate='Automatic scaling based on load',
                confidence=0.9,
                justification='HPA not configured for production workloads',
                implementation_complexity='medium'
            ))

        return recommendations

class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(self, project_root: str, config: Dict[str, Any]):
        self.project_root = project_root
        self.config = config
        self.profiler = SystemProfiler()
        self.db_tuner = DatabaseTuner(config.get('database', {}))
        self.app_tuner = ApplicationTuner(project_root)
        self.k8s_tuner = KubernetesTuner(config.get('namespace', 'svg-ai-prod'))

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        logger.info("Starting complete performance analysis")

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'system_profile': self.profiler.profile_system_resources(),
            'database_analysis': self.db_tuner.analyze_database_performance(),
            'application_analysis': self.app_tuner.analyze_application_performance(),
            'kubernetes_analysis': self.k8s_tuner.analyze_kubernetes_performance()
        }

        logger.info("Complete performance analysis finished")
        return analysis

    def generate_tuning_recommendations(self, analysis: Dict[str, Any]) -> List[PerformanceTuning]:
        """Generate comprehensive tuning recommendations"""
        recommendations = []

        # Database recommendations
        db_recommendations = self.db_tuner.get_database_tuning_recommendations(
            analysis['database_analysis'], analysis['system_profile']
        )
        recommendations.extend(db_recommendations)

        # Application recommendations
        app_recommendations = self.app_tuner.get_application_tuning_recommendations(
            analysis['application_analysis'], analysis['system_profile']
        )
        recommendations.extend(app_recommendations)

        # Kubernetes recommendations
        k8s_recommendations = self.k8s_tuner.get_kubernetes_tuning_recommendations(
            analysis['kubernetes_analysis']
        )
        recommendations.extend(k8s_recommendations)

        # Sort by confidence and impact
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"Generated {len(recommendations)} tuning recommendations")
        return recommendations

    def apply_tuning_recommendation(self, recommendation: PerformanceTuning, dry_run: bool = True) -> OptimizationResult:
        """Apply a specific tuning recommendation"""
        logger.info(f"Applying tuning: {recommendation.component}.{recommendation.parameter}")

        # Capture before metrics
        before_metrics = self._capture_metrics()

        success = False
        if not dry_run:
            success = self._apply_tuning(recommendation)
        else:
            logger.info("Dry run - tuning not actually applied")
            success = True

        # Capture after metrics (in real scenario, would wait for metrics to stabilize)
        after_metrics = before_metrics if dry_run else self._capture_metrics()

        # Calculate improvement
        improvement = self._calculate_improvement(before_metrics, after_metrics)

        return OptimizationResult(
            component=recommendation.component,
            optimization_type=recommendation.parameter,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement,
            success=success,
            timestamp=datetime.now()
        )

    def _capture_metrics(self) -> Dict[str, float]:
        """Capture current performance metrics"""
        try:
            system_profile = self.profiler.profile_system_resources()
            return {
                'cpu_usage': system_profile['cpu']['usage_percent'],
                'memory_usage': system_profile['memory']['usage_percent'],
                'load_1min': system_profile['cpu']['load_1min'],
                'disk_usage': system_profile['disk']['usage_percent']
            }
        except Exception as e:
            logger.error(f"Failed to capture metrics: {e}")
            return {}

    def _apply_tuning(self, recommendation: PerformanceTuning) -> bool:
        """Apply the tuning recommendation"""
        try:
            if recommendation.component == 'postgresql':
                return self._apply_postgresql_tuning(recommendation)
            elif recommendation.component == 'gunicorn':
                return self._apply_gunicorn_tuning(recommendation)
            elif recommendation.component == 'kubernetes':
                return self._apply_kubernetes_tuning(recommendation)
            else:
                logger.warning(f"Unknown component for tuning: {recommendation.component}")
                return False

        except Exception as e:
            logger.error(f"Failed to apply tuning: {e}")
            return False

    def _apply_postgresql_tuning(self, recommendation: PerformanceTuning) -> bool:
        """Apply PostgreSQL tuning"""
        # This would modify postgresql.conf and restart the service
        logger.info(f"Would apply PostgreSQL tuning: {recommendation.parameter} = {recommendation.recommended_value}")
        return True

    def _apply_gunicorn_tuning(self, recommendation: PerformanceTuning) -> bool:
        """Apply Gunicorn tuning"""
        # This would modify gunicorn configuration and restart workers
        logger.info(f"Would apply Gunicorn tuning: {recommendation.parameter} = {recommendation.recommended_value}")
        return True

    def _apply_kubernetes_tuning(self, recommendation: PerformanceTuning) -> bool:
        """Apply Kubernetes tuning"""
        # This would modify K8s manifests and apply changes
        logger.info(f"Would apply Kubernetes tuning: {recommendation.parameter} = {recommendation.recommended_value}")
        return True

    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate overall performance improvement percentage"""
        if not before or not after:
            return 0.0

        improvements = []
        for metric in ['cpu_usage', 'memory_usage', 'load_1min']:
            if metric in before and metric in after:
                if before[metric] > 0:
                    improvement = ((before[metric] - after[metric]) / before[metric]) * 100
                    improvements.append(improvement)

        return statistics.mean(improvements) if improvements else 0.0

    def generate_performance_report(self, analysis: Dict[str, Any],
                                  recommendations: List[PerformanceTuning],
                                  output_file: str) -> str:
        """Generate comprehensive performance report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_summary': {
                'cpu_cores': analysis['system_profile']['cpu']['core_count'],
                'memory_gb': analysis['system_profile']['memory']['total_gb'],
                'disk_gb': analysis['system_profile']['disk']['total_gb'],
                'current_cpu_usage': analysis['system_profile']['cpu']['usage_percent'],
                'current_memory_usage': analysis['system_profile']['memory']['usage_percent']
            },
            'performance_analysis': analysis,
            'recommendations': [asdict(rec) for rec in recommendations],
            'priority_recommendations': [
                asdict(rec) for rec in recommendations
                if rec.confidence > 0.8 and 'high' in rec.improvement_estimate.lower()
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report generated: {output_file}")
        return output_file

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Performance Tuning Utilities")
    parser.add_argument('--config', default='config/performance_tuning.json',
                        help='Configuration file path')
    parser.add_argument('--output-dir', default='performance_reports',
                        help='Output directory for reports')
    parser.add_argument('--mode', choices=['analyze', 'tune', 'report'],
                        default='analyze', help='Operation mode')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate optimizations without applying')
    parser.add_argument('--component', choices=['database', 'application', 'kubernetes', 'all'],
                        default='all', help='Component to tune')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Initialize optimizer
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        optimizer = PerformanceOptimizer(project_root, config)

        if args.mode == 'analyze':
            # Run performance analysis
            logger.info("Running performance analysis...")
            analysis = optimizer.run_complete_analysis()

            # Generate recommendations
            recommendations = optimizer.generate_tuning_recommendations(analysis)

            # Print summary
            print(f"\nPerformance Analysis Summary:")
            print(f"CPU Usage: {analysis['system_profile']['cpu']['usage_percent']:.1f}%")
            print(f"Memory Usage: {analysis['system_profile']['memory']['usage_percent']:.1f}%")
            print(f"Generated {len(recommendations)} recommendations")

            # Print top recommendations
            print("\nTop Recommendations:")
            for rec in recommendations[:5]:
                print(f"  {rec.component}.{rec.parameter}: {rec.current_value} → {rec.recommended_value}")
                print(f"    Impact: {rec.improvement_estimate}")
                print(f"    Confidence: {rec.confidence:.1%}")
                print()

        elif args.mode == 'tune':
            # Apply tuning recommendations
            logger.info("Applying performance tuning...")
            analysis = optimizer.run_complete_analysis()
            recommendations = optimizer.generate_tuning_recommendations(analysis)

            # Apply high-confidence recommendations
            results = []
            for rec in recommendations:
                if rec.confidence > 0.8:
                    result = optimizer.apply_tuning_recommendation(rec, args.dry_run)
                    results.append(result)

            # Print results
            print(f"\nTuning Results:")
            for result in results:
                status = "SUCCESS" if result.success else "FAILED"
                print(f"  {result.component}.{result.optimization_type}: {status}")
                if result.improvement_percentage != 0:
                    print(f"    Improvement: {result.improvement_percentage:.1f}%")

        elif args.mode == 'report':
            # Generate comprehensive report
            logger.info("Generating performance report...")
            analysis = optimizer.run_complete_analysis()
            recommendations = optimizer.generate_tuning_recommendations(analysis)

            report_file = os.path.join(
                args.output_dir,
                f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            optimizer.generate_performance_report(analysis, recommendations, report_file)
            print(f"Performance report generated: {report_file}")

    except Exception as e:
        logger.error(f"Error in performance tuning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()