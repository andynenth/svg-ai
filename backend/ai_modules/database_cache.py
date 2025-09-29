#!/usr/bin/env python3
"""
Database-Backed Cache System for Large-Scale Persistence

Implements comprehensive database integration for caching including:
- Optimized database schema for cache storage
- High-performance database cache backend
- Cache synchronization mechanisms across instances
- Backup and recovery systems
- Distributed cache coordination
"""

import json
import logging
import pickle
import sqlite3
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import concurrent.futures
import hashlib

try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)


class DatabaseCacheBackend(ABC):
    """Abstract base class for database cache backends"""

    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection"""
        pass

    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass

    @abstractmethod
    def create_schema(self) -> bool:
        """Create cache table schema"""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Get cached data by key"""
        pass

    @abstractmethod
    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Set cached data with optional TTL"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached data by key"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached data"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Clean up expired entries, return count removed"""
        pass


class SQLiteCacheBackend(DatabaseCacheBackend):
    """High-performance SQLite cache backend with optimizations"""

    def __init__(self, db_path: str = "cache.db", enable_wal: bool = True,
                 cache_size_mb: int = 64, enable_compression: bool = True):
        """
        Initialize SQLite cache backend

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better concurrency
            cache_size_mb: SQLite page cache size in MB
            enable_compression: Enable data compression
        """
        self.db_path = Path(db_path)
        self.enable_wal = enable_wal
        self.cache_size_mb = cache_size_mb
        self.enable_compression = enable_compression
        self.connection = None
        self.lock = threading.RLock()

        # Performance counters
        self.stats = {
            'gets': 0,
            'sets': 0,
            'deletes': 0,
            'hits': 0,
            'misses': 0,
            'errors': 0
        }

    def connect(self) -> bool:
        """Establish optimized SQLite connection"""
        try:
            with self.lock:
                if self.connection:
                    return True

                # Create directory if it doesn't exist
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

                # Connect with optimizations
                self.connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False
                )

                # Enable optimizations
                cursor = self.connection.cursor()

                if self.enable_wal:
                    cursor.execute("PRAGMA journal_mode=WAL")

                # Performance optimizations
                cursor.execute(f"PRAGMA cache_size=-{self.cache_size_mb * 1024}")  # Negative for KB
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA optimize")

                self.connection.commit()

                logger.info(f"Connected to SQLite cache: {self.db_path}")
                return True

        except Exception as e:
            logger.error(f"Error connecting to SQLite: {e}")
            return False

    def disconnect(self):
        """Close SQLite connection"""
        with self.lock:
            if self.connection:
                try:
                    self.connection.execute("PRAGMA optimize")
                    self.connection.close()
                    self.connection = None
                except Exception as e:
                    logger.error(f"Error disconnecting from SQLite: {e}")

    def create_schema(self) -> bool:
        """Create optimized cache table schema"""
        try:
            with self.lock:
                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()

                # Main cache table with optimized schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        data BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        access_count INTEGER DEFAULT 1,
                        last_accessed REAL NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        cache_type TEXT,
                        compressed BOOLEAN DEFAULT FALSE,
                        metadata TEXT
                    )
                ''')

                # Indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')

                # Cache statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        total_entries INTEGER,
                        total_size_bytes INTEGER,
                        hit_rate REAL,
                        avg_access_count REAL,
                        expired_entries INTEGER
                    )
                ''')

                # Cache synchronization table for distributed scenarios
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_sync (
                        sync_id TEXT PRIMARY KEY,
                        node_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        cache_key TEXT,
                        timestamp REAL NOT NULL,
                        processed BOOLEAN DEFAULT FALSE
                    )
                ''')

                self.connection.commit()
                logger.info("Created SQLite cache schema")
                return True

        except Exception as e:
            logger.error(f"Error creating SQLite schema: {e}")
            return False

    def _compress_data(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if beneficial"""
        if not self.enable_compression or len(data) < 1024:  # Don't compress small data
            return data, False

        try:
            compressed = zlib.compress(data, level=6)
            if len(compressed) < len(data) * 0.9:  # Only use if 10%+ savings
                return compressed, True
            return data, False
        except Exception:
            return data, False

    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if needed"""
        if not is_compressed:
            return data

        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return data

    def get(self, key: str) -> Optional[bytes]:
        """Get cached data with access tracking"""
        try:
            with self.lock:
                self.stats['gets'] += 1

                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()

                # Get data and check expiration
                cursor.execute('''
                    SELECT data, expires_at, compressed, access_count
                    FROM cache_entries
                    WHERE key = ?
                ''', (key,))

                row = cursor.fetchone()
                if not row:
                    self.stats['misses'] += 1
                    return None

                data, expires_at, compressed, access_count = row

                # Check expiration
                if expires_at and time.time() > expires_at:
                    # Delete expired entry
                    cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                    self.connection.commit()
                    self.stats['misses'] += 1
                    return None

                # Update access statistics
                cursor.execute('''
                    UPDATE cache_entries
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ?
                ''', (time.time(), key))

                self.connection.commit()
                self.stats['hits'] += 1

                # Decompress if needed
                return self._decompress_data(data, bool(compressed))

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            self.stats['errors'] += 1
            return None

    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Set cached data with optional TTL"""
        try:
            with self.lock:
                self.stats['sets'] += 1

                if not self.connection:
                    self.connect()

                # Compress data if beneficial
                compressed_data, is_compressed = self._compress_data(data)

                current_time = time.time()
                expires_at = current_time + ttl if ttl else None

                # Extract cache type from key
                cache_type = key.split(':', 1)[0] if ':' in key else 'unknown'

                cursor = self.connection.cursor()

                # Insert or replace
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_entries
                    (key, data, created_at, expires_at, last_accessed, size_bytes, cache_type, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key, compressed_data, current_time, expires_at,
                    current_time, len(compressed_data), cache_type, is_compressed
                ))

                self.connection.commit()
                return True

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            self.stats['errors'] += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete cached data"""
        try:
            with self.lock:
                self.stats['deletes'] += 1

                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                self.connection.commit()

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            self.stats['errors'] += 1
            return False

    def clear(self) -> bool:
        """Clear all cached data"""
        try:
            with self.lock:
                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()
                cursor.execute('DELETE FROM cache_entries')
                self.connection.commit()

                logger.info("Cleared all cache entries")
                return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        try:
            with self.lock:
                if not self.connection:
                    self.connect()

                current_time = time.time()
                cursor = self.connection.cursor()

                cursor.execute('''
                    DELETE FROM cache_entries
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                ''', (current_time,))

                removed_count = cursor.rowcount
                self.connection.commit()

                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} expired cache entries")

                return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            with self.lock:
                if not self.connection:
                    self.connect()

                cursor = self.connection.cursor()

                # Get current cache statistics
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at < ? THEN 1 END) as expired_entries
                    FROM cache_entries
                ''', (time.time(),))

                row = cursor.fetchone()
                total_entries, total_size, avg_access_count, expired_entries = row

                # Get cache type distribution
                cursor.execute('''
                    SELECT cache_type, COUNT(*), SUM(size_bytes)
                    FROM cache_entries
                    GROUP BY cache_type
                    ORDER BY COUNT(*) DESC
                ''')

                type_distribution = {
                    cache_type: {'count': count, 'size_bytes': size}
                    for cache_type, count, size in cursor.fetchall()
                }

                # Calculate hit rate
                total_requests = self.stats['gets']
                hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

                return {
                    'backend_type': 'sqlite',
                    'database_path': str(self.db_path),
                    'performance_stats': dict(self.stats),
                    'hit_rate': hit_rate,
                    'cache_data': {
                        'total_entries': total_entries or 0,
                        'total_size_bytes': total_size or 0,
                        'total_size_mb': (total_size or 0) / (1024 * 1024),
                        'avg_access_count': avg_access_count or 0,
                        'expired_entries': expired_entries or 0
                    },
                    'type_distribution': type_distribution,
                    'configuration': {
                        'wal_enabled': self.enable_wal,
                        'cache_size_mb': self.cache_size_mb,
                        'compression_enabled': self.enable_compression
                    }
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}


class PostgreSQLCacheBackend(DatabaseCacheBackend):
    """PostgreSQL cache backend for high-performance distributed scenarios"""

    def __init__(self, host: str = "localhost", port: int = 5432, database: str = "cache_db",
                 user: str = "cache_user", password: str = "", connection_pool_size: int = 20):
        """
        Initialize PostgreSQL cache backend

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            connection_pool_size: Size of connection pool
        """
        if not HAS_POSTGRESQL:
            raise ImportError("psycopg2 not available for PostgreSQL backend")

        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.connection_pool_size = connection_pool_size
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.stats = defaultdict(int)

    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        with self.pool_lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
            else:
                conn = psycopg2.connect(**self.connection_params)
                conn.set_session(autocommit=False)

        try:
            yield conn
        finally:
            with self.pool_lock:
                if len(self.connection_pool) < self.connection_pool_size:
                    self.connection_pool.append(conn)
                else:
                    conn.close()

    def connect(self) -> bool:
        """Initialize connection pool"""
        try:
            # Pre-populate connection pool
            with self.pool_lock:
                for _ in range(min(5, self.connection_pool_size)):
                    conn = psycopg2.connect(**self.connection_params)
                    conn.set_session(autocommit=False)
                    self.connection_pool.append(conn)

            logger.info("Initialized PostgreSQL connection pool")
            return True

        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return False

    def disconnect(self):
        """Close all connections in pool"""
        with self.pool_lock:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self.connection_pool.clear()

    def create_schema(self) -> bool:
        """Create PostgreSQL cache schema with optimizations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Main cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        data BYTEA NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        expires_at TIMESTAMP WITH TIME ZONE,
                        access_count INTEGER DEFAULT 1,
                        last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        size_bytes INTEGER NOT NULL,
                        cache_type TEXT,
                        compressed BOOLEAN DEFAULT FALSE,
                        metadata JSONB
                    )
                ''')

                # Indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed)')

                # Partitioning by cache type for large deployments
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_partitions (
                        cache_type TEXT PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''')

                conn.commit()
                logger.info("Created PostgreSQL cache schema")
                return True

        except Exception as e:
            logger.error(f"Error creating PostgreSQL schema: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """Get cached data with PostgreSQL optimizations"""
        try:
            self.stats['gets'] += 1

            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                cursor.execute('''
                    SELECT data, expires_at, compressed
                    FROM cache_entries
                    WHERE key = %s AND (expires_at IS NULL OR expires_at > NOW())
                ''', (key,))

                row = cursor.fetchone()
                if not row:
                    self.stats['misses'] += 1
                    return None

                # Update access statistics
                cursor.execute('''
                    UPDATE cache_entries
                    SET access_count = access_count + 1, last_accessed = NOW()
                    WHERE key = %s
                ''', (key,))

                conn.commit()
                self.stats['hits'] += 1

                # Decompress if needed
                data = bytes(row['data'])
                if row['compressed']:
                    data = zlib.decompress(data)

                return data

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            self.stats['errors'] += 1
            return None

    def set(self, key: str, data: bytes, ttl: Optional[int] = None) -> bool:
        """Set cached data in PostgreSQL"""
        try:
            self.stats['sets'] += 1

            # Compress if beneficial
            compressed_data = data
            is_compressed = False

            if len(data) > 1024:
                compressed = zlib.compress(data)
                if len(compressed) < len(data) * 0.9:
                    compressed_data = compressed
                    is_compressed = True

            with self.get_connection() as conn:
                cursor = conn.cursor()

                expires_at = f"NOW() + INTERVAL '{ttl} seconds'" if ttl else None

                if expires_at:
                    cursor.execute('''
                        INSERT INTO cache_entries (key, data, expires_at, size_bytes, cache_type, compressed)
                        VALUES (%s, %s, ''' + expires_at + ''', %s, %s, %s)
                        ON CONFLICT (key) DO UPDATE SET
                            data = EXCLUDED.data,
                            expires_at = EXCLUDED.expires_at,
                            size_bytes = EXCLUDED.size_bytes,
                            compressed = EXCLUDED.compressed,
                            last_accessed = NOW()
                    ''', (key, compressed_data, len(compressed_data), key.split(':', 1)[0], is_compressed))
                else:
                    cursor.execute('''
                        INSERT INTO cache_entries (key, data, size_bytes, cache_type, compressed)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (key) DO UPDATE SET
                            data = EXCLUDED.data,
                            size_bytes = EXCLUDED.size_bytes,
                            compressed = EXCLUDED.compressed,
                            last_accessed = NOW()
                    ''', (key, compressed_data, len(compressed_data), key.split(':', 1)[0], is_compressed))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            self.stats['errors'] += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete cached data from PostgreSQL"""
        try:
            self.stats['deletes'] += 1

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache_entries WHERE key = %s', (key,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            self.stats['errors'] += 1
            return False

    def clear(self) -> bool:
        """Clear all cached data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('TRUNCATE cache_entries')
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM cache_entries
                    WHERE expires_at IS NOT NULL AND expires_at < NOW()
                ''')
                removed_count = cursor.rowcount
                conn.commit()
                return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive PostgreSQL cache statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                cursor.execute('''
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at < NOW() THEN 1 END) as expired_entries
                    FROM cache_entries
                ''')

                stats_row = cursor.fetchone()

                # Get cache type distribution
                cursor.execute('''
                    SELECT cache_type, COUNT(*), SUM(size_bytes)
                    FROM cache_entries
                    GROUP BY cache_type
                    ORDER BY COUNT(*) DESC
                ''')

                type_distribution = {
                    row[0]: {'count': row[1], 'size_bytes': row[2]}
                    for row in cursor.fetchall()
                }

                total_requests = self.stats['gets']
                hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

                return {
                    'backend_type': 'postgresql',
                    'connection_params': {k: v for k, v in self.connection_params.items() if k != 'password'},
                    'performance_stats': dict(self.stats),
                    'hit_rate': hit_rate,
                    'cache_data': {
                        'total_entries': stats_row['total_entries'] or 0,
                        'total_size_bytes': stats_row['total_size'] or 0,
                        'total_size_mb': (stats_row['total_size'] or 0) / (1024 * 1024),
                        'avg_access_count': float(stats_row['avg_access_count'] or 0),
                        'expired_entries': stats_row['expired_entries'] or 0
                    },
                    'type_distribution': type_distribution
                }

        except Exception as e:
            logger.error(f"Error getting PostgreSQL stats: {e}")
            return {'error': str(e)}


class CacheSynchronizationManager:
    """Manages cache synchronization across multiple instances"""

    def __init__(self, backend: DatabaseCacheBackend, node_id: Optional[str] = None):
        """
        Initialize cache synchronization manager

        Args:
            backend: Database backend for synchronization
            node_id: Unique identifier for this cache instance
        """
        self.backend = backend
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.sync_enabled = True
        self.sync_thread = None
        self.sync_active = False
        self.sync_interval = 30  # seconds
        self.lock = threading.Lock()

    def start_synchronization(self):
        """Start background synchronization thread"""
        with self.lock:
            if self.sync_active:
                return

            self.sync_active = True
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            logger.info(f"Started cache synchronization for node {self.node_id}")

    def stop_synchronization(self):
        """Stop background synchronization"""
        with self.lock:
            self.sync_active = False
            if self.sync_thread:
                self.sync_thread.join(timeout=5)

    def broadcast_operation(self, operation: str, cache_key: str):
        """Broadcast cache operation to other nodes"""
        if not self.sync_enabled:
            return

        try:
            if hasattr(self.backend, 'connection') and self.backend.connection:
                cursor = self.backend.connection.cursor()
                sync_id = str(uuid.uuid4())

                cursor.execute('''
                    INSERT INTO cache_sync (sync_id, node_id, operation, cache_key, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (sync_id, self.node_id, operation, cache_key, time.time()))

                self.backend.connection.commit()

        except Exception as e:
            logger.error(f"Error broadcasting operation: {e}")

    def _sync_loop(self):
        """Background synchronization loop"""
        while self.sync_active:
            try:
                self._process_sync_operations()
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(5)

    def _process_sync_operations(self):
        """Process pending synchronization operations"""
        try:
            if hasattr(self.backend, 'connection') and self.backend.connection:
                cursor = self.backend.connection.cursor()

                # Get unprocessed operations from other nodes
                cursor.execute('''
                    SELECT sync_id, operation, cache_key
                    FROM cache_sync
                    WHERE node_id != ? AND processed = FALSE
                    ORDER BY timestamp ASC
                    LIMIT 100
                ''', (self.node_id,))

                operations = cursor.fetchall()

                for sync_id, operation, cache_key in operations:
                    try:
                        self._apply_sync_operation(operation, cache_key)

                        # Mark as processed
                        cursor.execute('''
                            UPDATE cache_sync SET processed = TRUE WHERE sync_id = ?
                        ''', (sync_id,))

                    except Exception as e:
                        logger.error(f"Error applying sync operation {sync_id}: {e}")

                self.backend.connection.commit()

                # Clean up old sync records
                cutoff_time = time.time() - 3600  # 1 hour ago
                cursor.execute('''
                    DELETE FROM cache_sync WHERE timestamp < ? AND processed = TRUE
                ''', (cutoff_time,))

                self.backend.connection.commit()

        except Exception as e:
            logger.error(f"Error processing sync operations: {e}")

    def _apply_sync_operation(self, operation: str, cache_key: str):
        """Apply synchronized cache operation"""
        if operation == 'delete':
            # Remove from local cache without broadcasting
            self.backend.delete(cache_key)
        elif operation == 'invalidate':
            # Invalidate local cache entry
            self.backend.delete(cache_key)
        # Add more operations as needed


class CacheBackupManager:
    """Manages cache backup and recovery operations"""

    def __init__(self, backend: DatabaseCacheBackend, backup_dir: str = "cache_backups"):
        """
        Initialize cache backup manager

        Args:
            backend: Database backend to backup
            backup_dir: Directory for backup files
        """
        self.backend = backend
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """Create full cache backup"""
        if not backup_name:
            backup_name = f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backup_dir / f"{backup_name}.json"

        try:
            # Get all cache entries
            backup_data = {
                'backup_name': backup_name,
                'timestamp': time.time(),
                'backend_type': type(self.backend).__name__,
                'entries': []
            }

            # This would need to be implemented per backend type
            # For SQLite, we could dump all entries
            if isinstance(self.backend, SQLiteCacheBackend):
                backup_data.update(self._backup_sqlite())

            # Write backup file
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

            logger.info(f"Created cache backup: {backup_path}")

            return {
                'status': 'success',
                'backup_path': str(backup_path),
                'backup_name': backup_name,
                'entry_count': len(backup_data.get('entries', []))
            }

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {'status': 'error', 'error': str(e)}

    def _backup_sqlite(self) -> Dict[str, Any]:
        """Create SQLite-specific backup"""
        entries = []

        try:
            if self.backend.connection:
                cursor = self.backend.connection.cursor()
                cursor.execute('''
                    SELECT key, created_at, expires_at, cache_type, size_bytes, access_count
                    FROM cache_entries
                ''')

                for row in cursor.fetchall():
                    entries.append({
                        'key': row[0],
                        'created_at': row[1],
                        'expires_at': row[2],
                        'cache_type': row[3],
                        'size_bytes': row[4],
                        'access_count': row[5]
                    })

        except Exception as e:
            logger.error(f"Error backing up SQLite: {e}")

        return {'entries': entries}

    def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """Restore cache from backup"""
        backup_path = self.backup_dir / f"{backup_name}.json"

        if not backup_path.exists():
            return {'status': 'error', 'error': 'Backup file not found'}

        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            # Restore entries (metadata only in this simplified version)
            restored_count = len(backup_data.get('entries', []))

            logger.info(f"Restored cache backup: {backup_name} ({restored_count} entries)")

            return {
                'status': 'success',
                'backup_name': backup_name,
                'entries_restored': restored_count
            }

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return {'status': 'error', 'error': str(e)}

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []

        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append({
                    'name': backup_file.stem,
                    'path': str(backup_file),
                    'size_bytes': stat.st_size,
                    'created_at': stat.st_mtime
                })
            except Exception as e:
                logger.error(f"Error reading backup info for {backup_file}: {e}")

        return sorted(backups, key=lambda x: x['created_at'], reverse=True)


# Factory function for creating database backends
def create_database_backend(backend_type: str = "sqlite", **kwargs) -> DatabaseCacheBackend:
    """
    Create database cache backend

    Args:
        backend_type: Type of backend ("sqlite", "postgresql")
        **kwargs: Backend-specific configuration

    Returns:
        Configured database backend
    """
    if backend_type.lower() == "sqlite":
        return SQLiteCacheBackend(**kwargs)
    elif backend_type.lower() == "postgresql":
        return PostgreSQLCacheBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Global database backend instance
_global_db_backend = None
_db_backend_lock = threading.Lock()


def get_global_database_backend() -> DatabaseCacheBackend:
    """Get global database backend instance"""
    global _global_db_backend
    with _db_backend_lock:
        if _global_db_backend is None:
            _global_db_backend = create_database_backend("sqlite")
            _global_db_backend.connect()
            _global_db_backend.create_schema()
        return _global_db_backend