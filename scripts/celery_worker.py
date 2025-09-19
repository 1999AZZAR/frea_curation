#!/usr/bin/env python3
"""
Celery worker script for running background tasks.

Usage:
    python celery_worker.py

Environment variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
    CELERY_LOG_LEVEL: Log level for Celery worker (default: INFO)
    CELERY_CONCURRENCY: Number of worker processes (default: 4)
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('CELERY_LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    from curator.core.celery_app import celery_app
    
    # Worker configuration
    concurrency = int(os.environ.get('CELERY_CONCURRENCY', '4'))
    log_level = os.environ.get('CELERY_LOG_LEVEL', 'INFO').lower()
    
    print(f"Starting Celery worker with concurrency={concurrency}, log_level={log_level}")
    print(f"Redis URL: {os.environ.get('REDIS_URL', 'redis://localhost:6379/0')}")
    
    # Start the worker
    celery_app.worker_main([
        'worker',
        '--loglevel', log_level,
        '--concurrency', str(concurrency),
        '--pool', 'prefork',  # Use prefork pool for better isolation
        '--without-gossip',   # Disable gossip for better performance
        '--without-mingle',   # Disable mingle for faster startup
        '--without-heartbeat', # Disable heartbeat for simpler setup
    ])