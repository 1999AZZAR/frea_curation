"""
Celery application configuration for background job processing.

Provides asynchronous task processing for batch analysis operations
using Redis as the message broker and result backend.
"""

import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def make_celery():
    """Create and configure Celery application instance."""
    # Redis configuration
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Create Celery instance
    celery = Celery(
        'curator',
        broker=redis_url,
        backend=redis_url,
        include=['curator.core.tasks']
    )
    
    # Configuration
    celery.conf.update(
        # Task routing
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Task execution
        task_always_eager=os.environ.get('CELERY_ALWAYS_EAGER', 'False').lower() == 'true',
        task_eager_propagates=True,
        
        # Result backend settings
        result_expires=3600,  # 1 hour
        result_backend_transport_options={
            'master_name': 'mymaster',
            'visibility_timeout': 3600,
        },
        
        # Worker settings
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=1000,
        
        # Task time limits
        task_soft_time_limit=300,  # 5 minutes
        task_time_limit=600,       # 10 minutes
        
        # Retry settings
        task_default_retry_delay=60,
        task_max_retries=3,
    )
    
    return celery

# Create the Celery instance
celery_app = make_celery()