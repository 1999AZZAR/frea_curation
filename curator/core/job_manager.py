"""
Job manager for handling background task submission and status tracking.

Provides a high-level interface for submitting analysis jobs and monitoring
their progress through the Celery task queue.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
import json

from curator.core.celery_app import celery_app
from curator.core.models import ScoringConfig

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages background job submission and status tracking.
    
    Provides methods to submit analysis jobs, check their status,
    and retrieve results when complete.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize job manager with Redis connection.
        
        Args:
            redis_url: Redis connection URL (uses REDIS_URL env var if None)
        """
        import os
        self.redis_url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def submit_batch_analysis(
        self,
        urls: List[str],
        query: Optional[str] = None,
        config: Optional[ScoringConfig] = None,
        apply_diversity: Optional[bool] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Submit a batch analysis job to the background queue.
        
        Args:
            urls: List of article URLs to analyze
            query: Optional query for relevance scoring
            config: Scoring configuration
            apply_diversity: Whether to apply diversity filtering
            job_id: Optional custom job ID (auto-generated if None)
            
        Returns:
            Job ID for tracking the task
        """
        # Convert config to dictionary for serialization
        config_dict = None
        if config:
            config_dict = {
                'readability_weight': config.readability_weight,
                'ner_density_weight': config.ner_density_weight,
                'sentiment_weight': config.sentiment_weight,
                'tfidf_relevance_weight': config.tfidf_relevance_weight,
                'recency_weight': config.recency_weight,
                'reputation_weight': getattr(config, 'reputation_weight', 0.0),
                'topic_coherence_weight': getattr(config, 'topic_coherence_weight', 0.0),
                'min_word_count': config.min_word_count,
                'max_articles_per_topic': config.max_articles_per_topic,
                'default_recency_half_life_days': getattr(config, 'default_recency_half_life_days', 7.0),
                'topic_half_life_days': getattr(config, 'topic_half_life_days', {}),
            }
        
        # Submit task
        task = celery_app.send_task(
            'curator.batch_analyze_task',
            args=[urls, query, config_dict, apply_diversity],
            task_id=job_id,
        )
        
        # Store job metadata in Redis
        if self.redis_client:
            try:
                job_data = {
                    'task_id': task.id,
                    'type': 'batch_analysis',
                    'submitted_at': datetime.utcnow().isoformat(),
                    'urls_count': len(urls),
                    'query': query,
                    'apply_diversity': apply_diversity,
                }
                self.redis_client.setex(
                    f"job:{task.id}",
                    timedelta(hours=24),  # Expire after 24 hours
                    json.dumps(job_data)
                )
            except Exception as e:
                logger.warning(f"Failed to store job metadata: {e}")
        
        logger.info(f"Submitted batch analysis job {task.id} for {len(urls)} URLs")
        return task.id
    
    def submit_single_analysis(
        self,
        url: str,
        query: Optional[str] = None,
        config: Optional[ScoringConfig] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Submit a single article analysis job to the background queue.
        
        Args:
            url: Article URL to analyze
            query: Optional query for relevance scoring
            config: Scoring configuration
            job_id: Optional custom job ID (auto-generated if None)
            
        Returns:
            Job ID for tracking the task
        """
        # Convert config to dictionary for serialization
        config_dict = None
        if config:
            config_dict = {
                'readability_weight': config.readability_weight,
                'ner_density_weight': config.ner_density_weight,
                'sentiment_weight': config.sentiment_weight,
                'tfidf_relevance_weight': config.tfidf_relevance_weight,
                'recency_weight': config.recency_weight,
                'reputation_weight': getattr(config, 'reputation_weight', 0.0),
                'topic_coherence_weight': getattr(config, 'topic_coherence_weight', 0.0),
                'min_word_count': config.min_word_count,
                'max_articles_per_topic': config.max_articles_per_topic,
                'default_recency_half_life_days': getattr(config, 'default_recency_half_life_days', 7.0),
                'topic_half_life_days': getattr(config, 'topic_half_life_days', {}),
            }
        
        # Submit task
        task = celery_app.send_task(
            'curator.analyze_single_task',
            args=[url, query, config_dict],
            task_id=job_id,
        )
        
        # Store job metadata in Redis
        if self.redis_client:
            try:
                job_data = {
                    'task_id': task.id,
                    'type': 'single_analysis',
                    'submitted_at': datetime.utcnow().isoformat(),
                    'url': url,
                    'query': query,
                }
                self.redis_client.setex(
                    f"job:{task.id}",
                    timedelta(hours=24),  # Expire after 24 hours
                    json.dumps(job_data)
                )
            except Exception as e:
                logger.warning(f"Failed to store job metadata: {e}")
        
        logger.info(f"Submitted single analysis job {task.id} for URL: {url}")
        return task.id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a background job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Dictionary containing job status and metadata
        """
        try:
            # Get task result from Celery
            task_result = celery_app.AsyncResult(job_id)
            
            # Get job metadata from Redis
            job_metadata = {}
            if self.redis_client:
                try:
                    metadata_json = self.redis_client.get(f"job:{job_id}")
                    if metadata_json:
                        job_metadata = json.loads(metadata_json)
                except Exception as e:
                    logger.warning(f"Failed to get job metadata: {e}")
            
            # Build status response
            status = {
                'job_id': job_id,
                'state': task_result.state,
                'submitted_at': job_metadata.get('submitted_at'),
                'type': job_metadata.get('type', 'unknown'),
            }
            
            if task_result.state == 'PENDING':
                status.update({
                    'status': 'Job is waiting to be processed',
                    'current': 0,
                    'total': job_metadata.get('urls_count', 1),
                })
            elif task_result.state == 'PROGRESS':
                # Task is running, get progress info
                info = task_result.info or {}
                status.update({
                    'status': info.get('status', 'Processing...'),
                    'current': info.get('current', 0),
                    'total': info.get('total', job_metadata.get('urls_count', 1)),
                    'processed_urls': info.get('processed_urls', []),
                    'failed_urls': info.get('failed_urls', []),
                })
            elif task_result.state == 'SUCCESS':
                # Task completed successfully
                result = task_result.result or {}
                status.update({
                    'status': 'Job completed successfully',
                    'current': result.get('processed_count', 0),
                    'total': result.get('total_urls', job_metadata.get('urls_count', 1)),
                    'result': result,
                })
            elif task_result.state == 'FAILURE':
                # Task failed
                info = task_result.info or {}
                status.update({
                    'status': f"Job failed: {info.get('error', 'Unknown error')}",
                    'error': info.get('error', str(task_result.result)),
                })
            else:
                # Unknown state
                status.update({
                    'status': f'Job is in state: {task_result.state}',
                })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return {
                'job_id': job_id,
                'state': 'ERROR',
                'status': f'Failed to get job status: {str(e)}',
                'error': str(e),
            }
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed job.
        
        Args:
            job_id: Job ID to get result for
            
        Returns:
            Job result if available, None otherwise
        """
        try:
            task_result = celery_app.AsyncResult(job_id)
            
            if task_result.state == 'SUCCESS':
                return task_result.result
            elif task_result.state == 'FAILURE':
                return {
                    'error': str(task_result.result),
                    'state': 'FAILURE',
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get job result for {job_id}: {e}")
            return {
                'error': str(e),
                'state': 'ERROR',
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False otherwise
        """
        try:
            celery_app.control.revoke(job_id, terminate=True)
            
            # Clean up job metadata
            if self.redis_client:
                try:
                    self.redis_client.delete(f"job:{job_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean up job metadata: {e}")
            
            logger.info(f"Cancelled job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """
        List all active jobs.
        
        Returns:
            List of active job information
        """
        active_jobs = []
        
        if not self.redis_client:
            return active_jobs
        
        try:
            # Get all job keys
            job_keys = self.redis_client.keys("job:*")
            
            for key in job_keys:
                try:
                    job_data = json.loads(self.redis_client.get(key))
                    job_id = job_data.get('task_id')
                    
                    if job_id:
                        status = self.get_job_status(job_id)
                        if status.get('state') in ['PENDING', 'PROGRESS']:
                            active_jobs.append(status)
                            
                except Exception as e:
                    logger.warning(f"Failed to process job key {key}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to list active jobs: {e}")
        
        return active_jobs
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the job processing system.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            'redis_connected': False,
            'celery_workers': 0,
            'active_jobs': 0,
        }
        
        # Check Redis connection
        try:
            if self.redis_client:
                self.redis_client.ping()
                health_status['redis_connected'] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check Celery workers
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            if stats:
                health_status['celery_workers'] = len(stats)
        except Exception as e:
            logger.error(f"Celery health check failed: {e}")
        
        # Count active jobs
        try:
            active_jobs = self.list_active_jobs()
            health_status['active_jobs'] = len(active_jobs)
        except Exception as e:
            logger.error(f"Active jobs count failed: {e}")
        
        return health_status