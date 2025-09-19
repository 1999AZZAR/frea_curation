"""
Unit tests for background job processing system.

Tests the Celery task system, job manager, and Flask routes
for background job processing functionality.
"""

import os
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Set test environment
os.environ['CELERY_ALWAYS_EAGER'] = 'True'  # Run tasks synchronously in tests
os.environ['REDIS_URL'] = 'redis://localhost:6379/1'  # Use test database


class TestCeleryTasks:
    """Test Celery task functions."""
    
    def test_health_check_task(self):
        """Test health check task."""
        from curator.core.tasks import health_check_task
        
        result = health_check_task()
        
        assert result['status'] == 'healthy'
        assert 'timestamp' in result
        assert isinstance(result['timestamp'], str)
    
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.nlp.get_spacy_model')
    @patch('curator.core.nlp.get_vader_analyzer')
    def test_batch_analyze_task_success(self, mock_vader, mock_spacy, mock_batch_analyze):
        """Test successful batch analysis task."""
        from curator.core.tasks import batch_analyze_task
        from curator.core.models import Article, ScoreCard
        
        # Mock dependencies
        mock_spacy.return_value = Mock()
        mock_vader.return_value = Mock()
        
        # Mock batch_analyze result
        mock_article = Article(
            url='https://example.com/test',
            title='Test Article',
            author='Test Author',
            content='Test content',
            summary='Test summary'
        )
        mock_scorecard = ScoreCard(
            overall_score=85.0,
            readability_score=80.0,
            ner_density_score=75.0,
            sentiment_score=90.0,
            tfidf_relevance_score=85.0,
            recency_score=95.0,
            reputation_score=70.0,
            topic_coherence_score=80.0,
            article=mock_article
        )
        mock_batch_analyze.return_value = [mock_scorecard]
        
        # Create a mock task instance
        mock_task = Mock()
        mock_task.update_state = Mock()
        
        # Test the task
        urls = ['https://example.com/test']
        result = batch_analyze_task(mock_task, urls, query='test', apply_diversity=False)
        
        # Verify result
        assert result['status'] == 'SUCCESS'
        assert result['total_urls'] == 1
        assert result['processed_count'] == 1
        assert result['failed_count'] == 0
        assert len(result['results']) == 1
        
        # Verify task state updates were called
        assert mock_task.update_state.called
    
    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.nlp.get_spacy_model')
    @patch('curator.core.nlp.get_vader_analyzer')
    def test_analyze_single_task_success(self, mock_vader, mock_spacy, mock_analyze):
        """Test successful single analysis task."""
        from curator.core.tasks import analyze_single_task
        from curator.core.models import Article, ScoreCard
        
        # Mock dependencies
        mock_spacy.return_value = Mock()
        mock_vader.return_value = Mock()
        
        # Mock analyze_article result
        mock_article = Article(
            url='https://example.com/test',
            title='Test Article',
            author='Test Author',
            content='Test content',
            summary='Test summary'
        )
        mock_scorecard = ScoreCard(
            overall_score=85.0,
            readability_score=80.0,
            ner_density_score=75.0,
            sentiment_score=90.0,
            tfidf_relevance_score=85.0,
            recency_score=95.0,
            reputation_score=70.0,
            topic_coherence_score=80.0,
            article=mock_article
        )
        mock_analyze.return_value = mock_scorecard
        
        # Create a mock task instance
        mock_task = Mock()
        mock_task.update_state = Mock()
        
        # Test the task
        result = analyze_single_task(mock_task, 'https://example.com/test', query='test')
        
        # Verify result
        assert result['status'] == 'SUCCESS'
        assert result['url'] == 'https://example.com/test'
        assert result['query'] == 'test'
        assert 'result' in result
        
        # Verify task state updates were called
        assert mock_task.update_state.called


class TestJobManager:
    """Test JobManager functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch('curator.core.job_manager.redis.from_url') as mock_redis_from_url:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.setex = Mock()
            mock_client.get = Mock()
            mock_client.delete = Mock()
            mock_client.keys = Mock(return_value=[])
            mock_redis_from_url.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_celery(self):
        """Mock Celery app."""
        with patch('curator.core.job_manager.celery_app') as mock_app:
            mock_task = Mock()
            mock_task.id = 'test-job-id'
            mock_app.send_task.return_value = mock_task
            mock_app.AsyncResult.return_value = Mock(state='PENDING', info=None, result=None)
            mock_app.control.revoke = Mock()
            mock_app.control.inspect.return_value.stats.return_value = {'worker1': {}}
            yield mock_app
    
    def test_job_manager_init(self, mock_redis):
        """Test JobManager initialization."""
        from curator.core.job_manager import JobManager
        
        job_manager = JobManager()
        
        assert job_manager.redis_client is not None
        mock_redis.ping.assert_called_once()
    
    def test_submit_single_analysis(self, mock_redis, mock_celery):
        """Test submitting single analysis job."""
        from curator.core.job_manager import JobManager
        from curator.core.models import ScoringConfig
        
        job_manager = JobManager()
        config = ScoringConfig()
        
        job_id = job_manager.submit_single_analysis(
            url='https://example.com/test',
            query='test query',
            config=config
        )
        
        assert job_id == 'test-job-id'
        mock_celery.send_task.assert_called_once()
        mock_redis.setex.assert_called_once()
    
    def test_submit_batch_analysis(self, mock_redis, mock_celery):
        """Test submitting batch analysis job."""
        from curator.core.job_manager import JobManager
        from curator.core.models import ScoringConfig
        
        job_manager = JobManager()
        config = ScoringConfig()
        
        urls = ['https://example.com/test1', 'https://example.com/test2']
        job_id = job_manager.submit_batch_analysis(
            urls=urls,
            query='test query',
            config=config,
            apply_diversity=True
        )
        
        assert job_id == 'test-job-id'
        mock_celery.send_task.assert_called_once()
        mock_redis.setex.assert_called_once()
    
    def test_get_job_status(self, mock_redis, mock_celery):
        """Test getting job status."""
        from curator.core.job_manager import JobManager
        
        # Mock Redis metadata
        mock_redis.get.return_value = json.dumps({
            'task_id': 'test-job-id',
            'type': 'single_analysis',
            'submitted_at': '2023-01-01T00:00:00',
            'url': 'https://example.com/test'
        })
        
        job_manager = JobManager()
        status = job_manager.get_job_status('test-job-id')
        
        assert status['job_id'] == 'test-job-id'
        assert status['state'] == 'PENDING'
        assert status['type'] == 'single_analysis'
    
    def test_cancel_job(self, mock_redis, mock_celery):
        """Test cancelling a job."""
        from curator.core.job_manager import JobManager
        
        job_manager = JobManager()
        success = job_manager.cancel_job('test-job-id')
        
        assert success is True
        mock_celery.control.revoke.assert_called_once_with('test-job-id', terminate=True)
        mock_redis.delete.assert_called_once()
    
    def test_health_check(self, mock_redis, mock_celery):
        """Test health check."""
        from curator.core.job_manager import JobManager
        
        job_manager = JobManager()
        health = job_manager.health_check()
        
        assert health['redis_connected'] is True
        assert health['celery_workers'] == 1
        assert health['active_jobs'] == 0


class TestFlaskRoutes:
    """Test Flask routes for job management."""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        from app import create_app
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    @patch('curator.core.job_manager.JobManager')
    def test_submit_job_single_analysis(self, mock_job_manager_class, client):
        """Test submitting single analysis job via API."""
        # Mock JobManager
        mock_job_manager = Mock()
        mock_job_manager.submit_single_analysis.return_value = 'test-job-id'
        mock_job_manager_class.return_value = mock_job_manager
        
        response = client.post('/jobs', 
            json={
                'type': 'single_analysis',
                'url': 'https://example.com/test',
                'query': 'test query'
            }
        )
        
        assert response.status_code == 202
        data = response.get_json()
        assert data['job_id'] == 'test-job-id'
        assert data['status'] == 'submitted'
        assert data['type'] == 'single_analysis'
    
    @patch('curator.core.job_manager.JobManager')
    def test_submit_job_batch_analysis(self, mock_job_manager_class, client):
        """Test submitting batch analysis job via API."""
        # Mock JobManager
        mock_job_manager = Mock()
        mock_job_manager.submit_batch_analysis.return_value = 'test-job-id'
        mock_job_manager_class.return_value = mock_job_manager
        
        response = client.post('/jobs',
            json={
                'type': 'batch_analysis',
                'urls': ['https://example.com/test1', 'https://example.com/test2'],
                'query': 'test query',
                'apply_diversity': True
            }
        )
        
        assert response.status_code == 202
        data = response.get_json()
        assert data['job_id'] == 'test-job-id'
        assert data['status'] == 'submitted'
        assert data['type'] == 'batch_analysis'
    
    @patch('curator.core.job_manager.JobManager')
    def test_job_status_api(self, mock_job_manager_class, client):
        """Test job status API endpoint."""
        # Mock JobManager
        mock_job_manager = Mock()
        mock_job_manager.get_job_status.return_value = {
            'job_id': 'test-job-id',
            'state': 'PROGRESS',
            'status': 'Processing...',
            'current': 5,
            'total': 10
        }
        mock_job_manager_class.return_value = mock_job_manager
        
        response = client.get('/jobs/test-job-id/status')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['job_id'] == 'test-job-id'
        assert data['state'] == 'PROGRESS'
        assert data['current'] == 5
        assert data['total'] == 10
    
    @patch('curator.core.job_manager.JobManager')
    def test_cancel_job_api(self, mock_job_manager_class, client):
        """Test job cancellation API endpoint."""
        # Mock JobManager
        mock_job_manager = Mock()
        mock_job_manager.cancel_job.return_value = True
        mock_job_manager_class.return_value = mock_job_manager
        
        response = client.delete('/jobs/test-job-id')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'cancelled'
        assert data['job_id'] == 'test-job-id'
    
    @patch('curator.core.job_manager.JobManager')
    def test_jobs_health_check(self, mock_job_manager_class, client):
        """Test jobs health check endpoint."""
        # Mock JobManager
        mock_job_manager = Mock()
        mock_job_manager.health_check.return_value = {
            'redis_connected': True,
            'celery_workers': 2,
            'active_jobs': 1
        }
        mock_job_manager_class.return_value = mock_job_manager
        
        response = client.get('/health/jobs')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['redis_connected'] is True
        assert data['celery_workers'] == 2
        assert data['active_jobs'] == 1
    
    def test_invalid_job_type(self, client):
        """Test submitting job with invalid type."""
        response = client.post('/jobs',
            json={
                'type': 'invalid_type',
                'url': 'https://example.com/test'
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_missing_job_type(self, client):
        """Test submitting job without type."""
        response = client.post('/jobs',
            json={
                'url': 'https://example.com/test'
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'Job type is required' in data['error']


if __name__ == '__main__':
    pytest.main([__file__])