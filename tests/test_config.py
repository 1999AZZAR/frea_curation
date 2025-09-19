"""
Test configuration and utilities for integration tests.

Provides common test settings, fixtures, and helper functions
used across multiple test modules.
"""

import os
import tempfile
from unittest.mock import Mock
from datetime import datetime, timedelta

try:
    from curator.core.models import Article, ScoreCard, Entity, ScoringConfig
except ImportError:
    # Fallback for existing project structure
    from models import Article, ScoreCard, Entity, ScoringConfig


class TestConfig:
    """Test configuration constants and settings."""
    
    # Performance thresholds
    SINGLE_ANALYSIS_THRESHOLD = 3.0    # seconds
    HOMEPAGE_LOAD_THRESHOLD = 1.0      # seconds  
    BATCH_ANALYSIS_THRESHOLD = 10.0    # seconds for 5 articles
    MEMORY_INCREASE_THRESHOLD = 100    # MB per request
    
    # Test data settings
    MIN_WORD_COUNT = 50               # Lower threshold for testing
    MAX_ARTICLES_PER_TOPIC = 20       # Default max articles
    
    # Concurrent testing settings
    MAX_CONCURRENT_THREADS = 5        # Max threads for concurrent tests
    SUSTAINED_LOAD_REQUESTS = 20      # Number of requests for sustained load test
    
    # External service settings
    MOCK_API_DELAY = 0.1             # Simulated API response delay (seconds)
    MOCK_PARSING_DELAY = 0.05        # Simulated parsing delay (seconds)


class TestFixtures:
    """Common test fixtures and sample data."""
    
    @staticmethod
    def create_sample_article(url_suffix="", title_suffix="", content_multiplier=1):
        """Create a sample article for testing."""
        base_content = (
            "Artificial intelligence researchers have achieved a significant breakthrough "
            "in natural language processing. The new deep learning model demonstrates "
            "unprecedented accuracy in understanding human language nuances. This "
            "advancement could revolutionize how computers interact with humans. "
            "Machine learning algorithms continue to evolve rapidly. The research team "
            "published their findings in a peer-reviewed journal. Industry experts "
            "predict widespread adoption within the next few years. The technology "
            "shows promise for applications in healthcare, education, and customer service. "
        )
        
        return Article(
            url=f"https://example.com/article{url_suffix}",
            title=f"Revolutionary AI Breakthrough{title_suffix}",
            author="Tech Reporter",
            publish_date=datetime.now() - timedelta(hours=2),
            content=base_content * content_multiplier,
            summary="AI researchers achieve breakthrough in natural language processing.",
            entities=[
                Entity(text="AI", label="ORG", confidence=0.95),
                Entity(text="Natural Language Processing", label="TECH", confidence=0.90),
                Entity(text="Machine Learning", label="TECH", confidence=0.85)
            ]
        )
    
    @staticmethod
    def create_sample_scorecard(article=None, overall_score=85.0):
        """Create a sample scorecard for testing."""
        if article is None:
            article = TestFixtures.create_sample_article()
            
        return ScoreCard(
            overall_score=overall_score,
            readability_score=80.0,
            ner_density_score=75.0,
            sentiment_score=70.0,
            tfidf_relevance_score=90.0,
            recency_score=95.0,
            reputation_score=75.0,
            topic_coherence_score=85.0,
            article=article
        )
    
    @staticmethod
    def create_test_config():
        """Create a test scoring configuration."""
        return ScoringConfig(
            readability_weight=0.15,
            ner_density_weight=0.15,
            sentiment_weight=0.15,
            tfidf_relevance_weight=0.25,
            recency_weight=0.15,
            reputation_weight=0.15,
            min_word_count=TestConfig.MIN_WORD_COUNT,
            max_articles_per_topic=TestConfig.MAX_ARTICLES_PER_TOPIC
        )
    
    @staticmethod
    def create_newsapi_response(num_articles=3):
        """Create a mock NewsAPI response."""
        articles = []
        for i in range(num_articles):
            articles.append({
                "source": {"id": f"source{i}", "name": f"Source {i}"},
                "author": f"Author {i}",
                "title": f"Test Article {i}",
                "description": f"Description for article {i}",
                "url": f"https://example.com/news{i}",
                "urlToImage": f"https://example.com/image{i}.jpg",
                "publishedAt": (datetime.now() - timedelta(hours=i)).isoformat() + "Z",
                "content": f"Content for test article {i}..."
            })
        
        return {
            "status": "ok",
            "totalResults": num_articles,
            "articles": articles
        }


class TestHelpers:
    """Helper functions for integration tests."""
    
    @staticmethod
    def create_temp_env_file(**env_vars):
        """Create a temporary .env file with specified variables."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False)
        
        for key, value in env_vars.items():
            temp_file.write(f"{key}={value}\n")
        
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def mock_successful_analysis(url, **kwargs):
        """Create a mock successful analysis result."""
        article = TestFixtures.create_sample_article(url_suffix=f"_{hash(url) % 1000}")
        article.url = url
        return TestFixtures.create_sample_scorecard(article)
    
    @staticmethod
    def mock_failing_analysis(url, **kwargs):
        """Create a mock failing analysis."""
        raise Exception(f"Analysis failed for {url}")
    
    @staticmethod
    def mock_news_source(num_articles=5):
        """Create a mock NewsSource instance."""
        mock_source = Mock()
        mock_source.get_article_urls.return_value = [
            f"https://example.com/news{i}" for i in range(num_articles)
        ]
        return mock_source
    
    @staticmethod
    def assert_valid_scorecard(test_case, scorecard):
        """Assert that a scorecard has valid structure and values."""
        test_case.assertIsInstance(scorecard, ScoreCard)
        
        # Check score ranges
        test_case.assertGreaterEqual(scorecard.overall_score, 0)
        test_case.assertLessEqual(scorecard.overall_score, 100)
        test_case.assertGreaterEqual(scorecard.readability_score, 0)
        test_case.assertLessEqual(scorecard.readability_score, 100)
        test_case.assertGreaterEqual(scorecard.ner_density_score, 0)
        test_case.assertLessEqual(scorecard.ner_density_score, 100)
        test_case.assertGreaterEqual(scorecard.sentiment_score, 0)
        test_case.assertLessEqual(scorecard.sentiment_score, 100)
        test_case.assertGreaterEqual(scorecard.tfidf_relevance_score, 0)
        test_case.assertLessEqual(scorecard.tfidf_relevance_score, 100)
        test_case.assertGreaterEqual(scorecard.recency_score, 0)
        test_case.assertLessEqual(scorecard.recency_score, 100)
        test_case.assertGreaterEqual(scorecard.reputation_score, 0)
        test_case.assertLessEqual(scorecard.reputation_score, 100)
        test_case.assertGreaterEqual(scorecard.topic_coherence_score, 0)
        test_case.assertLessEqual(scorecard.topic_coherence_score, 100)
        
        # Check article structure
        test_case.assertIsInstance(scorecard.article, Article)
        test_case.assertIsNotNone(scorecard.article.url)
        test_case.assertIsNotNone(scorecard.article.title)
    
    @staticmethod
    def assert_valid_json_response(test_case, response, expected_keys=None):
        """Assert that a JSON response has valid structure."""
        test_case.assertEqual(response.status_code, 200)
        test_case.assertEqual(response.content_type, 'application/json')
        
        data = response.get_json()
        test_case.assertIsInstance(data, dict)
        
        if expected_keys:
            for key in expected_keys:
                test_case.assertIn(key, data)
    
    @staticmethod
    def measure_performance(func, *args, **kwargs):
        """Measure execution time and memory usage of a function."""
        import time
        import psutil
        import os
        import gc
        
        # Force garbage collection before measurement
        gc.collect()
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_increase': memory_after - memory_before
        }


class MockEnvironment:
    """Context manager for temporarily setting environment variables."""
    
    def __init__(self, **env_vars):
        self.env_vars = env_vars
        self.original_values = {}
    
    def __enter__(self):
        # Save original values and set new ones
        for key, value in self.env_vars.items():
            self.original_values[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key in self.env_vars:
            if self.original_values[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = self.original_values[key]


# Test data constants
SAMPLE_URLS = [
    "https://techcrunch.com/ai-breakthrough",
    "https://wired.com/machine-learning-advance", 
    "https://arxiv.org/abs/2024.01234",
    "https://nature.com/articles/nature12345",
    "https://blog.openai.com/new-model"
]

SAMPLE_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "natural language processing",
    "computer vision",
    "robotics"
]

INVALID_URLS = [
    "",
    "not-a-url",
    "http://",
    "ftp://invalid-protocol.com",
    "javascript:alert('xss')"
]

INVALID_TOPICS = [
    "",
    "   ",
    "a" * 1000,  # Too long
    "!@#$%^&*()",  # Special characters only
]