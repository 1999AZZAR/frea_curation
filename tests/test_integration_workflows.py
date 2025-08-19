"""
Comprehensive integration tests for complete user workflows.

Tests end-to-end user scenarios, external service integration,
performance requirements, and concurrent usage patterns.
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import json
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
import os
import tempfile
import responses

from app import create_app
try:
    from curator.core.models import Article, ScoreCard, ScoringConfig, Entity
    from curator.services.news_source import NewsSource
except ImportError:
    # Fallback for existing project structure
    from models import Article, ScoreCard, ScoringConfig, Entity
    try:
        from curator.services.news_source import NewsSource
    except ImportError:
        NewsSource = None


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end user workflows."""
    
    def setUp(self):
        """Set up test client and fixtures."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Sample article data for mocking
        self.sample_article = Article(
            url="https://example.com/tech-article",
            title="Revolutionary AI Breakthrough in Natural Language Processing",
            author="Tech Reporter",
            publish_date=datetime.now() - timedelta(hours=2),
            content="Artificial intelligence researchers have achieved a significant breakthrough "
                   "in natural language processing. The new deep learning model demonstrates "
                   "unprecedented accuracy in understanding human language nuances. This "
                   "advancement could revolutionize how computers interact with humans. "
                   "Machine learning algorithms continue to evolve rapidly. The research team "
                   "published their findings in a peer-reviewed journal. Industry experts "
                   "predict widespread adoption within the next few years. The technology "
                   "shows promise for applications in healthcare, education, and customer service.",
            summary="AI researchers achieve breakthrough in natural language processing with new deep learning model.",
            entities=[
                Entity(text="AI", label="ORG", confidence=0.95),
                Entity(text="Natural Language Processing", label="TECH", confidence=0.90)
            ]
        )
        
        self.sample_scorecard = ScoreCard(
            overall_score=85.5,
            readability_score=80.0,
            ner_density_score=75.0,
            sentiment_score=60.0,
            tfidf_relevance_score=90.0,
            recency_score=95.0,
            article=self.sample_article
        )

    def test_homepage_loads_successfully(self):
        """Test that homepage loads with proper UI elements."""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Curator', response.data)
        # Check for form elements
        self.assertIn(b'analyze', response.data.lower())
        self.assertIn(b'curate', response.data.lower())

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_manual_analysis_complete_workflow_json(self, mock_validate, mock_analyze):
        """Test complete manual analysis workflow with JSON response."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        # Test JSON request
        response = self.client.post('/analyze', 
                                  json={
                                      'url': 'https://example.com/tech-article',
                                      'query': 'artificial intelligence'
                                  },
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Verify response structure
        self.assertIn('overall_score', data)
        self.assertIn('article', data)
        self.assertIn('stats', data)
        self.assertEqual(data['overall_score'], 85.5)
        self.assertEqual(data['article']['title'], self.sample_article.title)
        self.assertEqual(len(data['article']['entities']), 2)

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_manual_analysis_complete_workflow_form(self, mock_validate, mock_analyze):
        """Test complete manual analysis workflow with form submission."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        mock_analyze.return_value = self.sample_scorecard
        
        # Test form submission
        response = self.client.post('/analyze', 
                                  data={
                                      'url': 'https://example.com/tech-article',
                                      'query': 'artificial intelligence'
                                  })
        
        self.assertEqual(response.status_code, 200)
        # Should render HTML template
        self.assertIn(b'85.5', response.data)  # Overall score
        self.assertIn(b'Revolutionary AI Breakthrough', response.data)  # Title

    def test_manual_analysis_invalid_url_workflow(self):
        """Test manual analysis workflow with invalid URL."""
        response = self.client.post('/analyze', 
                                  json={'url': 'invalid-url'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_topic_curation_complete_workflow_json(self, mock_validate, mock_batch, mock_source):
        """Test complete topic curation workflow with JSON response."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = [
            'https://example.com/article1',
            'https://example.com/article2'
        ]
        mock_source.return_value = mock_instance
        
        # Create multiple scorecards for ranking test
        scorecard1 = ScoreCard(
            overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
            sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
            article=Article(url="https://example.com/article1", title="Article 1", 
                          author="Author 1", summary="Summary 1")
        )
        scorecard2 = ScoreCard(
            overall_score=90.0, readability_score=85.0, ner_density_score=80.0,
            sentiment_score=75.0, tfidf_relevance_score=95.0, recency_score=90.0,
            article=Article(url="https://example.com/article2", title="Article 2",
                          author="Author 2", summary="Summary 2")
        )
        
        mock_batch.return_value = [scorecard1, scorecard2]
        
        # Test JSON request
        response = self.client.post('/curate-topic',
                                  json={
                                      'topic': 'artificial intelligence',
                                      'max_articles': 10
                                  },
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Verify response structure and sorting
        self.assertIn('count', data)
        self.assertIn('results', data)
        self.assertEqual(data['count'], 2)
        
        # Should be sorted by overall_score descending
        self.assertEqual(data['results'][0]['overall_score'], 90.0)
        self.assertEqual(data['results'][1]['overall_score'], 75.0)

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_topic_curation_complete_workflow_form(self, mock_validate, mock_batch, mock_source):
        """Test complete topic curation workflow with form submission."""
        # Setup mocks
        mock_validate.return_value = (True, None)
        
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = ['https://example.com/article1']
        mock_source.return_value = mock_instance
        
        scorecard = ScoreCard(
            overall_score=85.0, readability_score=80.0, ner_density_score=75.0,
            sentiment_score=70.0, tfidf_relevance_score=90.0, recency_score=95.0,
            article=Article(url="https://example.com/article1", title="Test Article",
                          author="Test Author", summary="Test Summary")
        )
        mock_batch.return_value = [scorecard]
        
        # Test form submission
        response = self.client.post('/curate-topic',
                                  data={
                                      'topic': 'artificial intelligence',
                                      'max_articles': '5'
                                  })
        
        self.assertEqual(response.status_code, 200)
        # Should render HTML template
        self.assertIn(b'Test Article', response.data)
        self.assertIn(b'85', response.data)  # Score

    def test_topic_curation_invalid_topic_workflow(self):
        """Test topic curation workflow with invalid topic."""
        response = self.client.post('/curate-topic',
                                  json={'topic': ''},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    @patch('curator.services.parser.parse_article')
    @patch('curator.core.validation.validate_url')
    def test_compare_workflow_urls(self, mock_validate, mock_parse):
        """Test similarity comparison workflow with URLs."""
        mock_validate.return_value = (True, None)
        
        article_a = Article(url="https://example.com/a", title="Article A", 
                          content="Content about artificial intelligence")
        article_b = Article(url="https://example.com/b", title="Article B",
                          content="Content about machine learning")
        
        mock_parse.side_effect = [article_a, article_b]
        
        response = self.client.post('/compare',
                                  json={
                                      'a_url': 'https://example.com/a',
                                      'b_url': 'https://example.com/b',
                                      'use_embeddings': False
                                  },
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Verify response structure
        self.assertIn('a', data)
        self.assertIn('b', data)
        self.assertIn('tfidf', data)
        self.assertIn('a_to_b', data['tfidf'])
        self.assertIn('b_to_a', data['tfidf'])
        self.assertIn('avg', data['tfidf'])

    def test_compare_workflow_text(self):
        """Test similarity comparison workflow with raw text."""
        response = self.client.post('/compare',
                                  json={
                                      'a_text': 'Content about artificial intelligence and machine learning',
                                      'b_text': 'Article discussing AI and ML technologies',
                                      'use_embeddings': False
                                  },
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        # Verify response structure
        self.assertIn('tfidf', data)
        self.assertGreater(data['tfidf']['avg'], 0)  # Should have some similarity

    def test_compare_page_loads(self):
        """Test that compare page loads successfully."""
        response = self.client.get('/compare')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'compare', response.data.lower())

    def test_error_pages_workflow(self):
        """Test error page rendering workflow."""
        # Test 404
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'not found', response.data.lower())
        
        # Test 400 with invalid analyze request
        response = self.client.post('/analyze', data={})
        self.assertEqual(response.status_code, 400)

    @patch('curator.services.analyzer.analyze_article')
    def test_analysis_error_handling_workflow(self, mock_analyze):
        """Test error handling in analysis workflow."""
        mock_analyze.side_effect = Exception("Analysis failed")
        
        response = self.client.post('/analyze',
                                  json={'url': 'https://example.com/article'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Analysis failed', data['error'])


class TestExternalServiceIntegration(unittest.TestCase):
    """Test integration with external services using mocked responses."""
    
    def setUp(self):
        """Set up test client and responses mock."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @responses.activate
    def test_newsapi_integration_success(self):
        """Test successful NewsAPI integration with mocked responses."""
        # Mock NewsAPI response
        mock_response = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": {"id": "techcrunch", "name": "TechCrunch"},
                    "author": "Tech Writer",
                    "title": "AI Breakthrough in Healthcare",
                    "description": "New AI system shows promise",
                    "url": "https://techcrunch.com/ai-healthcare",
                    "urlToImage": "https://example.com/image.jpg",
                    "publishedAt": "2024-01-15T10:00:00Z",
                    "content": "Full article content here..."
                },
                {
                    "source": {"id": "wired", "name": "Wired"},
                    "author": "Science Reporter",
                    "title": "Machine Learning Advances",
                    "description": "Latest ML developments",
                    "url": "https://wired.com/ml-advances",
                    "urlToImage": "https://example.com/image2.jpg",
                    "publishedAt": "2024-01-15T09:00:00Z",
                    "content": "Machine learning content..."
                }
            ]
        }
        
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            json=mock_response,
            status=200
        )
        
        # Mock article parsing
        with patch('curator.services.analyzer.batch_analyze') as mock_batch:
            mock_batch.return_value = []  # Empty results for simplicity
            
            response = self.client.post('/curate-topic',
                                      json={'topic': 'artificial intelligence'},
                                      content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            # Verify NewsAPI was called
            self.assertEqual(len(responses.calls), 1)
            self.assertIn('artificial intelligence', responses.calls[0].request.url)

    @responses.activate
    def test_newsapi_integration_rate_limit(self):
        """Test NewsAPI rate limit handling."""
        # Mock rate limit response
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            json={"status": "error", "code": "rateLimited", "message": "Rate limit exceeded"},
            status=429
        )
        
        response = self.client.post('/curate-topic',
                                  json={'topic': 'technology'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn('error', data)

    @responses.activate
    def test_newsapi_integration_api_error(self):
        """Test NewsAPI error response handling."""
        # Mock API error response
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            json={"status": "error", "code": "apiKeyInvalid", "message": "Invalid API key"},
            status=401
        )
        
        response = self.client.post('/curate-topic',
                                  json={'topic': 'science'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn('error', data)

    @responses.activate
    def test_newsapi_integration_network_error(self):
        """Test NewsAPI network error handling."""
        # Mock network error
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            body=Exception("Network error")
        )
        
        response = self.client.post('/curate-topic',
                                  json={'topic': 'technology'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)

    @responses.activate
    def test_article_parsing_integration(self):
        """Test article parsing with mocked HTTP responses."""
        # Mock article HTML response
        article_html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
        <article>
        <h1>Test Article Title</h1>
        <p>This is the article content with enough words to meet minimum requirements.
        It contains information about artificial intelligence and machine learning technologies.
        The content is substantial enough for proper analysis and scoring.</p>
        </article>
        </body>
        </html>
        """
        
        responses.add(
            responses.GET,
            "https://example.com/test-article",
            body=article_html,
            status=200,
            content_type='text/html'
        )
        
        # Mock validation and analysis
        with patch('curator.core.validation.validate_url') as mock_validate:
            with patch('curator.services.analyzer.analyze_article') as mock_analyze:
                mock_validate.return_value = (True, None)
                
                # Create a realistic scorecard
                article = Article(
                    url="https://example.com/test-article",
                    title="Test Article Title",
                    content="Article content for analysis",
                    publish_date=datetime.now()
                )
                scorecard = ScoreCard(
                    overall_score=80.0, readability_score=75.0, ner_density_score=70.0,
                    sentiment_score=65.0, tfidf_relevance_score=85.0, recency_score=90.0,
                    article=article
                )
                mock_analyze.return_value = scorecard
                
                response = self.client.post('/analyze',
                                          json={'url': 'https://example.com/test-article'},
                                          content_type='application/json')
                
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data['overall_score'], 80.0)


class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements and response times."""
    
    def setUp(self):
        """Set up test client."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_single_analysis_response_time(self, mock_validate, mock_analyze):
        """Test that single article analysis meets response time requirements."""
        mock_validate.return_value = (True, None)
        
        # Create realistic scorecard
        article = Article(
            url="https://example.com/article",
            title="Performance Test Article",
            content="Content for performance testing",
            publish_date=datetime.now()
        )
        scorecard = ScoreCard(
            overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
            sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
            article=article
        )
        mock_analyze.return_value = scorecard
        
        # Measure response time
        start_time = time.time()
        response = self.client.post('/analyze',
                                  json={'url': 'https://example.com/article'},
                                  content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        # Response should be under 3 seconds as per design requirements
        self.assertLess(response_time, 3.0, 
                       f"Response time {response_time:.2f}s exceeds 3s requirement")

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_topic_curation_response_time(self, mock_validate, mock_batch, mock_source):
        """Test that topic curation meets response time requirements."""
        mock_validate.return_value = (True, None)
        
        # Mock NewsSource
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = [
            f'https://example.com/article{i}' for i in range(5)
        ]
        mock_source.return_value = mock_instance
        
        # Mock batch analysis with realistic delay
        def mock_batch_analyze(*args, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return [
                ScoreCard(
                    overall_score=80.0, readability_score=75.0, ner_density_score=70.0,
                    sentiment_score=65.0, tfidf_relevance_score=85.0, recency_score=90.0,
                    article=Article(url=f"https://example.com/article{i}", 
                                  title=f"Article {i}", content="Content")
                ) for i in range(5)
            ]
        
        mock_batch.side_effect = mock_batch_analyze
        
        # Measure response time
        start_time = time.time()
        response = self.client.post('/curate-topic',
                                  json={'topic': 'technology', 'max_articles': 5},
                                  content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        # Should complete within reasonable time for 5 articles
        self.assertLess(response_time, 10.0,
                       f"Curation response time {response_time:.2f}s too high")

    def test_homepage_response_time(self):
        """Test homepage load time."""
        start_time = time.time()
        response = self.client.get('/')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        # Homepage should load very quickly
        self.assertLess(response_time, 1.0,
                       f"Homepage load time {response_time:.2f}s too high")

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_memory_usage_single_analysis(self, mock_validate, mock_analyze):
        """Test memory usage during single article analysis."""
        import psutil
        import os
        
        mock_validate.return_value = (True, None)
        
        # Create large content to test memory handling
        large_content = "Large article content. " * 1000
        article = Article(
            url="https://example.com/large-article",
            title="Large Article",
            content=large_content,
            publish_date=datetime.now()
        )
        scorecard = ScoreCard(
            overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
            sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
            article=article
        )
        mock_analyze.return_value = scorecard
        
        # Measure memory before and after
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        response = self.client.post('/analyze',
                                  json={'url': 'https://example.com/large-article'},
                                  content_type='application/json')
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        self.assertEqual(response.status_code, 200)
        # Memory increase should be reasonable (less than 100MB for single article)
        self.assertLess(memory_increase, 100,
                       f"Memory increase {memory_increase:.2f}MB too high")


class TestConcurrentUserScenarios(unittest.TestCase):
    """Test concurrent user scenarios and resource management."""
    
    def setUp(self):
        """Set up test client."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_concurrent_analysis_requests(self, mock_validate, mock_analyze):
        """Test handling of concurrent analysis requests."""
        mock_validate.return_value = (True, None)
        
        # Create different scorecards for each request
        def create_scorecard(url):
            article = Article(
                url=url,
                title=f"Article for {url}",
                content="Content for concurrent testing",
                publish_date=datetime.now()
            )
            return ScoreCard(
                overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
                sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
                article=article
            )
        
        mock_analyze.side_effect = lambda url, **kwargs: create_scorecard(url)
        
        def make_request(url):
            """Make a single analysis request."""
            return self.client.post('/analyze',
                                  json={'url': url},
                                  content_type='application/json')
        
        # Test concurrent requests
        urls = [f'https://example.com/article{i}' for i in range(10)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, url) for url in urls]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('overall_score', data)

    @patch('curator.services.news_source.NewsSource')
    @patch('curator.services.analyzer.batch_analyze')
    @patch('curator.core.validation.validate_topic_keywords')
    def test_concurrent_curation_requests(self, mock_validate, mock_batch, mock_source):
        """Test handling of concurrent curation requests."""
        mock_validate.return_value = (True, None)
        
        # Mock NewsSource
        mock_instance = Mock()
        mock_instance.get_article_urls.return_value = ['https://example.com/article1']
        mock_source.return_value = mock_instance
        
        # Mock batch analysis
        def create_results(topic):
            article = Article(
                url="https://example.com/article1",
                title=f"Article about {topic}",
                content="Content for testing",
                publish_date=datetime.now()
            )
            return [ScoreCard(
                overall_score=80.0, readability_score=75.0, ner_density_score=70.0,
                sentiment_score=65.0, tfidf_relevance_score=85.0, recency_score=90.0,
                article=article
            )]
        
        mock_batch.side_effect = lambda urls, query=None, **kwargs: create_results(query)
        
        def make_curation_request(topic):
            """Make a single curation request."""
            return self.client.post('/curate-topic',
                                  json={'topic': topic, 'max_articles': 5},
                                  content_type='application/json')
        
        # Test concurrent curation requests
        topics = ['technology', 'science', 'health', 'business', 'sports']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_curation_request, topic) for topic in topics]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('count', data)
            self.assertIn('results', data)

    def test_concurrent_mixed_requests(self):
        """Test handling of mixed concurrent requests (analysis + curation)."""
        def make_homepage_request():
            """Make homepage request."""
            return self.client.get('/')
        
        def make_compare_request():
            """Make compare page request."""
            return self.client.get('/compare')
        
        # Test mixed concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Submit various types of requests
            for i in range(5):
                futures.append(executor.submit(make_homepage_request))
                futures.append(executor.submit(make_compare_request))
            
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_resource_cleanup_after_requests(self, mock_validate, mock_analyze):
        """Test that resources are properly cleaned up after requests."""
        import gc
        import psutil
        import os
        
        mock_validate.return_value = (True, None)
        
        article = Article(
            url="https://example.com/cleanup-test",
            title="Cleanup Test Article",
            content="Content for cleanup testing",
            publish_date=datetime.now()
        )
        scorecard = ScoreCard(
            overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
            sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
            article=article
        )
        mock_analyze.return_value = scorecard
        
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple requests
        for i in range(20):
            response = self.client.post('/analyze',
                                      json={'url': f'https://example.com/article{i}'},
                                      content_type='application/json')
            self.assertEqual(response.status_code, 200)
        
        # Force garbage collection
        gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB for 20 requests)
        self.assertLess(memory_increase, 200,
                       f"Memory increase {memory_increase:.2f}MB suggests memory leak")

    def test_error_handling_under_load(self):
        """Test error handling under concurrent load."""
        def make_invalid_request():
            """Make an invalid request that should return 400."""
            return self.client.post('/analyze',
                                  json={'url': ''},  # Invalid empty URL
                                  content_type='application/json')
        
        # Test concurrent invalid requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_invalid_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should return 400 (not crash)
        for response in responses:
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn('error', data)

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_request_isolation(self, mock_validate, mock_analyze):
        """Test that concurrent requests don't interfere with each other."""
        mock_validate.return_value = (True, None)
        
        # Create unique responses for each URL
        def create_unique_scorecard(url, **kwargs):
            # Extract number from URL for unique scoring
            url_num = int(url.split('article')[1]) if 'article' in url else 0
            article = Article(
                url=url,
                title=f"Article {url_num}",
                content=f"Content for article {url_num}",
                publish_date=datetime.now()
            )
            return ScoreCard(
                overall_score=50.0 + url_num,  # Unique score based on URL
                readability_score=70.0, ner_density_score=65.0,
                sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
                article=article
            )
        
        mock_analyze.side_effect = create_unique_scorecard
        
        def make_request_and_verify(url_num):
            """Make request and verify response matches expected URL."""
            url = f'https://example.com/article{url_num}'
            response = self.client.post('/analyze',
                                      json={'url': url},
                                      content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            
            # Verify response matches the specific request
            expected_score = 50.0 + url_num
            self.assertEqual(data['overall_score'], expected_score)
            self.assertEqual(data['article']['title'], f"Article {url_num}")
            
            return response
        
        # Test concurrent requests with different URLs
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request_and_verify, i) for i in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should have succeeded with correct isolation
        self.assertEqual(len(responses), 10)


class TestResourceManagement(unittest.TestCase):
    """Test resource management and system limits."""
    
    def setUp(self):
        """Set up test client."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_large_request_handling(self):
        """Test handling of large request payloads."""
        # Test with very long topic string
        long_topic = "artificial intelligence " * 100  # Very long topic
        
        response = self.client.post('/curate-topic',
                                  json={'topic': long_topic},
                                  content_type='application/json')
        
        # Should handle gracefully (either process or reject appropriately)
        self.assertIn(response.status_code, [200, 400, 413])  # OK, Bad Request, or Payload Too Large

    @patch('curator.services.analyzer.analyze_article')
    @patch('curator.core.validation.validate_url')
    def test_timeout_handling(self, mock_validate, mock_analyze):
        """Test handling of slow operations."""
        mock_validate.return_value = (True, None)
        
        # Mock slow analysis
        def slow_analysis(*args, **kwargs):
            time.sleep(2)  # Simulate slow operation
            article = Article(
                url="https://example.com/slow-article",
                title="Slow Article",
                content="Content",
                publish_date=datetime.now()
            )
            return ScoreCard(
                overall_score=75.0, readability_score=70.0, ner_density_score=65.0,
                sentiment_score=60.0, tfidf_relevance_score=80.0, recency_score=85.0,
                article=article
            )
        
        mock_analyze.side_effect = slow_analysis
        
        start_time = time.time()
        response = self.client.post('/analyze',
                                  json={'url': 'https://example.com/slow-article'},
                                  content_type='application/json')
        end_time = time.time()
        
        # Should complete (may be slow but shouldn't timeout in test environment)
        self.assertEqual(response.status_code, 200)
        
        # Verify it actually took time (confirming our mock worked)
        self.assertGreater(end_time - start_time, 1.5)

    def test_invalid_content_type_handling(self):
        """Test handling of invalid content types."""
        # Test with invalid content type
        response = self.client.post('/analyze',
                                  data='invalid data',
                                  content_type='text/plain')
        
        # Should handle gracefully
        self.assertIn(response.status_code, [400, 415])  # Bad Request or Unsupported Media Type

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON requests."""
        response = self.client.post('/analyze',
                                  data='{"invalid": json}',
                                  content_type='application/json')
        
        # Should handle malformed JSON gracefully
        self.assertIn(response.status_code, [400, 500])

    @patch('curator.services.news_source.NewsSource')
    def test_external_service_failure_resilience(self, mock_source):
        """Test resilience to external service failures."""
        # Mock NewsSource to fail
        mock_instance = Mock()
        mock_instance.get_article_urls.side_effect = Exception("NewsAPI unavailable")
        mock_source.return_value = mock_instance
        
        response = self.client.post('/curate-topic',
                                  json={'topic': 'technology'},
                                  content_type='application/json')
        
        # Should handle external service failure gracefully
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn('error', data)


if __name__ == '__main__':
    # Add responses library to requirements if not present
    try:
        import responses
    except ImportError:
        print("Warning: 'responses' library not found. Install with: pip install responses")
        print("Some tests may be skipped.")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)