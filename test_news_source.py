"""
Unit tests for NewsSource class.

Tests NewsAPI integration with mocked responses to verify
error handling, rate limiting, and retry logic.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

from news_source import NewsSource, NewsAPIError, RateLimitError


class TestNewsSource(unittest.TestCase):
    """Test cases for NewsSource class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.news_source = NewsSource(api_key=self.api_key)
        
        # Sample valid article data
        self.sample_article = {
            "title": "Test Article Title",
            "description": "Test article description with content",
            "url": "https://example.com/article",
            "publishedAt": "2023-12-01T10:00:00Z",
            "author": "Test Author",
            "content": "Test article content..."
        }
        
        # Sample API response
        self.sample_response = {
            "status": "ok",
            "totalResults": 1,
            "articles": [self.sample_article]
        }
    
    def test_init_with_api_key(self):
        """Test NewsSource initialization with API key."""
        news_source = NewsSource(api_key="test-key")
        self.assertEqual(news_source.api_key, "test-key")
        self.assertIsNotNone(news_source.session)
        self.assertEqual(news_source.session.headers['X-API-Key'], "test-key")
    
    @patch.dict('os.environ', {'NEWS_API_KEY': 'env-api-key'})
    def test_init_with_env_api_key(self):
        """Test NewsSource initialization with environment variable."""
        news_source = NewsSource()
        self.assertEqual(news_source.api_key, "env-api-key")
    
    def test_init_without_api_key(self):
        """Test NewsSource initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(NewsAPIError) as context:
                NewsSource()
            self.assertIn("NewsAPI key is required", str(context.exception))
    
    @patch('time.sleep')
    def test_rate_limiting(self, mock_sleep):
        """Test rate limiting functionality."""
        # Set last request time to recent
        self.news_source.last_request_time = time.time() - 0.5
        
        # Call rate limiting method
        self.news_source._handle_rate_limiting()
        
        # Verify sleep was called
        mock_sleep.assert_called_once()
        # Verify sleep time is reasonable (0.5s + jitter)
        sleep_time = mock_sleep.call_args[0][0]
        self.assertGreater(sleep_time, 0.5)
        self.assertLess(sleep_time, 1.0)
    
    def test_validate_response_success(self):
        """Test successful response validation."""
        # Should not raise exception
        self.news_source._validate_response(self.sample_response)
    
    def test_validate_response_invalid_status(self):
        """Test response validation with error status."""
        error_response = {
            "status": "error",
            "code": "apiKeyInvalid",
            "message": "Your API key is invalid"
        }
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._validate_response(error_response)
        self.assertIn("apiKeyInvalid", str(context.exception))
    
    def test_validate_response_missing_articles(self):
        """Test response validation with missing articles field."""
        invalid_response = {"status": "ok"}
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._validate_response(invalid_response)
        self.assertIn("missing articles field", str(context.exception))
    
    def test_validate_response_invalid_articles_type(self):
        """Test response validation with invalid articles type."""
        invalid_response = {
            "status": "ok",
            "articles": "not a list"
        }
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._validate_response(invalid_response)
        self.assertIn("articles must be a list", str(context.exception))
    
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_response
        mock_get.return_value = mock_response
        
        result = self.news_source._make_request('everything', {'q': 'test'})
        
        self.assertEqual(result, self.sample_response)
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_make_request_rate_limit(self, mock_get):
        """Test API request with rate limit error."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response
        
        with self.assertRaises(RateLimitError) as context:
            self.news_source._make_request('everything', {'q': 'test'})
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    @patch('requests.Session.get')
    def test_make_request_timeout(self, mock_get):
        """Test API request with timeout error."""
        mock_get.side_effect = Timeout("Request timeout")
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._make_request('everything', {'q': 'test'})
        self.assertIn("Request timeout", str(context.exception))
    
    @patch('requests.Session.get')
    def test_make_request_connection_error(self, mock_get):
        """Test API request with connection error."""
        mock_get.side_effect = ConnectionError("Connection failed")
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._make_request('everything', {'q': 'test'})
        self.assertIn("Connection error", str(context.exception))
    
    @patch('requests.Session.get')
    def test_make_request_http_error_401(self, mock_get):
        """Test API request with 401 HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._make_request('everything', {'q': 'test'})
        self.assertIn("Invalid API key", str(context.exception))
    
    @patch('requests.Session.get')
    def test_make_request_http_error_400(self, mock_get):
        """Test API request with 400 HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source._make_request('everything', {'q': 'test'})
        self.assertIn("Invalid request parameters", str(context.exception))
    
    def test_is_valid_article_success(self):
        """Test valid article validation."""
        self.assertTrue(self.news_source._is_valid_article(self.sample_article))
    
    def test_is_valid_article_missing_title(self):
        """Test article validation with missing title."""
        invalid_article = self.sample_article.copy()
        del invalid_article['title']
        
        self.assertFalse(self.news_source._is_valid_article(invalid_article))
    
    def test_is_valid_article_empty_url(self):
        """Test article validation with empty URL."""
        invalid_article = self.sample_article.copy()
        invalid_article['url'] = ""
        
        self.assertFalse(self.news_source._is_valid_article(invalid_article))
    
    def test_is_valid_article_invalid_url_format(self):
        """Test article validation with invalid URL format."""
        invalid_article = self.sample_article.copy()
        invalid_article['url'] = "not-a-valid-url"
        
        self.assertFalse(self.news_source._is_valid_article(invalid_article))
    
    def test_is_valid_article_no_content(self):
        """Test article validation with no content."""
        invalid_article = self.sample_article.copy()
        invalid_article['description'] = ""
        invalid_article['content'] = ""
        
        self.assertFalse(self.news_source._is_valid_article(invalid_article))
    
    @patch.object(NewsSource, '_make_request')
    def test_fetch_articles_by_topic_success(self, mock_make_request):
        """Test successful article fetching by topic."""
        mock_make_request.return_value = self.sample_response
        
        articles = self.news_source.fetch_articles_by_topic("technology")
        
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0], self.sample_article)
        mock_make_request.assert_called_once()
    
    def test_fetch_articles_by_topic_empty_topic(self):
        """Test article fetching with empty topic."""
        with self.assertRaises(ValueError) as context:
            self.news_source.fetch_articles_by_topic("")
        self.assertIn("Topic cannot be empty", str(context.exception))
    
    def test_fetch_articles_by_topic_invalid_page_size(self):
        """Test article fetching with invalid page size."""
        with self.assertRaises(ValueError) as context:
            self.news_source.fetch_articles_by_topic("test", page_size=0)
        self.assertIn("Page size must be between 1 and 100", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.news_source.fetch_articles_by_topic("test", page_size=101)
        self.assertIn("Page size must be between 1 and 100", str(context.exception))
    
    def test_fetch_articles_by_topic_invalid_sort_by(self):
        """Test article fetching with invalid sort_by parameter."""
        with self.assertRaises(ValueError) as context:
            self.news_source.fetch_articles_by_topic("test", sort_by="invalid")
        self.assertIn("sort_by must be one of", str(context.exception))
    
    @patch.object(NewsSource, '_make_request')
    def test_fetch_articles_by_topic_with_dates(self, mock_make_request):
        """Test article fetching with date parameters."""
        mock_make_request.return_value = self.sample_response
        
        from_date = datetime(2023, 12, 1)
        to_date = datetime(2023, 12, 31)
        
        self.news_source.fetch_articles_by_topic(
            "test",
            from_date=from_date,
            to_date=to_date
        )
        
        # Verify date parameters were passed correctly
        call_args = mock_make_request.call_args[0][1]  # Get params dict
        self.assertEqual(call_args['from'], '2023-12-01')
        self.assertEqual(call_args['to'], '2023-12-31')
    
    @patch.object(NewsSource, '_make_request')
    def test_fetch_articles_filters_invalid_articles(self, mock_make_request):
        """Test that invalid articles are filtered out."""
        # Response with mix of valid and invalid articles
        response_with_invalid = {
            "status": "ok",
            "totalResults": 3,
            "articles": [
                self.sample_article,  # Valid
                {"title": "", "url": "https://example.com", "publishedAt": "2023-12-01T10:00:00Z"},  # Invalid: empty title
                {"title": "Valid Title", "url": "invalid-url", "publishedAt": "2023-12-01T10:00:00Z"}  # Invalid: bad URL
            ]
        }
        
        mock_make_request.return_value = response_with_invalid
        
        articles = self.news_source.fetch_articles_by_topic("test")
        
        # Should only return the valid article
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0], self.sample_article)
    
    @patch.object(NewsSource, 'fetch_articles_by_topic')
    def test_get_article_urls_success(self, mock_fetch):
        """Test successful URL extraction."""
        mock_fetch.return_value = [self.sample_article]
        
        urls = self.news_source.get_article_urls("technology", max_articles=5)
        
        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0], "https://example.com/article")
        mock_fetch.assert_called_once_with(
            topic="technology",
            page_size=5,
            sort_by="relevancy"
        )
    
    def test_get_article_urls_invalid_max_articles(self):
        """Test URL extraction with invalid max_articles."""
        with self.assertRaises(ValueError) as context:
            self.news_source.get_article_urls("test", max_articles=0)
        self.assertIn("max_articles must be positive", str(context.exception))
    
    @patch('requests.Session.get')
    def test_check_api_status_success(self, mock_get):
        """Test successful API status check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'X-RateLimit-Remaining': '1000',
            'X-RateLimit-Reset': '1640995200'
        }
        mock_get.return_value = mock_response
        
        status = self.news_source.check_api_status()
        
        self.assertEqual(status['status_code'], 200)
        self.assertEqual(status['rate_limit_remaining'], '1000')
        self.assertTrue(status['api_accessible'])
    
    @patch('requests.Session.get')
    def test_check_api_status_failure(self, mock_get):
        """Test API status check with failure."""
        mock_get.side_effect = RequestException("Connection failed")
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source.check_api_status()
        self.assertIn("Status check failed", str(context.exception))
    
    @patch.object(NewsSource, '_make_request')
    def test_rate_limit_error_propagation(self, mock_make_request):
        """Test that RateLimitError is properly propagated."""
        mock_make_request.side_effect = RateLimitError("Rate limit exceeded")
        
        with self.assertRaises(RateLimitError):
            self.news_source.fetch_articles_by_topic("test")
    
    @patch.object(NewsSource, '_make_request')
    def test_unexpected_error_handling(self, mock_make_request):
        """Test handling of unexpected errors."""
        mock_make_request.side_effect = Exception("Unexpected error")
        
        with self.assertRaises(NewsAPIError) as context:
            self.news_source.fetch_articles_by_topic("test")
        self.assertIn("Unexpected error", str(context.exception))


class TestNewsAPIErrorClasses(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_news_api_error(self):
        """Test NewsAPIError exception."""
        error = NewsAPIError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError("Rate limit exceeded")
        self.assertEqual(str(error), "Rate limit exceeded")
        self.assertIsInstance(error, NewsAPIError)
        self.assertIsInstance(error, Exception)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)