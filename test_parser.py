"""
Unit tests for the article parsing module.

Tests cover parsing functionality, error handling, retry logic,
content validation, and batch processing with mocked responses.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests
from newspaper import Article as NewspaperArticle

from parser import (
    parse_article, batch_parse_articles, validate_content,
    parse_article_with_config, get_article_word_count, is_article_recent,
    ArticleParsingError, ContentValidationError, USER_AGENTS
)
from models import Article


class TestArticleParser(unittest.TestCase):
    """Test cases for article parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_url = "https://example.com/test-article"
        self.test_content = "This is a test article content. " * 100  # 600 words
        self.test_title = "Test Article Title"
        self.test_author = "Test Author"
        self.test_date = datetime.now() - timedelta(hours=2)
        
    def create_mock_newspaper_article(self, content=None, title=None, 
                                    authors=None, publish_date=None, 
                                    html="<html>content</html>"):
        """Create a mock newspaper article object."""
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.text = content or self.test_content
        mock_article.title = title or self.test_title
        mock_article.authors = authors or [self.test_author]
        mock_article.publish_date = publish_date or self.test_date
        mock_article.summary = "Test summary"
        mock_article.html = html
        return mock_article

    @patch('parser.NewspaperArticle')
    def test_parse_article_success(self, mock_newspaper_class):
        """Test successful article parsing."""
        # Setup mock
        mock_article = self.create_mock_newspaper_article()
        mock_newspaper_class.return_value = mock_article
        
        # Parse article
        result = parse_article(self.test_url)
        
        # Verify result
        self.assertIsInstance(result, Article)
        self.assertEqual(result.url, self.test_url)
        self.assertEqual(result.title, self.test_title)
        self.assertEqual(result.author, self.test_author)
        self.assertEqual(result.content, self.test_content)
        self.assertEqual(result.publish_date, self.test_date)
        
        # Verify newspaper3k was called correctly
        mock_newspaper_class.assert_called_once()
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()

    def test_parse_article_empty_url(self):
        """Test parsing with empty URL raises error."""
        with self.assertRaises(ArticleParsingError) as context:
            parse_article("")
        
        self.assertIn("URL cannot be empty", str(context.exception))

    @patch('parser.NewspaperArticle')
    def test_parse_article_download_failure(self, mock_newspaper_class):
        """Test handling of download failures."""
        # Setup mock to simulate download failure
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.html = None  # Simulate failed download
        mock_newspaper_class.return_value = mock_article
        
        # Should raise error after all retries
        with self.assertRaises(ArticleParsingError) as context:
            parse_article(self.test_url, max_retries=2)
        
        self.assertIn("Failed to parse article after 2 attempts", str(context.exception))

    @patch('parser.NewspaperArticle')
    def test_parse_article_network_timeout(self, mock_newspaper_class):
        """Test handling of network timeouts."""
        # Setup mock to raise timeout
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.download.side_effect = requests.exceptions.Timeout("Timeout")
        mock_newspaper_class.return_value = mock_article
        
        # Should retry and eventually fail
        with self.assertRaises(ArticleParsingError) as context:
            parse_article(self.test_url, max_retries=2)
        
        self.assertIn("Failed to parse article after 2 attempts", str(context.exception))

    @patch('parser.NewspaperArticle')
    def test_parse_article_retry_success(self, mock_newspaper_class):
        """Test successful parsing after initial failure."""
        # Setup mock to fail first, succeed second
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.download.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            None  # Success on second attempt
        ]
        mock_article.html = "<html>content</html>"
        mock_article.text = self.test_content
        mock_article.title = self.test_title
        mock_article.authors = [self.test_author]
        mock_article.publish_date = self.test_date
        mock_article.summary = "Test summary"
        
        mock_newspaper_class.return_value = mock_article
        
        # Should succeed on retry
        result = parse_article(self.test_url, max_retries=3)
        
        self.assertIsInstance(result, Article)
        self.assertEqual(result.title, self.test_title)
        
        # Verify multiple attempts were made
        self.assertEqual(mock_article.download.call_count, 2)

    def test_validate_content_success(self):
        """Test successful content validation."""
        mock_article = self.create_mock_newspaper_article()
        
        # Should not raise any exception
        validate_content(mock_article, min_word_count=300)

    def test_validate_content_empty_text(self):
        """Test validation failure for empty content."""
        mock_article = self.create_mock_newspaper_article(content="")
        
        with self.assertRaises(ContentValidationError) as context:
            validate_content(mock_article)
        
        self.assertIn("Article content is empty", str(context.exception))

    def test_validate_content_too_short(self):
        """Test validation failure for content too short."""
        short_content = "Short content"  # Only 2 words
        mock_article = self.create_mock_newspaper_article(content=short_content)
        
        with self.assertRaises(ContentValidationError) as context:
            validate_content(mock_article, min_word_count=300)
        
        self.assertIn("Article too short", str(context.exception))

    def test_validate_content_missing_title(self):
        """Test validation failure for missing title."""
        mock_article = self.create_mock_newspaper_article(title="")
        
        with self.assertRaises(ContentValidationError) as context:
            validate_content(mock_article)
        
        self.assertIn("Article title is missing", str(context.exception))

    @patch('parser.parse_article')
    def test_batch_parse_articles_success(self, mock_parse):
        """Test successful batch parsing of multiple articles."""
        # Setup mock to return different articles
        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
        mock_articles = [
            Article(url=urls[0], title="Article 1", content="Content 1"),
            Article(url=urls[1], title="Article 2", content="Content 2"),
            Article(url=urls[2], title="Article 3", content="Content 3")
        ]
        mock_parse.side_effect = mock_articles
        
        # Parse batch
        results = batch_parse_articles(urls)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].title, "Article 1")
        self.assertEqual(results[1].title, "Article 2")
        self.assertEqual(results[2].title, "Article 3")
        
        # Verify parse_article was called for each URL
        self.assertEqual(mock_parse.call_count, 3)

    @patch('parser.parse_article')
    def test_batch_parse_articles_partial_failure(self, mock_parse):
        """Test batch parsing with some failures."""
        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
        
        # Setup mock to fail on second article
        mock_parse.side_effect = [
            Article(url=urls[0], title="Article 1", content="Content 1"),
            ArticleParsingError("Failed to parse"),
            Article(url=urls[2], title="Article 3", content="Content 3")
        ]
        
        # Parse batch
        results = batch_parse_articles(urls)
        
        # Should return only successful articles
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Article 1")
        self.assertEqual(results[1].title, "Article 3")

    def test_batch_parse_articles_empty_list(self):
        """Test batch parsing with empty URL list."""
        results = batch_parse_articles([])
        self.assertEqual(results, [])

    def test_get_article_word_count(self):
        """Test word count calculation."""
        article = Article(url="test", content="This is a test article with ten words total")
        count = get_article_word_count(article)
        self.assertEqual(count, 10)

    def test_get_article_word_count_empty(self):
        """Test word count for empty content."""
        article = Article(url="test", content="")
        count = get_article_word_count(article)
        self.assertEqual(count, 0)

    def test_is_article_recent_with_date(self):
        """Test recency check with publication date."""
        recent_date = datetime.now() - timedelta(days=5)
        old_date = datetime.now() - timedelta(days=50)
        
        recent_article = Article(url="test", publish_date=recent_date)
        old_article = Article(url="test", publish_date=old_date)
        
        self.assertTrue(is_article_recent(recent_article, max_age_days=30))
        self.assertFalse(is_article_recent(old_article, max_age_days=30))

    def test_is_article_recent_no_date(self):
        """Test recency check without publication date."""
        article = Article(url="test", publish_date=None)
        
        # Should return True when no date is available
        self.assertTrue(is_article_recent(article))

    @patch('parser.NewspaperArticle')
    def test_parse_article_with_config_success(self, mock_newspaper_class):
        """Test parsing with specific configuration."""
        mock_article = self.create_mock_newspaper_article()
        mock_newspaper_class.return_value = mock_article
        
        user_agent = USER_AGENTS[0]
        result = parse_article_with_config(self.test_url, user_agent, timeout=30)
        
        self.assertEqual(result, mock_article)
        mock_newspaper_class.assert_called_once()
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()

    @patch('parser.NewspaperArticle')
    def test_parse_article_with_config_timeout(self, mock_newspaper_class):
        """Test parsing with timeout error."""
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.download.side_effect = requests.exceptions.Timeout("Timeout")
        mock_newspaper_class.return_value = mock_article
        
        with self.assertRaises(ArticleParsingError) as context:
            parse_article_with_config(self.test_url, USER_AGENTS[0])
        
        self.assertIn("Request timeout", str(context.exception))

    @patch('parser.NewspaperArticle')
    def test_parse_article_with_config_connection_error(self, mock_newspaper_class):
        """Test parsing with connection error."""
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.download.side_effect = requests.exceptions.ConnectionError("Connection failed")
        mock_newspaper_class.return_value = mock_article
        
        with self.assertRaises(ArticleParsingError) as context:
            parse_article_with_config(self.test_url, USER_AGENTS[0])
        
        self.assertIn("Network connection error", str(context.exception))

    @patch('parser.time.sleep')
    @patch('parser.NewspaperArticle')
    def test_parse_article_retry_delay(self, mock_newspaper_class, mock_sleep):
        """Test that retry delay is applied between attempts."""
        # Setup mock to fail twice, succeed third time
        mock_article = Mock(spec=NewspaperArticle)
        mock_article.download.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.ConnectionError("Connection failed"),
            None  # Success on third attempt
        ]
        mock_article.html = "<html>content</html>"
        mock_article.text = self.test_content
        mock_article.title = self.test_title
        mock_article.authors = [self.test_author]
        mock_article.publish_date = self.test_date
        mock_article.summary = "Test summary"
        
        mock_newspaper_class.return_value = mock_article
        
        # Parse with retry
        result = parse_article(self.test_url, max_retries=3, retry_delay=1.0)
        
        # Verify sleep was called between retries
        self.assertEqual(mock_sleep.call_count, 2)  # Two delays for three attempts
        
        # Verify exponential backoff
        mock_sleep.assert_any_call(1.0)  # First delay
        mock_sleep.assert_any_call(1.5)  # Second delay (1.0 * 1.5)


class TestArticleParserIntegration(unittest.TestCase):
    """Integration tests for article parser with real-world scenarios."""
    
    def test_user_agents_list(self):
        """Test that user agents list is properly configured."""
        self.assertIsInstance(USER_AGENTS, list)
        self.assertGreater(len(USER_AGENTS), 0)
        
        # Verify all user agents are strings
        for ua in USER_AGENTS:
            self.assertIsInstance(ua, str)
            self.assertGreater(len(ua), 0)

    def test_article_model_integration(self):
        """Test integration with Article model."""
        # Create article using model
        article = Article(
            url="https://example.com/test",
            title="Test Title",
            author="Test Author",
            content="Test content with enough words to pass validation " * 50
        )
        
        # Verify model properties
        self.assertEqual(article.url, "https://example.com/test")
        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.author, "Test Author")
        self.assertGreater(len(article.content), 300)  # Should pass word count validation


if __name__ == '__main__':
    unittest.main()