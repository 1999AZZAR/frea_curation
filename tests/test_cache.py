"""
Unit tests for Redis caching functionality.

Tests cache key generation, serialization/deserialization, and cache operations
with graceful degradation when Redis is unavailable.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from curator.core.cache import (
    generate_cache_key,
    serialize_article,
    deserialize_article,
    serialize_scorecard,
    deserialize_scorecard,
    CacheManager,
    get_cached_article,
    cache_article,
    get_cached_scorecard,
    cache_scorecard,
)
from curator.core.models import Article, ScoreCard, Entity


class TestCacheKeyGeneration:
    """Test cache key generation functionality."""
    
    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        url = "https://example.com/article"
        key = generate_cache_key(url)
        
        # Should be a SHA-256 hash (64 characters)
        assert len(key) == 64
        assert isinstance(key, str)
    
    def test_generate_cache_key_with_prefix(self):
        """Test cache key generation with prefix."""
        url = "https://example.com/article"
        prefix = "article"
        key = generate_cache_key(url, prefix)
        
        assert key.startswith(f"{prefix}:")
        assert len(key) == len(prefix) + 1 + 64  # prefix + : + hash
    
    def test_generate_cache_key_normalization(self):
        """Test URL normalization in cache key generation."""
        url1 = "https://example.com/article"
        url2 = "HTTPS://EXAMPLE.COM/ARTICLE"
        url3 = "  https://example.com/article  "
        
        key1 = generate_cache_key(url1)
        key2 = generate_cache_key(url2)
        key3 = generate_cache_key(url3)
        
        # All should generate the same key due to normalization
        assert key1 == key2 == key3


class TestSerialization:
    """Test serialization and deserialization functionality."""
    
    def test_serialize_deserialize_article(self):
        """Test article serialization and deserialization."""
        # Create test article with entities
        entities = [
            Entity(text="OpenAI", label="ORG", confidence=0.95),
            Entity(text="San Francisco", label="GPE", confidence=0.88)
        ]
        
        article = Article(
            url="https://example.com/article",
            title="Test Article",
            author="Test Author",
            publish_date=datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            content="This is test content about OpenAI in San Francisco.",
            summary="Test summary",
            entities=entities
        )
        
        # Serialize and deserialize
        serialized = serialize_article(article)
        deserialized = deserialize_article(serialized)
        
        # Verify all fields are preserved
        assert deserialized.url == article.url
        assert deserialized.title == article.title
        assert deserialized.author == article.author
        assert deserialized.publish_date == article.publish_date
        assert deserialized.content == article.content
        assert deserialized.summary == article.summary
        assert len(deserialized.entities) == len(article.entities)
        
        # Verify entities are preserved
        for orig, deser in zip(article.entities, deserialized.entities):
            assert deser.text == orig.text
            assert deser.label == orig.label
            assert deser.confidence == orig.confidence
    
    def test_serialize_deserialize_article_no_date(self):
        """Test article serialization with no publish date."""
        article = Article(
            url="https://example.com/article",
            title="Test Article",
            content="Test content"
        )
        
        serialized = serialize_article(article)
        deserialized = deserialize_article(serialized)
        
        assert deserialized.url == article.url
        assert deserialized.title == article.title
        assert deserialized.publish_date is None
    
    def test_serialize_deserialize_scorecard(self):
        """Test scorecard serialization and deserialization."""
        # Create test article
        article = Article(
            url="https://example.com/article",
            title="Test Article",
            content="Test content",
            publish_date=datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        # Create test scorecard
        scorecard = ScoreCard(
            overall_score=85.5,
            readability_score=90.0,
            ner_density_score=75.0,
            sentiment_score=80.0,
            tfidf_relevance_score=88.0,
            recency_score=95.0,
            reputation_score=70.0,
            topic_coherence_score=82.0,
            article=article
        )
        
        # Serialize and deserialize
        serialized = serialize_scorecard(scorecard)
        deserialized = deserialize_scorecard(serialized)
        
        # Verify all scores are preserved
        assert deserialized.overall_score == scorecard.overall_score
        assert deserialized.readability_score == scorecard.readability_score
        assert deserialized.ner_density_score == scorecard.ner_density_score
        assert deserialized.sentiment_score == scorecard.sentiment_score
        assert deserialized.tfidf_relevance_score == scorecard.tfidf_relevance_score
        assert deserialized.recency_score == scorecard.recency_score
        assert deserialized.reputation_score == scorecard.reputation_score
        assert deserialized.topic_coherence_score == scorecard.topic_coherence_score
        
        # Verify article is preserved
        assert deserialized.article.url == article.url
        assert deserialized.article.title == article.title
        assert deserialized.article.publish_date == article.publish_date


class TestCacheManager:
    """Test CacheManager functionality."""
    
    @patch('curator.core.cache.get_redis_client')
    def test_cache_manager_redis_unavailable(self, mock_get_redis):
        """Test cache manager when Redis is unavailable."""
        mock_get_redis.return_value = None
        
        cache_manager = CacheManager()
        
        assert not cache_manager.is_available()
        
        # All operations should return None/False gracefully
        article = Article(url="https://example.com/test", title="Test")
        
        assert cache_manager.get_article("https://example.com/test") is None
        assert not cache_manager.set_article("https://example.com/test", article)
        assert cache_manager.get_scorecard("https://example.com/test") is None
    
    @patch('curator.core.cache.get_redis_client')
    def test_cache_manager_redis_available(self, mock_get_redis):
        """Test cache manager when Redis is available."""
        # Mock Redis client
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis
        
        cache_manager = CacheManager()
        
        assert cache_manager.is_available()
        assert cache_manager.client == mock_redis
    
    @patch('curator.core.cache.get_redis_client')
    def test_article_caching(self, mock_get_redis):
        """Test article caching operations."""
        # Mock Redis client
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis
        
        cache_manager = CacheManager()
        
        # Test article
        article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="Test content"
        )
        
        # Test cache miss
        mock_redis.get.return_value = None
        result = cache_manager.get_article("https://example.com/test")
        assert result is None
        
        # Test cache set
        mock_redis.setex.return_value = True
        success = cache_manager.set_article("https://example.com/test", article)
        assert success
        
        # Verify setex was called with correct parameters
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0].startswith("article:")  # key with prefix
        assert args[1] == cache_manager.article_ttl  # TTL
        
        # Test cache hit
        serialized_article = serialize_article(article)
        mock_redis.get.return_value = serialized_article
        result = cache_manager.get_article("https://example.com/test")
        
        assert result is not None
        assert result.url == article.url
        assert result.title == article.title
    
    @patch('curator.core.cache.get_redis_client')
    def test_scorecard_caching(self, mock_get_redis):
        """Test scorecard caching operations."""
        # Mock Redis client
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis
        
        cache_manager = CacheManager()
        
        # Test scorecard
        article = Article(url="https://example.com/test", title="Test")
        scorecard = ScoreCard(
            overall_score=85.0,
            readability_score=90.0,
            ner_density_score=75.0,
            sentiment_score=80.0,
            tfidf_relevance_score=88.0,
            recency_score=95.0,
            reputation_score=70.0,
            topic_coherence_score=82.0,
            article=article
        )
        
        # Test cache miss
        mock_redis.get.return_value = None
        result = cache_manager.get_scorecard("https://example.com/test", "test query")
        assert result is None
        
        # Test cache set
        mock_redis.setex.return_value = True
        success = cache_manager.set_scorecard("https://example.com/test", scorecard, "test query")
        assert success
        
        # Test cache hit
        serialized_scorecard = serialize_scorecard(scorecard)
        mock_redis.get.return_value = serialized_scorecard
        result = cache_manager.get_scorecard("https://example.com/test", "test query")
        
        assert result is not None
        assert result.overall_score == scorecard.overall_score
        assert result.article.url == article.url
    
    @patch('curator.core.cache.get_redis_client')
    def test_cache_stats(self, mock_get_redis):
        """Test cache statistics functionality."""
        # Mock Redis client
        mock_redis = Mock()
        mock_get_redis.return_value = mock_redis
        
        # Mock Redis info and keys responses
        mock_redis.info.return_value = {
            'used_memory_human': '1.5M',
            'connected_clients': 2,
            'uptime_in_seconds': 3600
        }
        mock_redis.keys.side_effect = [
            ['article:key1', 'article:key2'],  # article keys
            ['scorecard:key1']  # scorecard keys
        ]
        
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        
        assert stats['available'] is True
        assert stats['article_count'] == 2
        assert stats['scorecard_count'] == 1
        assert stats['total_keys'] == 3
        assert stats['memory_usage'] == '1.5M'
        assert stats['connected_clients'] == 2
        assert stats['uptime_seconds'] == 3600
    
    @patch('curator.core.cache.get_redis_client')
    def test_cache_stats_unavailable(self, mock_get_redis):
        """Test cache statistics when Redis is unavailable."""
        mock_get_redis.return_value = None
        
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        
        assert stats['available'] is False
        assert 'error' in stats


class TestConvenienceFunctions:
    """Test convenience functions for caching."""
    
    @patch('curator.core.cache.cache_manager')
    def test_get_cached_article(self, mock_cache_manager):
        """Test get_cached_article convenience function."""
        article = Article(url="https://example.com/test", title="Test")
        mock_cache_manager.get_article.return_value = article
        
        result = get_cached_article("https://example.com/test")
        
        assert result == article
        mock_cache_manager.get_article.assert_called_once_with("https://example.com/test")
    
    @patch('curator.core.cache.cache_manager')
    def test_cache_article_function(self, mock_cache_manager):
        """Test cache_article convenience function."""
        article = Article(url="https://example.com/test", title="Test")
        mock_cache_manager.set_article.return_value = True
        
        result = cache_article("https://example.com/test", article)
        
        assert result is True
        mock_cache_manager.set_article.assert_called_once_with("https://example.com/test", article, None)
    
    @patch('curator.core.cache.cache_manager')
    def test_get_cached_scorecard(self, mock_cache_manager):
        """Test get_cached_scorecard convenience function."""
        article = Article(url="https://example.com/test", title="Test")
        scorecard = ScoreCard(
            overall_score=85.0, readability_score=90.0, ner_density_score=75.0,
            sentiment_score=80.0, tfidf_relevance_score=88.0, recency_score=95.0,
            reputation_score=70.0, topic_coherence_score=82.0, article=article
        )
        mock_cache_manager.get_scorecard.return_value = scorecard
        
        result = get_cached_scorecard("https://example.com/test", "query")
        
        assert result == scorecard
        mock_cache_manager.get_scorecard.assert_called_once_with("https://example.com/test", "query")
    
    @patch('curator.core.cache.cache_manager')
    def test_cache_scorecard_function(self, mock_cache_manager):
        """Test cache_scorecard convenience function."""
        article = Article(url="https://example.com/test", title="Test")
        scorecard = ScoreCard(
            overall_score=85.0, readability_score=90.0, ner_density_score=75.0,
            sentiment_score=80.0, tfidf_relevance_score=88.0, recency_score=95.0,
            reputation_score=70.0, topic_coherence_score=82.0, article=article
        )
        mock_cache_manager.set_scorecard.return_value = True
        
        result = cache_scorecard("https://example.com/test", scorecard, "query")
        
        assert result is True
        mock_cache_manager.set_scorecard.assert_called_once_with("https://example.com/test", scorecard, "query", None)


class TestErrorHandling:
    """Test error handling in caching operations."""
    
    def test_serialize_article_error(self):
        """Test serialization error handling."""
        # Create an article with invalid data that can't be serialized
        article = Article(url="https://example.com/test", title="Test")
        
        # Mock json.dumps to raise an exception
        with patch('curator.core.cache.json.dumps', side_effect=ValueError("Serialization error")):
            with pytest.raises(ValueError):
                serialize_article(article)
    
    def test_deserialize_article_error(self):
        """Test deserialization error handling."""
        invalid_json = "invalid json data"
        
        with pytest.raises(ValueError):
            deserialize_article(invalid_json)
    
    @patch('curator.core.cache.get_redis_client')
    def test_cache_operation_error_handling(self, mock_get_redis):
        """Test error handling in cache operations."""
        # Mock Redis client that raises exceptions
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis error")
        mock_redis.setex.side_effect = Exception("Redis error")
        mock_get_redis.return_value = mock_redis
        
        cache_manager = CacheManager()
        article = Article(url="https://example.com/test", title="Test")
        
        # Operations should return None/False on error, not raise exceptions
        assert cache_manager.get_article("https://example.com/test") is None
        assert not cache_manager.set_article("https://example.com/test", article)