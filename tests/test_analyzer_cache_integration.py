"""
Integration tests for analyzer caching functionality.

Tests that the analyzer properly uses caching for articles and scorecards
to improve performance and reduce redundant processing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from curator.services._analyzer import analyze_article, batch_analyze
from curator.core.models import Article, ScoreCard, ScoringConfig, Entity
from curator.services._parser import ArticleParsingError


class TestAnalyzerCacheIntegration:
    """Test analyzer integration with caching layer."""
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.cache_scorecard')
    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_cache_hit_scorecard(self, mock_parse, mock_cache_scorecard, mock_get_cached_scorecard):
        """Test analyze_article returns cached scorecard when available."""
        # Setup cached scorecard
        article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="Test content"
        )
        cached_scorecard = ScoreCard(
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
        
        mock_get_cached_scorecard.return_value = cached_scorecard
        
        # Call analyze_article
        result = analyze_article("https://example.com/test", query="test query")
        
        # Should return cached scorecard without parsing
        assert result == cached_scorecard
        mock_get_cached_scorecard.assert_called_once_with("https://example.com/test", "test query")
        mock_parse.assert_not_called()
        mock_cache_scorecard.assert_not_called()
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.get_cached_article')
    @patch('curator.services._analyzer.cache_article')
    @patch('curator.services._analyzer.cache_scorecard')
    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_cache_hit_article(self, mock_parse, mock_cache_scorecard, 
                                             mock_cache_article, mock_get_cached_article, 
                                             mock_get_cached_scorecard):
        """Test analyze_article uses cached article when scorecard not cached."""
        # Setup: no cached scorecard, but cached article
        mock_get_cached_scorecard.return_value = None
        
        cached_article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="Test content with enough words to meet minimum requirements for scoring."
        )
        mock_get_cached_article.return_value = cached_article
        
        # Call analyze_article
        result = analyze_article("https://example.com/test", query="test query")
        
        # Should use cached article without parsing
        assert result.article == cached_article
        mock_get_cached_article.assert_called_once_with("https://example.com/test")
        mock_parse.assert_not_called()
        mock_cache_article.assert_not_called()  # Already cached
        mock_cache_scorecard.assert_called_once()  # Should cache new scorecard
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.get_cached_article')
    @patch('curator.services._analyzer.cache_article')
    @patch('curator.services._analyzer.cache_scorecard')
    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_cache_miss(self, mock_parse, mock_cache_scorecard, 
                                      mock_cache_article, mock_get_cached_article, 
                                      mock_get_cached_scorecard):
        """Test analyze_article when nothing is cached."""
        # Setup: no cached data
        mock_get_cached_scorecard.return_value = None
        mock_get_cached_article.return_value = None
        
        parsed_article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="Test content with enough words to meet minimum requirements for scoring."
        )
        mock_parse.return_value = parsed_article
        
        # Call analyze_article
        result = analyze_article("https://example.com/test", query="test query")
        
        # Should parse article and cache both article and scorecard
        assert result.article == parsed_article
        mock_parse.assert_called_once_with("https://example.com/test", min_word_count=300)
        mock_cache_article.assert_called_once_with("https://example.com/test", parsed_article)
        mock_cache_scorecard.assert_called_once()
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.get_cached_article')
    @patch('curator.services._analyzer.cache_scorecard')
    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_mixed_cache_hits(self, mock_batch_parse, mock_cache_scorecard,
                                          mock_get_cached_article, mock_get_cached_scorecard):
        """Test batch_analyze with mixed cache hits and misses."""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2", 
            "https://example.com/article3"
        ]
        
        # Setup cache responses
        def get_scorecard_side_effect(url, query):
            if url == "https://example.com/article1":
                # Return cached scorecard for article1
                article = Article(url=url, title="Cached Article 1", content="Content 1")
                return ScoreCard(
                    overall_score=85.0, readability_score=90.0, ner_density_score=75.0,
                    sentiment_score=80.0, tfidf_relevance_score=88.0, recency_score=95.0,
                    reputation_score=70.0, topic_coherence_score=82.0, article=article
                )
            return None
        
        def get_article_side_effect(url):
            if url == "https://example.com/article2":
                # Return cached article for article2 with enough content
                return Article(url=url, title="Cached Article 2", content="Content 2 with enough words to meet minimum requirements for scoring analysis.")
            return None
        
        mock_get_cached_scorecard.side_effect = get_scorecard_side_effect
        mock_get_cached_article.side_effect = get_article_side_effect
        
        # Setup batch parsing for article3 (not cached)
        article3 = Article(
            url="https://example.com/article3",
            title="New Article 3",
            content="Content 3 with enough words to meet minimum requirements for scoring analysis and processing."
        )
        mock_batch_parse.return_value = [article3]
        
        # Mock cache_article for newly parsed articles
        with patch('curator.services._analyzer.cache_article') as mock_cache_article:
        
            # Call batch_analyze (disable diversity filtering for predictable results)
            results = batch_analyze(urls, query="test query", apply_diversity=False)
            
            # Should return 3 results: 1 from scorecard cache, 2 newly processed
            assert len(results) == 3
            
            # Verify caching behavior
            assert mock_get_cached_scorecard.call_count == 3  # Check all URLs
            assert mock_get_cached_article.call_count == 2   # Check non-scorecard-cached URLs
            mock_batch_parse.assert_called_once_with(["https://example.com/article3"], min_word_count=300)
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.get_cached_article') 
    @patch('curator.services._analyzer.cache_article')
    @patch('curator.services._analyzer.cache_scorecard')
    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_all_cached(self, mock_batch_parse, mock_cache_scorecard,
                                    mock_cache_article, mock_get_cached_article,
                                    mock_get_cached_scorecard):
        """Test batch_analyze when all scorecards are cached."""
        urls = ["https://example.com/article1", "https://example.com/article2"]
        
        # Setup cached scorecards for all URLs
        def get_scorecard_side_effect(url, query):
            article = Article(url=url, title=f"Cached Article {url[-1]}", content="Content")
            return ScoreCard(
                overall_score=85.0, readability_score=90.0, ner_density_score=75.0,
                sentiment_score=80.0, tfidf_relevance_score=88.0, recency_score=95.0,
                reputation_score=70.0, topic_coherence_score=82.0, article=article
            )
        
        mock_get_cached_scorecard.side_effect = get_scorecard_side_effect
        
        # Call batch_analyze
        results = batch_analyze(urls, query="test query")
        
        # Should return cached results without any parsing
        assert len(results) == 2
        mock_batch_parse.assert_not_called()
        mock_get_cached_article.assert_not_called()
        mock_cache_article.assert_not_called()
        mock_cache_scorecard.assert_not_called()
    
    @patch('curator.core.cache.get_redis_client')
    def test_analyze_article_cache_unavailable(self, mock_get_redis):
        """Test analyze_article works normally when cache is unavailable."""
        # Mock Redis unavailable
        mock_get_redis.return_value = None
        
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            article = Article(
                url="https://example.com/test",
                title="Test Article", 
                content="Test content with enough words to meet minimum requirements for scoring."
            )
            mock_parse.return_value = article
            
            # Should work normally without caching
            result = analyze_article("https://example.com/test")
            
            assert result.article == article
            mock_parse.assert_called_once()
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.cache_scorecard')
    def test_analyze_article_cache_error_handling(self, mock_cache_scorecard, mock_get_cached_scorecard):
        """Test analyze_article handles cache errors gracefully."""
        # Mock cache operations to raise exceptions
        mock_get_cached_scorecard.side_effect = Exception("Cache error")
        mock_cache_scorecard.side_effect = Exception("Cache error")
        
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            article = Article(
                url="https://example.com/test",
                title="Test Article",
                content="Test content with enough words to meet minimum requirements for scoring."
            )
            mock_parse.return_value = article
            
            # Should work normally despite cache errors
            result = analyze_article("https://example.com/test")
            
            assert result.article == article
            mock_parse.assert_called_once()
    
    def test_cache_key_includes_query(self):
        """Test that cache keys include query for relevance-specific caching."""
        with patch('curator.services._analyzer.get_cached_scorecard') as mock_get_cached:
            mock_get_cached.return_value = None
            
            with patch('curator.services._analyzer.parse_article') as mock_parse:
                article = Article(
                    url="https://example.com/test",
                    title="Test Article",
                    content="Test content with enough words to meet minimum requirements."
                )
                mock_parse.return_value = article
                
                with patch('curator.services._analyzer.cache_scorecard') as mock_cache:
                    # Analyze with different queries
                    analyze_article("https://example.com/test", query="query1")
                    analyze_article("https://example.com/test", query="query2")
                    
                    # Should cache with different keys for different queries
                    assert mock_cache.call_count == 2
                    
                    # Verify different query strings were used
                    call1_query = mock_cache.call_args_list[0][0][2]  # Third argument is query
                    call2_query = mock_cache.call_args_list[1][0][2]
                    
                    assert call1_query == "query1"
                    assert call2_query == "query2"


class TestCachePerformance:
    """Test caching performance benefits."""
    
    @patch('curator.services._analyzer.get_cached_scorecard')
    @patch('curator.services._analyzer.parse_article')
    def test_cache_reduces_parsing_calls(self, mock_parse, mock_get_cached_scorecard):
        """Test that caching reduces expensive parsing operations."""
        # Setup cached scorecard
        article = Article(url="https://example.com/test", title="Test", content="Content")
        cached_scorecard = ScoreCard(
            overall_score=85.0, readability_score=90.0, ner_density_score=75.0,
            sentiment_score=80.0, tfidf_relevance_score=88.0, recency_score=95.0,
            reputation_score=70.0, topic_coherence_score=82.0, article=article
        )
        mock_get_cached_scorecard.return_value = cached_scorecard
        
        # Call analyze_article multiple times
        url = "https://example.com/test"
        for _ in range(5):
            result = analyze_article(url, query="test")
            assert result == cached_scorecard
        
        # Parsing should never be called due to caching
        mock_parse.assert_not_called()
        
        # Cache should be checked each time
        assert mock_get_cached_scorecard.call_count == 5