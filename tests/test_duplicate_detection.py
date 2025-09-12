"""
Unit tests for duplicate detection and diversity controls.

Tests URL canonicalization, near-duplicate collapse, and domain diversity caps
to ensure proper deduplication and result diversification.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import os

from curator.core.models import Article, ScoreCard, ScoringConfig
from curator.services._analyzer import _apply_diversity_and_dedup, _extract_domain
from curator.services._news_source import NewsSource
from curator.core.utils import canonicalize_url, extract_domain


class TestURLCanonicalization(unittest.TestCase):
    """Test URL canonicalization functionality."""
    
    def test_canonicalize_url_basic(self):
        """Test basic URL canonicalization."""
        url = "https://WWW.Example.com/article"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_remove_utm_params(self):
        """Test removal of UTM tracking parameters."""
        url = "https://example.com/article?utm_source=twitter&utm_medium=social&utm_campaign=test"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_remove_tracking_params(self):
        """Test removal of various tracking parameters."""
        url = "https://example.com/article?gclid=123&fbclid=456&mc_eid=789&mc_cid=abc&_ga=xyz&ref=twitter"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_preserve_valid_params(self):
        """Test preservation of valid query parameters."""
        url = "https://example.com/article?id=123&category=tech&utm_source=spam"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article?category=tech&id=123")
    
    def test_canonicalize_url_remove_fragment(self):
        """Test removal of URL fragments."""
        url = "https://example.com/article#section1"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_normalize_www(self):
        """Test normalization of www subdomain."""
        url = "https://www.example.com/article"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_sort_params(self):
        """Test sorting of query parameters for stability."""
        url = "https://example.com/article?z=1&a=2&m=3"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article?a=2&m=3&z=1")
    
    def test_canonicalize_url_invalid_url(self):
        """Test handling of invalid URLs."""
        url = "not-a-valid-url"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, url)  # Should return original on error
    
    def test_canonicalize_url_missing_scheme(self):
        """Test handling of URLs with missing scheme."""
        url = "example.com/article"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
    
    def test_canonicalize_url_mobile_variants(self):
        """Test normalization of mobile URL variants."""
        urls = [
            "https://m.example.com/article",
            "https://mobile.example.com/article", 
            "https://www.example.com/article"
        ]
        expected = "https://example.com/article"
        
        for url in urls:
            canonical = canonicalize_url(url)
            self.assertEqual(canonical, expected)
    
    def test_canonicalize_url_amp_variants(self):
        """Test normalization of AMP URL variants."""
        urls = [
            "https://example.com/article/amp",
            "https://example.com/amp/article",
            "https://example.com/article"
        ]
        expected = "https://example.com/article"
        
        for url in urls:
            canonical = canonicalize_url(url)
            self.assertEqual(canonical, expected)
    
    def test_canonicalize_url_trailing_slash(self):
        """Test removal of trailing slashes."""
        url = "https://example.com/article/"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/article")
        
        # Root path should keep trailing slash
        url = "https://example.com/"
        canonical = canonicalize_url(url)
        self.assertEqual(canonical, "https://example.com/")
    
    def test_canonicalize_url_comprehensive_tracking_removal(self):
        """Test removal of comprehensive set of tracking parameters."""
        url = ("https://example.com/article?utm_source=google&gclid=123&fbclid=456"
               "&_ga=GA1.2.123&ref=twitter&newsletter=true&affiliate=partner123"
               "&id=article123&category=tech")
        canonical = canonicalize_url(url)
        # Should only keep id and category parameters
        self.assertEqual(canonical, "https://example.com/article?category=tech&id=article123")


class TestDomainExtraction(unittest.TestCase):
    """Test domain extraction functionality."""
    
    def test_extract_domain_basic(self):
        """Test basic domain extraction."""
        url = "https://example.com/article"
        domain = extract_domain(url)
        self.assertEqual(domain, "example.com")
    
    def test_extract_domain_with_www(self):
        """Test domain extraction with www prefix."""
        url = "https://www.example.com/article"
        domain = extract_domain(url)
        self.assertEqual(domain, "example.com")
    
    def test_extract_domain_with_subdomain(self):
        """Test domain extraction with subdomain."""
        url = "https://news.example.com/article"
        domain = extract_domain(url)
        self.assertEqual(domain, "news.example.com")
    
    def test_extract_domain_with_port(self):
        """Test domain extraction with port."""
        url = "https://example.com:8080/article"
        domain = extract_domain(url)
        self.assertEqual(domain, "example.com:8080")
    
    def test_extract_domain_invalid_url(self):
        """Test domain extraction with invalid URL."""
        url = "not-a-valid-url"
        domain = extract_domain(url)
        self.assertEqual(domain, "")


class TestDiversityAndDeduplication(unittest.TestCase):
    """Test diversity controls and deduplication functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ScoringConfig()
        
        # Create sample articles from different domains
        self.articles = [
            Article(
                url="https://example.com/article1",
                title="First Article",
                content="Content about AI technology and machine learning.",
                publish_date=datetime.now()
            ),
            Article(
                url="https://example.com/article2", 
                title="Second Article",
                content="More content about AI technology and deep learning.",
                publish_date=datetime.now()
            ),
            Article(
                url="https://different.com/article3",
                title="Third Article", 
                content="Different content about blockchain technology.",
                publish_date=datetime.now()
            ),
            Article(
                url="https://another.com/article4",
                title="Fourth Article",
                content="Content about quantum computing and physics.",
                publish_date=datetime.now()
            )
        ]
        
        # Create corresponding scorecards
        self.scorecards = []
        for i, article in enumerate(self.articles):
            scorecard = ScoreCard(
                overall_score=90.0 - i * 5,  # Descending scores
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0 - i * 5,
                recency_score=95.0,
                reputation_score=75.0,
                article=article
            )
            self.scorecards.append(scorecard)
    
    def test_domain_diversity_cap(self):
        """Test domain diversity cap functionality."""
        # Create multiple articles from same domain
        same_domain_articles = []
        for i in range(5):
            article = Article(
                url=f"https://example.com/article{i}",
                title=f"Article {i}",
                content=f"Content {i}",
                publish_date=datetime.now()
            )
            scorecard = ScoreCard(
                overall_score=90.0 - i,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=article
            )
            same_domain_articles.append(scorecard)
        
        # Apply diversity with domain cap of 2
        result = _apply_diversity_and_dedup(same_domain_articles, domain_cap=2)
        
        # Should only keep 2 articles from example.com
        self.assertEqual(len(result), 2)
        
        # Should keep the highest scoring articles
        self.assertEqual(result[0].overall_score, 90.0)
        self.assertEqual(result[1].overall_score, 89.0)
    
    def test_no_diversity_filtering_when_under_cap(self):
        """Test that diversity filtering doesn't affect results under domain cap."""
        # Use articles from different domains
        result = _apply_diversity_and_dedup(self.scorecards, domain_cap=2)
        
        # Should keep all articles since they're from different domains
        self.assertEqual(len(result), 4)
    
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_near_duplicate_collapse_with_embeddings(self, mock_get_model):
        """Test near-duplicate collapse using embeddings."""
        # Mock sentence transformer model
        mock_model = Mock()
        mock_model.encode.return_value = [
            [1.0, 0.0, 0.0],  # First article embedding
            [0.98, 0.1, 0.1],  # Very similar to first (should be filtered)
            [0.0, 1.0, 0.0],  # Different embedding
            [0.0, 0.0, 1.0]   # Another different embedding
        ]
        mock_get_model.return_value = mock_model
        
        # Apply diversity with high similarity threshold
        result = _apply_diversity_and_dedup(self.scorecards, sim_threshold=0.95)
        
        # Should filter out the similar article
        self.assertLess(len(result), 4)
        
        # Should keep the highest scoring article from similar pair
        kept_urls = [r.article.url for r in result]
        self.assertIn("https://example.com/article1", kept_urls)
    
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_near_duplicate_collapse_no_embeddings(self, mock_get_model):
        """Test near-duplicate collapse fallback when embeddings unavailable."""
        # Mock no embeddings available
        mock_get_model.return_value = None
        
        # Create articles with same title (should be detected as duplicates)
        duplicate_articles = [
            ScoreCard(
                overall_score=90.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://example.com/article1",
                    title="Same Title",
                    content="Different content 1"
                )
            ),
            ScoreCard(
                overall_score=85.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://different.com/article2",
                    title="Same Title",
                    content="Different content 2"
                )
            )
        ]
        
        result = _apply_diversity_and_dedup(duplicate_articles)
        
        # Should filter out duplicate by title
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].overall_score, 90.0)  # Keep higher scoring
    
    def test_empty_input(self):
        """Test diversity filtering with empty input."""
        result = _apply_diversity_and_dedup([])
        self.assertEqual(result, [])
    
    def test_single_article(self):
        """Test diversity filtering with single article."""
        result = _apply_diversity_and_dedup([self.scorecards[0]])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], self.scorecards[0])
    
    @patch.dict('os.environ', {'DOMAIN_CAP': '3', 'DUP_SIM_THRESHOLD': '0.8'})
    def test_environment_variable_configuration(self):
        """Test that environment variables are used for configuration."""
        # Create articles to test environment variable usage
        many_articles = []
        for i in range(6):
            article = Article(
                url=f"https://example.com/article{i}",
                title=f"Article {i}",
                content=f"Content {i}",
                publish_date=datetime.now()
            )
            scorecard = ScoreCard(
                overall_score=90.0 - i,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=article
            )
            many_articles.append(scorecard)
        
        # Should use DOMAIN_CAP=3 from environment
        result = _apply_diversity_and_dedup(many_articles)
        self.assertEqual(len(result), 3)
    
    def test_preserve_score_ordering(self):
        """Test that diversity filtering preserves score-based ordering."""
        result = _apply_diversity_and_dedup(self.scorecards, domain_cap=10)
        
        # Should maintain descending score order
        scores = [r.overall_score for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embedding_error_handling(self, mock_get_model):
        """Test graceful handling of embedding errors."""
        # Mock model that raises exception during encoding
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_get_model.return_value = mock_model
        
        # Should fall back to basic deduplication
        result = _apply_diversity_and_dedup(self.scorecards)
        
        # Should still return results (fallback to basic dedup)
        self.assertGreater(len(result), 0)
    
    @patch('curator.services._analyzer._get_sentence_transformer')
    @patch('curator.services._analyzer._try_simhash_dedup')
    def test_simhash_fallback(self, mock_simhash, mock_get_model):
        """Test simhash fallback when embeddings unavailable."""
        # Mock no embeddings available
        mock_get_model.return_value = None
        
        # Mock simhash available
        mock_simhash.return_value = True
        
        # Create articles with similar content for simhash testing
        similar_articles = [
            ScoreCard(
                overall_score=90.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://example.com/article1",
                    title="AI Technology Advances",
                    content="Artificial intelligence technology is advancing rapidly with new breakthroughs."
                )
            ),
            ScoreCard(
                overall_score=85.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://different.com/article2",
                    title="AI Tech Progress",
                    content="Artificial intelligence technology is advancing rapidly with new developments."
                )
            )
        ]
        
        # Mock simhash objects
        with patch('simhash.Simhash') as mock_simhash_class:
            mock_hash1 = Mock()
            mock_hash2 = Mock()
            mock_hash1.distance.return_value = 1  # Very similar (low hamming distance)
            mock_hash2.distance.return_value = 1
            
            mock_simhash_class.side_effect = [mock_hash1, mock_hash2]
            
            result = _apply_diversity_and_dedup(similar_articles)
            
            # Should detect similarity and filter duplicates
            mock_simhash.assert_called_once()
    
    @patch('curator.services._analyzer._get_sentence_transformer')
    @patch('curator.services._analyzer._try_simhash_dedup')
    def test_basic_fallback_when_no_advanced_methods(self, mock_simhash, mock_get_model):
        """Test basic deduplication when neither embeddings nor simhash available."""
        # Mock no embeddings available
        mock_get_model.return_value = None
        
        # Mock no simhash available
        mock_simhash.return_value = False
        
        # Create articles with exact same title
        duplicate_articles = [
            ScoreCard(
                overall_score=90.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://example.com/article1",
                    title="Exact Same Title",
                    content="Different content 1"
                )
            ),
            ScoreCard(
                overall_score=85.0,
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                article=Article(
                    url="https://different.com/article2",
                    title="Exact Same Title",
                    content="Different content 2"
                )
            )
        ]
        
        result = _apply_diversity_and_dedup(duplicate_articles)
        
        # Should detect duplicate by title and keep only one
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].overall_score, 90.0)  # Keep higher scoring


if __name__ == '__main__':
    unittest.main(verbosity=2)