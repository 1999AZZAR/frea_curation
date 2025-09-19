"""
Unit tests for topic coherence and coverage functionality.
"""

import pytest
from unittest.mock import Mock, patch
from curator.core.topic_coherence import (
    extract_keyphrases,
    normalize_text_for_matching,
    extract_query_keywords,
    calculate_keyword_coverage_ratio,
    calculate_keyphrase_relevance,
    compute_topic_coherence_score,
    get_article_keyphrases,
    get_yake_extractor
)


class TestTextNormalization:
    """Test text normalization utilities."""
    
    def test_normalize_text_for_matching(self):
        """Test text normalization for keyword matching."""
        # Basic normalization
        assert normalize_text_for_matching("Hello World!") == "hello world"
        
        # Punctuation removal
        assert normalize_text_for_matching("AI, machine-learning & NLP.") == "ai machine learning nlp"
        
        # Multiple spaces collapse
        assert normalize_text_for_matching("  multiple   spaces  ") == "multiple spaces"
        
        # Empty/None handling
        assert normalize_text_for_matching("") == ""
        assert normalize_text_for_matching(None) == ""
    
    def test_extract_query_keywords(self):
        """Test query keyword extraction."""
        # Basic keyword extraction
        keywords = extract_query_keywords("machine learning AI")
        assert keywords == {"machine", "learning", "ai"}
        
        # Punctuation handling
        keywords = extract_query_keywords("AI, machine-learning & NLP!")
        assert keywords == {"ai", "machine", "learning", "nlp"}
        
        # Short word filtering
        keywords = extract_query_keywords("AI is a big topic")
        assert keywords == {"ai", "big", "topic"}  # "is" and "a" filtered out
        
        # Empty query
        assert extract_query_keywords("") == set()
        assert extract_query_keywords(None) == set()


class TestKeywordCoverage:
    """Test keyword coverage calculation."""
    
    def test_calculate_keyword_coverage_ratio_perfect_match(self):
        """Test perfect keyword coverage."""
        article = "This article discusses machine learning and artificial intelligence applications."
        query = "machine learning AI"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        # All keywords (machine, learning, ai) should be found
        # Note: "ai" matches "artificial intelligence" through "ai" in "artificial"
        assert coverage == 1.0
    
    def test_calculate_keyword_coverage_ratio_partial_match(self):
        """Test partial keyword coverage."""
        article = "This article discusses machine learning applications."
        query = "machine learning AI robotics"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        # Only 2 out of 4 keywords found (machine, learning)
        assert coverage == 0.5
    
    def test_calculate_keyword_coverage_ratio_no_match(self):
        """Test no keyword coverage."""
        article = "This article discusses cooking recipes."
        query = "machine learning AI"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        assert coverage == 0.0
    
    def test_calculate_keyword_coverage_ratio_empty_inputs(self):
        """Test coverage with empty inputs."""
        assert calculate_keyword_coverage_ratio("", "query") == 0.0
        assert calculate_keyword_coverage_ratio("article", "") == 0.0
        assert calculate_keyword_coverage_ratio("", "") == 0.0


class TestKeyphraseExtraction:
    """Test keyphrase extraction functionality."""
    
    @patch('curator.core.topic_coherence.get_yake_extractor')
    def test_extract_keyphrases_success(self, mock_get_extractor):
        """Test successful keyphrase extraction."""
        # Mock YAKE extractor
        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = [
            ("machine learning", 0.1),
            ("artificial intelligence", 0.2),
            ("neural networks", 0.3)
        ]
        mock_get_extractor.return_value = mock_extractor
        
        text = "This article discusses machine learning and artificial intelligence."
        keyphrases = extract_keyphrases(text)
        
        expected = [
            ("machine learning", 0.1),
            ("artificial intelligence", 0.2),
            ("neural networks", 0.3)
        ]
        assert keyphrases == expected
    
    @patch('curator.core.topic_coherence.get_yake_extractor')
    def test_extract_keyphrases_yake_unavailable(self, mock_get_extractor):
        """Test keyphrase extraction when YAKE is unavailable."""
        mock_get_extractor.return_value = None
        
        text = "This article discusses machine learning."
        keyphrases = extract_keyphrases(text)
        
        assert keyphrases == []
    
    @patch('curator.core.topic_coherence.get_yake_extractor')
    def test_extract_keyphrases_extraction_error(self, mock_get_extractor):
        """Test keyphrase extraction with YAKE error."""
        mock_extractor = Mock()
        mock_extractor.extract_keywords.side_effect = Exception("YAKE error")
        mock_get_extractor.return_value = mock_extractor
        
        text = "This article discusses machine learning."
        keyphrases = extract_keyphrases(text)
        
        assert keyphrases == []
    
    def test_extract_keyphrases_empty_text(self):
        """Test keyphrase extraction with empty text."""
        assert extract_keyphrases("") == []
        assert extract_keyphrases(None) == []


class TestKeyphraseRelevance:
    """Test keyphrase relevance calculation."""
    
    def test_calculate_keyphrase_relevance_high_relevance(self):
        """Test high keyphrase relevance."""
        keyphrases = [
            ("machine learning", 0.1),  # Low YAKE score = high relevance
            ("artificial intelligence", 0.2),
            ("data science", 0.3)
        ]
        query = "machine learning AI"
        
        relevance = calculate_keyphrase_relevance(keyphrases, query)
        assert relevance > 0.5  # Should be high relevance
    
    def test_calculate_keyphrase_relevance_low_relevance(self):
        """Test low keyphrase relevance."""
        keyphrases = [
            ("cooking recipes", 5.0),  # High YAKE score = low relevance
            ("food preparation", 6.0),
        ]
        query = "machine learning AI"
        
        relevance = calculate_keyphrase_relevance(keyphrases, query)
        assert relevance == 0.0  # No overlap with query
    
    def test_calculate_keyphrase_relevance_empty_inputs(self):
        """Test keyphrase relevance with empty inputs."""
        assert calculate_keyphrase_relevance([], "query") == 0.0
        assert calculate_keyphrase_relevance([("phrase", 0.1)], "") == 0.0
        assert calculate_keyphrase_relevance([], "") == 0.0


class TestTopicCoherenceScore:
    """Test overall topic coherence scoring."""
    
    @patch('curator.core.topic_coherence.extract_keyphrases')
    def test_compute_topic_coherence_score_high_coherence(self, mock_extract):
        """Test high topic coherence score."""
        # Mock keyphrase extraction
        mock_extract.return_value = [
            ("machine learning", 0.1),
            ("artificial intelligence", 0.2)
        ]
        
        article_text = "This comprehensive article discusses machine learning algorithms and artificial intelligence applications in detail."
        article_title = "Machine Learning and AI: A Complete Guide"
        query = "machine learning artificial intelligence"
        
        score = compute_topic_coherence_score(article_text, article_title, query)
        
        # Should have high coherence due to keyword coverage and relevant keyphrases
        assert score > 70.0
        assert score <= 100.0
    
    @patch('curator.core.topic_coherence.extract_keyphrases')
    def test_compute_topic_coherence_score_low_coherence(self, mock_extract):
        """Test low topic coherence score."""
        # Mock keyphrase extraction with irrelevant phrases
        mock_extract.return_value = [
            ("cooking recipes", 5.0),
            ("food preparation", 6.0)
        ]
        
        article_text = "This article discusses cooking recipes and food preparation techniques."
        article_title = "Best Cooking Recipes"
        query = "machine learning artificial intelligence"
        
        score = compute_topic_coherence_score(article_text, article_title, query)
        
        # Should have low coherence due to no keyword overlap
        assert score == 0.0
    
    def test_compute_topic_coherence_score_empty_query(self):
        """Test topic coherence with empty query."""
        article_text = "This article discusses machine learning."
        article_title = "Machine Learning Guide"
        
        score = compute_topic_coherence_score(article_text, article_title, "")
        assert score == 0.0
        
        score = compute_topic_coherence_score(article_text, article_title, None)
        assert score == 0.0
    
    def test_compute_topic_coherence_score_empty_content(self):
        """Test topic coherence with empty content."""
        query = "machine learning"
        
        score = compute_topic_coherence_score("", "", query)
        assert score == 0.0
        
        score = compute_topic_coherence_score(None, None, query)
        assert score == 0.0


class TestArticleKeyphrases:
    """Test article keyphrase extraction for display."""
    
    @patch('curator.core.topic_coherence.extract_keyphrases')
    def test_get_article_keyphrases_success(self, mock_extract):
        """Test successful article keyphrase extraction."""
        mock_extract.return_value = [
            ("machine learning", 0.1),
            ("artificial intelligence", 0.2),
            ("neural networks", 0.3)
        ]
        
        article_text = "This article discusses machine learning."
        article_title = "ML Guide"
        
        keyphrases = get_article_keyphrases(article_text, article_title, max_keyphrases=5)
        
        expected = ["machine learning", "artificial intelligence", "neural networks"]
        assert keyphrases == expected
    
    @patch('curator.core.topic_coherence.extract_keyphrases')
    def test_get_article_keyphrases_limit(self, mock_extract):
        """Test keyphrase extraction with limit."""
        mock_extract.return_value = [
            ("phrase1", 0.1),
            ("phrase2", 0.2),
            ("phrase3", 0.3),
            ("phrase4", 0.4),
            ("phrase5", 0.5)
        ]
        
        article_text = "Sample article text."
        article_title = "Sample Title"
        
        keyphrases = get_article_keyphrases(article_text, article_title, max_keyphrases=3)
        
        # Should respect the limit
        expected = ["phrase1", "phrase2", "phrase3"]
        assert keyphrases == expected
    
    def test_get_article_keyphrases_empty_content(self):
        """Test keyphrase extraction with empty content."""
        keyphrases = get_article_keyphrases("", "", max_keyphrases=5)
        assert keyphrases == []
        
        keyphrases = get_article_keyphrases(None, None, max_keyphrases=5)
        assert keyphrases == []


class TestYakeExtractor:
    """Test YAKE extractor initialization."""
    
    @patch('curator.core.topic_coherence.yake')
    def test_get_yake_extractor_success(self, mock_yake):
        """Test successful YAKE extractor creation."""
        mock_extractor = Mock()
        mock_yake.KeywordExtractor.return_value = mock_extractor
        
        extractor = get_yake_extractor()
        
        assert extractor == mock_extractor
        mock_yake.KeywordExtractor.assert_called_once_with(
            lan="en",
            n=3,
            dedupLim=0.9,
            top=20,
            features=None
        )
    
    def test_get_yake_extractor_import_error(self):
        """Test YAKE extractor when import fails."""
        with patch('curator.core.topic_coherence.yake', side_effect=ImportError):
            extractor = get_yake_extractor()
            assert extractor is None
    
    @patch('curator.core.topic_coherence.yake')
    def test_get_yake_extractor_creation_error(self, mock_yake):
        """Test YAKE extractor when creation fails."""
        mock_yake.KeywordExtractor.side_effect = Exception("Creation failed")
        
        extractor = get_yake_extractor()
        assert extractor is None