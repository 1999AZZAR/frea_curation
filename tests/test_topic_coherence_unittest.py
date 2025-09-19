#!/usr/bin/env python3
"""
Unit tests for topic coherence functionality using unittest.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


class TestTextNormalization(unittest.TestCase):
    """Test text normalization utilities."""
    
    def test_normalize_text_for_matching(self):
        """Test text normalization for keyword matching."""
        # Basic normalization
        self.assertEqual(normalize_text_for_matching("Hello World!"), "hello world")
        
        # Punctuation removal
        self.assertEqual(normalize_text_for_matching("AI, machine-learning & NLP."), "ai machine learning nlp")
        
        # Multiple spaces collapse
        self.assertEqual(normalize_text_for_matching("  multiple   spaces  "), "multiple spaces")
        
        # Empty/None handling
        self.assertEqual(normalize_text_for_matching(""), "")
        self.assertEqual(normalize_text_for_matching(None), "")
    
    def test_extract_query_keywords(self):
        """Test query keyword extraction."""
        # Basic keyword extraction
        keywords = extract_query_keywords("machine learning AI")
        self.assertEqual(keywords, {"machine", "learning", "ai"})
        
        # Punctuation handling
        keywords = extract_query_keywords("AI, machine-learning & NLP!")
        self.assertEqual(keywords, {"ai", "machine", "learning", "nlp"})
        
        # Short word filtering
        keywords = extract_query_keywords("AI is a big topic")
        self.assertEqual(keywords, {"ai", "is", "big", "topic"})  # "a" filtered out (< 2 chars)
        
        # Empty query
        self.assertEqual(extract_query_keywords(""), set())
        self.assertEqual(extract_query_keywords(None), set())


class TestKeywordCoverage(unittest.TestCase):
    """Test keyword coverage calculation."""
    
    def test_calculate_keyword_coverage_ratio_perfect_match(self):
        """Test perfect keyword coverage."""
        article = "This article discusses machine learning and AI applications."
        query = "machine learning AI"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        # All keywords (machine, learning, ai) should be found
        self.assertEqual(coverage, 1.0)
    
    def test_calculate_keyword_coverage_ratio_partial_match(self):
        """Test partial keyword coverage."""
        article = "This article discusses machine learning applications."
        query = "machine learning AI robotics"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        # Only 2 out of 4 keywords found (machine, learning)
        self.assertEqual(coverage, 0.5)
    
    def test_calculate_keyword_coverage_ratio_no_match(self):
        """Test no keyword coverage."""
        article = "This article discusses cooking recipes."
        query = "machine learning AI"
        
        coverage = calculate_keyword_coverage_ratio(article, query)
        self.assertEqual(coverage, 0.0)
    
    def test_calculate_keyword_coverage_ratio_empty_inputs(self):
        """Test coverage with empty inputs."""
        self.assertEqual(calculate_keyword_coverage_ratio("", "query"), 0.0)
        self.assertEqual(calculate_keyword_coverage_ratio("article", ""), 0.0)
        self.assertEqual(calculate_keyword_coverage_ratio("", ""), 0.0)


class TestKeyphraseExtraction(unittest.TestCase):
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
        self.assertEqual(keyphrases, expected)
    
    @patch('curator.core.topic_coherence.get_yake_extractor')
    def test_extract_keyphrases_yake_unavailable(self, mock_get_extractor):
        """Test keyphrase extraction when YAKE is unavailable."""
        mock_get_extractor.return_value = None
        
        text = "This article discusses machine learning."
        keyphrases = extract_keyphrases(text)
        
        self.assertEqual(keyphrases, [])
    
    def test_extract_keyphrases_empty_text(self):
        """Test keyphrase extraction with empty text."""
        self.assertEqual(extract_keyphrases(""), [])
        self.assertEqual(extract_keyphrases(None), [])


class TestTopicCoherenceScore(unittest.TestCase):
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
        self.assertGreater(score, 70.0)
        self.assertLessEqual(score, 100.0)
    
    def test_compute_topic_coherence_score_empty_query(self):
        """Test topic coherence with empty query."""
        article_text = "This article discusses machine learning."
        article_title = "Machine Learning Guide"
        
        score = compute_topic_coherence_score(article_text, article_title, "")
        self.assertEqual(score, 0.0)
        
        score = compute_topic_coherence_score(article_text, article_title, None)
        self.assertEqual(score, 0.0)
    
    def test_compute_topic_coherence_score_empty_content(self):
        """Test topic coherence with empty content."""
        query = "machine learning"
        
        score = compute_topic_coherence_score("", "", query)
        self.assertEqual(score, 0.0)
        
        score = compute_topic_coherence_score(None, None, query)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)