"""
Integration tests for the complete analysis pipeline.

Tests the full workflow from URL input to ScoreCard output,
including error handling and graceful degradation scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging

from curator.core.models import Article, ScoreCard, ScoringConfig, Entity
from curator.services.analyzer import analyze_article, batch_analyze
from curator.services._parser import ArticleParsingError, ContentValidationError


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for complete analysis workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
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
            entities=[]
        )
        
        self.config = ScoringConfig(
            readability_weight=0.2,
            ner_density_weight=0.2,
            sentiment_weight=0.15,
            tfidf_relevance_weight=0.25,
            recency_weight=0.2,
            min_word_count=50  # Lower for testing
        )

    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_complete_workflow(self, mock_parse):
        """Test complete analysis workflow with all components working."""
        mock_parse.return_value = self.sample_article
        
        # Mock NLP components
        mock_ent = Mock()
        mock_ent.text = "AI"
        mock_ent.label_ = "ORG"
        mock_doc = Mock()
        mock_doc.ents = [mock_ent]
        mock_nlp = Mock(return_value=mock_doc)
        
        mock_vader = Mock()
        mock_vader.polarity_scores.return_value = {"compound": 0.1}  # Slightly positive
        
        # Execute analysis
        result = analyze_article(
            url="https://example.com/tech-article",
            query="artificial intelligence breakthrough",
            config=self.config,
            nlp=mock_nlp,
            vader_analyzer=mock_vader
        )
        
        # Verify result structure
        self.assertIsInstance(result, ScoreCard)
        self.assertEqual(result.article, self.sample_article)
        
        # Verify all scores are valid
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertGreaterEqual(result.readability_score, 0)
        self.assertLessEqual(result.readability_score, 100)
        self.assertGreaterEqual(result.ner_density_score, 0)
        self.assertLessEqual(result.ner_density_score, 100)
        self.assertGreaterEqual(result.sentiment_score, 0)
        self.assertLessEqual(result.sentiment_score, 100)
        self.assertGreaterEqual(result.tfidf_relevance_score, 0)
        self.assertLessEqual(result.tfidf_relevance_score, 100)
        self.assertGreaterEqual(result.recency_score, 0)
        self.assertLessEqual(result.recency_score, 100)
        
        # Verify entities were extracted
        self.assertEqual(len(result.article.entities), 1)
        self.assertEqual(result.article.entities[0].text, "AI")
        self.assertEqual(result.article.entities[0].label, "ORG")

    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_parsing_failure(self, mock_parse):
        """Test analysis workflow when article parsing fails."""
        mock_parse.side_effect = ArticleParsingError("Failed to parse article")
        
        with self.assertRaises(ArticleParsingError):
            analyze_article("https://invalid-url.com")

    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_graceful_degradation_nlp_failure(self, mock_parse):
        """Test graceful degradation when NLP components fail."""
        mock_parse.return_value = self.sample_article
        
        # Mock failing NLP components
        mock_nlp = Mock(side_effect=Exception("spaCy model not available"))
        mock_vader = Mock()
        mock_vader.polarity_scores.side_effect = Exception("VADER not available")
        
        # Should not raise exception, but use fallback scores
        result = analyze_article(
            url="https://example.com/tech-article",
            query="test query",
            config=self.config,
            nlp=mock_nlp,
            vader_analyzer=mock_vader
        )
        
        # Verify result is still valid
        self.assertIsInstance(result, ScoreCard)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        
        # NER should fallback to 0
        self.assertEqual(result.ner_density_score, 0.0)
        
        # Sentiment should fallback to neutral (50)
        self.assertEqual(result.sentiment_score, 50.0)
        
        # Other scores should still work
        self.assertGreater(result.readability_score, 0)  # Should work with content
        self.assertGreater(result.recency_score, 0)  # Should work with date

    @patch('curator.services._analyzer.parse_article')
    def test_analyze_article_partial_component_failures(self, mock_parse):
        """Test analysis with some scoring components failing."""
        mock_parse.return_value = self.sample_article
        
        # Mock working NLP but failing relevance scoring
        mock_nlp = Mock()
        mock_nlp.return_value.ents = []
        mock_vader = Mock()
        mock_vader.polarity_scores.return_value = {"compound": 0.0}
        
        with patch('curator.services._analyzer.compute_relevance_score', 
                  side_effect=Exception("TF-IDF failed")):
            result = analyze_article(
                url="https://example.com/tech-article",
                query="test query",
                config=self.config,
                nlp=mock_nlp,
                vader_analyzer=mock_vader
            )
            
            # Should still return valid result
            self.assertIsInstance(result, ScoreCard)
            self.assertEqual(result.tfidf_relevance_score, 0.0)  # Fallback
            self.assertGreater(result.readability_score, 0)  # Should still work

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_complete_workflow(self, mock_batch_parse):
        """Test complete batch analysis workflow."""
        # Create multiple test articles
        articles = [
            Article(
                url="https://example.com/article1",
                title="First Article",
                content="Content about artificial intelligence and machine learning.",
                publish_date=datetime.now() - timedelta(hours=1)
            ),
            Article(
                url="https://example.com/article2", 
                title="Second Article",
                content="Different content about technology and innovation trends.",
                publish_date=datetime.now() - timedelta(hours=3)
            )
        ]
        mock_batch_parse.return_value = articles
        
        # Mock NLP components
        mock_nlp = Mock()
        mock_nlp.return_value.ents = []
        mock_vader = Mock()
        mock_vader.polarity_scores.return_value = {"compound": 0.0}
        
        urls = [article.url for article in articles]
        results = batch_analyze(
            urls=urls,
            query="artificial intelligence",
            config=self.config,
            nlp=mock_nlp,
            vader_analyzer=mock_vader
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, ScoreCard)
            self.assertGreaterEqual(result.overall_score, 0)
            self.assertLessEqual(result.overall_score, 100)

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_partial_failures(self, mock_batch_parse):
        """Test batch analysis with some articles failing during scoring."""
        # Return one valid article
        articles = [
            Article(
                url="https://example.com/good-article",
                title="Good Article",
                content="Valid content for analysis.",
                publish_date=datetime.now()
            )
        ]
        mock_batch_parse.return_value = articles
        
        # Mock one scoring component to fail intermittently
        call_count = 0
        def failing_sentiment(article, vader_analyzer=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Sentiment analysis failed")
            return 50.0
        
        with patch('curator.services._analyzer.compute_sentiment_score', 
                  side_effect=failing_sentiment):
            results = batch_analyze(
                urls=["https://example.com/good-article"],
                config=self.config
            )
            
            # Should still return results with fallback scores
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].sentiment_score, 50.0)  # Fallback

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_empty_input(self, mock_batch_parse):
        """Test batch analysis with empty input."""
        results = batch_analyze(urls=[])
        self.assertEqual(results, [])
        mock_batch_parse.assert_not_called()

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_all_parsing_failures(self, mock_batch_parse):
        """Test batch analysis when all articles fail to parse."""
        mock_batch_parse.return_value = []  # No articles parsed successfully
        
        results = batch_analyze(urls=["https://invalid1.com", "https://invalid2.com"])
        self.assertEqual(results, [])

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_individual_analysis_failures(self, mock_batch_parse):
        """Test batch analysis when individual article analysis fails completely."""
        # Return articles but make analysis fail
        articles = [
            Article(url="https://example.com/article1", title="Article 1", content="Content 1"),
            Article(url="https://example.com/article2", title="Article 2", content="Content 2")
        ]
        mock_batch_parse.return_value = articles
        
        # Mock composite score calculation to fail for first article
        original_calculate = None
        call_count = 0
        
        def failing_calculate(metrics, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Composite score calculation failed")
            return 75.0
        
        with patch('curator.services._analyzer.calculate_composite_score', 
                  side_effect=failing_calculate):
            results = batch_analyze(
                urls=["https://example.com/article1", "https://example.com/article2"],
                config=self.config
            )
            
            # Should return only the successful analysis
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].article.url, "https://example.com/article2")

    def test_analyze_article_with_default_config(self):
        """Test analysis with default configuration."""
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            mock_parse.return_value = self.sample_article
            
            result = analyze_article("https://example.com/article")
            
            # Should use default config
            self.assertIsInstance(result, ScoreCard)
            # Default config has min_word_count=300, our sample has more words
            self.assertGreater(result.readability_score, 0)

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_diversity_filtering(self, mock_batch_parse):
        """Test batch analysis with diversity filtering enabled."""
        # Create articles from same domain
        articles = [
            Article(
                url="https://example.com/article1",
                title="First Article",
                content="Content about AI technology.",
                publish_date=datetime.now()
            ),
            Article(
                url="https://example.com/article2",
                title="Second Article", 
                content="More content about AI technology.",
                publish_date=datetime.now()
            ),
            Article(
                url="https://different.com/article3",
                title="Third Article",
                content="Different content about technology.",
                publish_date=datetime.now()
            )
        ]
        mock_batch_parse.return_value = articles
        
        with patch.dict('os.environ', {'DIVERSIFY_RESULTS': 'true', 'DOMAIN_CAP': '1'}):
            results = batch_analyze(
                urls=[a.url for a in articles],
                config=self.config,
                apply_diversity=True
            )
            
            # Should apply diversity filtering
            self.assertLessEqual(len(results), 3)
            # Verify we don't have more than 1 article from example.com
            example_com_count = sum(1 for r in results if 'example.com' in r.article.url)
            self.assertLessEqual(example_com_count, 1)

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_diversity_filtering_failure(self, mock_batch_parse):
        """Test batch analysis when diversity filtering fails."""
        articles = [
            Article(url="https://example.com/article1", title="Article 1", content="Content 1")
        ]
        mock_batch_parse.return_value = articles
        
        with patch('curator.services._analyzer._apply_diversity_and_dedup', 
                  side_effect=Exception("Diversity filtering failed")):
            results = batch_analyze(
                urls=["https://example.com/article1"],
                config=self.config,
                apply_diversity=True
            )
            
            # Should return unfiltered results when diversity filtering fails
            self.assertEqual(len(results), 1)

    def test_error_logging_during_analysis(self):
        """Test that errors are properly logged during analysis."""
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            mock_parse.return_value = self.sample_article
            
            with patch('curator.services._analyzer.extract_entities', 
                      side_effect=Exception("Entity extraction failed")):
                with self.assertLogs('curator.services._analyzer', level='WARNING') as log:
                    result = analyze_article("https://example.com/article")
                    
                    # Should log the warning but continue
                    self.assertTrue(any('Entity extraction failed' in record.message 
                                      for record in log.records))
                    self.assertIsInstance(result, ScoreCard)

    @patch('curator.services._analyzer.batch_parse_articles')
    def test_batch_analyze_logging(self, mock_batch_parse):
        """Test that batch analysis logs progress and errors appropriately."""
        articles = [
            Article(url="https://example.com/article1", title="Article 1", content="Content 1")
        ]
        mock_batch_parse.return_value = articles
        
        with self.assertLogs('curator.services._analyzer', level='INFO') as log:
            results = batch_analyze(
                urls=["https://example.com/article1", "https://example.com/article2"],
                config=self.config
            )
            
            # Should log parsing success
            self.assertTrue(any('Successfully parsed' in record.message 
                              for record in log.records))
            # Should log analysis success
            self.assertTrue(any('Successfully analyzed' in record.message 
                              for record in log.records))


class TestAnalysisEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_analyze_article_invalid_url(self):
        """Test analysis with invalid URL."""
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            mock_parse.side_effect = ArticleParsingError("Invalid URL")
            
            with self.assertRaises(ArticleParsingError):
                analyze_article("")

    def test_analyze_article_with_minimal_content(self):
        """Test analysis with minimal article content."""
        minimal_article = Article(
            url="https://example.com/minimal",
            title="Short",
            content="Short content.",
            publish_date=None
        )
        
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            mock_parse.return_value = minimal_article
            
            result = analyze_article("https://example.com/minimal")
            
            # Should handle minimal content gracefully
            self.assertIsInstance(result, ScoreCard)
            self.assertEqual(result.recency_score, 100.0)  # No date = assume recent

    def test_analyze_article_with_invalid_scores(self):
        """Test analysis when scoring functions return invalid values."""
        article = Article(
            url="https://example.com/test",
            title="Test Article",
            content="Test content for analysis.",
            publish_date=datetime.now()
        )
        
        with patch('curator.services._analyzer.parse_article') as mock_parse:
            mock_parse.return_value = article
            
            # Mock scoring function to return invalid score
            with patch('curator.services._analyzer.compute_readability_score', 
                      return_value=150.0):  # Invalid score > 100
                result = analyze_article("https://example.com/test")
                
                # Should normalize invalid score to 0
                self.assertEqual(result.readability_score, 0.0)

    def test_batch_analyze_mixed_success_failure(self):
        """Test batch analysis with mixed success and failure scenarios."""
        with patch('curator.services._analyzer.batch_parse_articles') as mock_batch:
            # Return some articles
            articles = [
                Article(url="https://example.com/good", title="Good", content="Good content"),
                Article(url="https://example.com/bad", title="Bad", content="Bad content")
            ]
            mock_batch.return_value = articles
            
            # Make analysis fail for second article
            call_count = 0
            def failing_analysis(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second article
                    raise Exception("Analysis failed")
                return 75.0
            
            with patch('curator.services._analyzer.calculate_composite_score',
                      side_effect=failing_analysis):
                results = batch_analyze(
                    urls=["https://example.com/good", "https://example.com/bad"]
                )
                
                # Should return only successful analysis
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].article.url, "https://example.com/good")


if __name__ == "__main__":
    unittest.main()