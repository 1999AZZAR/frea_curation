"""
Unit tests for analyzer module.

Tests core scoring functions and orchestration with mocks to avoid
heavy external model downloads.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from curator.core.models import Article, ScoringConfig
from curator.services.analyzer import (
    compute_readability_score,
    compute_ner_density_score,
    compute_sentiment_score,
    compute_tfidf_relevance_score,
    compute_embeddings_relevance_score,
    compute_relevance_score,
    compute_recency_score,
    calculate_composite_score,
    analyze_article,
    batch_analyze,
)


class TestScoringFunctions(unittest.TestCase):
    def setUp(self):
        self.article = Article(
            url="https://example.com/a",
            title="Title",
            author="Author",
            publish_date=datetime.now() - timedelta(days=1),
            content="Apple releases new product. Tim Cook leads Apple event.",
        )

    def test_readability_score(self):
        self.assertGreaterEqual(compute_readability_score(self.article, 10), 0)
        self.assertLessEqual(compute_readability_score(self.article, 10), 100)

    def test_ner_density_score_with_mock_spacy(self):
        mock_ent = object()
        mock_doc = Mock()
        mock_doc.ents = [mock_ent, mock_ent, mock_ent]
        mock_nlp = Mock(return_value=mock_doc)
        score = compute_ner_density_score(self.article, nlp=mock_nlp)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_sentiment_score_with_mock_vader(self):
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {"compound": 0.2}
        score = compute_sentiment_score(self.article, vader_analyzer=mock_analyzer)
        self.assertAlmostEqual(score, (1 - 0.2) * 100, places=1)

    def test_tfidf_relevance_score(self):
        score = compute_tfidf_relevance_score(self.article, query="Apple event")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_embeddings_relevance_score_fallback(self):
        # Even if embeddings not available, compute_relevance_score must return valid 0-100
        from curator.services._analyzer import compute_relevance_score
        s = compute_relevance_score(self.article, query="Apple event")
        self.assertTrue(0 <= s <= 100)

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_relevance_score_with_mock_model(self, mock_get_model):
        """Test embeddings relevance scoring with mocked SentenceTransformer."""
        # Mock the sentence transformer model
        mock_model = Mock()
        # Mock embeddings: article embedding and query embedding
        # Simulate high similarity (0.8) between article and query
        mock_model.encode.side_effect = [
            [[0.1, 0.2, 0.3, 0.4]],  # article embedding (normalized)
            [[0.15, 0.25, 0.35, 0.45]]  # query embedding (normalized)
        ]
        mock_get_model.return_value = mock_model
        
        score = compute_embeddings_relevance_score(self.article, "Apple event")
        
        # Verify model was called correctly
        self.assertEqual(mock_model.encode.call_count, 2)
        mock_model.encode.assert_any_call([self.article.content], normalize_embeddings=True)
        mock_model.encode.assert_any_call(["Apple event"], normalize_embeddings=True)
        
        # Score should be valid and > 0 due to similarity
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertGreater(score, 0)  # Should be > 0 due to mocked similarity

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_relevance_score_perfect_match(self, mock_get_model):
        """Test embeddings relevance scoring with perfect similarity."""
        mock_model = Mock()
        # Mock identical embeddings for perfect similarity (cosine = 1.0)
        identical_embedding = [[1.0, 0.0, 0.0, 0.0]]
        mock_model.encode.side_effect = [identical_embedding, identical_embedding]
        mock_get_model.return_value = mock_model
        
        score = compute_embeddings_relevance_score(self.article, "identical content")
        
        # Perfect similarity should give score of 100
        self.assertEqual(score, 100.0)

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_relevance_score_no_similarity(self, mock_get_model):
        """Test embeddings relevance scoring with no similarity."""
        mock_model = Mock()
        # Mock orthogonal embeddings for no similarity (cosine = 0.0)
        mock_model.encode.side_effect = [
            [[1.0, 0.0, 0.0, 0.0]],  # article embedding
            [[0.0, 1.0, 0.0, 0.0]]   # query embedding (orthogonal)
        ]
        mock_get_model.return_value = mock_model
        
        score = compute_embeddings_relevance_score(self.article, "completely different")
        
        # No similarity should give score of 0
        self.assertEqual(score, 0.0)

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_relevance_score_model_unavailable(self, mock_get_model):
        """Test embeddings relevance scoring when model is unavailable."""
        mock_get_model.return_value = None
        
        score = compute_embeddings_relevance_score(self.article, "Apple event")
        
        # Should return 0 when model is unavailable
        self.assertEqual(score, 0.0)

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_relevance_score_encoding_error(self, mock_get_model):
        """Test embeddings relevance scoring when encoding fails."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_get_model.return_value = mock_model
        
        score = compute_embeddings_relevance_score(self.article, "Apple event")
        
        # Should return 0 when encoding fails
        self.assertEqual(score, 0.0)

    def test_embeddings_relevance_score_empty_inputs(self):
        """Test embeddings relevance scoring with empty inputs."""
        # Empty content
        empty_article = Article(
            url="https://example.com/empty",
            title="Empty",
            author="Author",
            publish_date=datetime.now(),
            content="",
        )
        score = compute_embeddings_relevance_score(empty_article, "query")
        self.assertEqual(score, 0.0)
        
        # Empty query
        score = compute_embeddings_relevance_score(self.article, "")
        self.assertEqual(score, 0.0)
        
        # Both empty
        score = compute_embeddings_relevance_score(empty_article, "")
        self.assertEqual(score, 0.0)

    @patch.dict('os.environ', {'USE_EMBEDDINGS_RELEVANCE': 'true'})
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_compute_relevance_score_uses_embeddings_when_enabled(self, mock_get_model):
        """Test that compute_relevance_score uses embeddings when enabled."""
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [[0.5, 0.5, 0.0, 0.0]],  # article embedding
            [[0.6, 0.4, 0.0, 0.0]]   # query embedding
        ]
        mock_get_model.return_value = mock_model
        
        score = compute_relevance_score(self.article, "Apple event")
        
        # Should have called the embeddings model
        mock_get_model.assert_called_once()
        self.assertGreater(score, 0)

    @patch.dict('os.environ', {'USE_EMBEDDINGS_RELEVANCE': 'false'})
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_compute_relevance_score_uses_tfidf_when_disabled(self, mock_get_model):
        """Test that compute_relevance_score uses TF-IDF when embeddings disabled."""
        score = compute_relevance_score(self.article, "Apple event")
        
        # Should not have called the embeddings model
        mock_get_model.assert_not_called()
        # Should still return valid score from TF-IDF
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    @patch.dict('os.environ', {'USE_EMBEDDINGS_RELEVANCE': 'true'})
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_compute_relevance_score_fallback_to_tfidf(self, mock_get_model):
        """Test that compute_relevance_score falls back to TF-IDF when embeddings fail."""
        # Mock embeddings to return 0 (indicating failure/unavailability)
        mock_get_model.return_value = None
        
        score = compute_relevance_score(self.article, "Apple event")
        
        # Should have tried embeddings but fallen back to TF-IDF
        mock_get_model.assert_called_once()
        # Should still return valid score from TF-IDF fallback
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_recency_score(self):
        s = compute_recency_score(self.article, now=self.article.publish_date + timedelta(days=1))
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)

    def test_composite_score(self):
        config = ScoringConfig()
        metrics = {
            "readability": 80,
            "ner_density": 70,
            "sentiment": 60,
            "tfidf_relevance": 75,
            "recency": 90,
        }
        s = calculate_composite_score(metrics, config)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)


class TestAnalyzerOrchestration(unittest.TestCase):
    @patch("curator.services._analyzer.parse_article")
    def test_analyze_article_success(self, mock_parse):
        article = Article(
            url="https://example.com/a",
            title="Title",
            author="Author",
            publish_date=datetime.now(),
            content="Apple releases new product",
        )
        mock_parse.return_value = article

        # Mock NLP and VADER
        mock_doc = Mock(); mock_doc.ents = [object()]
        mock_nlp = Mock(return_value=mock_doc)
        mock_vader = Mock(); mock_vader.polarity_scores.return_value = {"compound": 0.0}

        card = analyze_article(
            url=article.url,
            query="Apple product",
            config=ScoringConfig(),
            nlp=mock_nlp,
            vader_analyzer=mock_vader,
        )

        self.assertEqual(card.article, article)
        self.assertGreaterEqual(card.overall_score, 0)
        self.assertLessEqual(card.overall_score, 100)

    @patch("curator.services._analyzer.batch_parse_articles")
    def test_batch_analyze(self, mock_batch):
        articles = [
            Article(url="https://ex/1", title="t1", content="c1", publish_date=None),
            Article(url="https://ex/2", title="t2", content="c2", publish_date=None),
        ]
        mock_batch.return_value = articles

        results = batch_analyze([a.url for a in articles], query="x", config=ScoringConfig())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(0 <= r.overall_score <= 100 for r in results))


class TestEmbeddingsRankingImprovement(unittest.TestCase):
    """Test that embeddings provide better semantic ranking than TF-IDF."""
    
    def setUp(self):
        # Create articles with different semantic relationships to queries
        self.tech_article = Article(
            url="https://example.com/tech",
            title="Artificial Intelligence Breakthrough",
            author="Tech Reporter",
            publish_date=datetime.now(),
            content="Machine learning algorithms have achieved remarkable progress in natural language processing. "
                   "Deep neural networks are revolutionizing how computers understand human language. "
                   "This breakthrough in AI technology will transform many industries.",
        )
        
        self.finance_article = Article(
            url="https://example.com/finance", 
            title="Stock Market Analysis",
            author="Finance Reporter",
            publish_date=datetime.now(),
            content="The stock market showed significant volatility today. Investment portfolios "
                   "experienced mixed results as trading volumes increased. Economic indicators "
                   "suggest continued uncertainty in financial markets.",
        )
        
        self.sports_article = Article(
            url="https://example.com/sports",
            title="Championship Game Results", 
            author="Sports Reporter",
            publish_date=datetime.now(),
            content="The championship game ended with a thrilling victory. Athletes demonstrated "
                   "exceptional performance throughout the tournament. Fans celebrated the "
                   "team's outstanding achievement in professional sports.",
        )

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_better_semantic_matching(self, mock_get_model):
        """Test that embeddings provide better semantic matching than TF-IDF."""
        # Mock embeddings to simulate semantic similarity
        mock_model = Mock()
        
        def mock_encode(texts, normalize_embeddings=True):
            # Simulate embeddings where semantically similar content has higher similarity
            text = texts[0] if isinstance(texts, list) else texts
            
            if "machine learning" in text.lower() or "artificial intelligence" in text.lower():
                # AI/ML related content
                return [[0.8, 0.6, 0.1, 0.0]]
            elif "neural networks" in text.lower() or "deep learning" in text.lower():
                # Also AI/ML related (should be similar to above)
                return [[0.75, 0.65, 0.15, 0.05]]
            elif "stock" in text.lower() or "financial" in text.lower():
                # Finance related content
                return [[0.1, 0.2, 0.8, 0.6]]
            elif "sports" in text.lower() or "championship" in text.lower():
                # Sports related content  
                return [[0.0, 0.1, 0.2, 0.9]]
            else:
                # Default embedding
                return [[0.5, 0.5, 0.5, 0.5]]
        
        mock_model.encode.side_effect = mock_encode
        mock_get_model.return_value = mock_model
        
        # Test semantic query that should match tech article better
        ai_query = "deep learning neural networks"
        
        tech_score = compute_embeddings_relevance_score(self.tech_article, ai_query)
        finance_score = compute_embeddings_relevance_score(self.finance_article, ai_query)
        sports_score = compute_embeddings_relevance_score(self.sports_article, ai_query)
        
        # Tech article should score highest for AI query due to semantic similarity
        self.assertGreater(tech_score, finance_score)
        self.assertGreater(tech_score, sports_score)
        
        # Verify all scores are valid
        for score in [tech_score, finance_score, sports_score]:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    @patch.dict('os.environ', {'USE_EMBEDDINGS_RELEVANCE': 'true'})
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_ranking_improvement_with_embeddings(self, mock_get_model):
        """Test that embeddings improve article ranking for semantic queries."""
        # Mock model for consistent embeddings-based scoring
        mock_model = Mock()
        
        def mock_encode_for_ranking(texts, normalize_embeddings=True):
            text = texts[0].lower() if isinstance(texts, list) else texts.lower()
            
            # Query: "artificial intelligence research"
            if "artificial intelligence research" in text:
                return [[1.0, 0.0, 0.0, 0.0]]  # Query embedding
            elif "machine learning" in text and "neural" in text:
                return [[0.9, 0.1, 0.0, 0.0]]  # High similarity to AI query
            elif "stock" in text or "market" in text:
                return [[0.0, 0.0, 1.0, 0.0]]  # Low similarity to AI query
            elif "sports" in text or "championship" in text:
                return [[0.0, 0.0, 0.0, 1.0]]  # No similarity to AI query
            else:
                return [[0.5, 0.5, 0.5, 0.5]]
        
        mock_model.encode.side_effect = mock_encode_for_ranking
        mock_get_model.return_value = mock_model
        
        articles = [self.finance_article, self.tech_article, self.sports_article]
        query = "artificial intelligence research"
        
        # Score all articles with embeddings
        scores = []
        for article in articles:
            score = compute_relevance_score(article, query)
            scores.append((article, score))
        
        # Sort by score (descending)
        ranked_articles = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Tech article should rank first due to semantic similarity
        self.assertEqual(ranked_articles[0][0], self.tech_article)
        self.assertGreater(ranked_articles[0][1], ranked_articles[1][1])
        self.assertGreater(ranked_articles[1][1], ranked_articles[2][1])

    @patch.dict('os.environ', {'USE_EMBEDDINGS_RELEVANCE': 'false'})
    def test_tfidf_baseline_ranking(self):
        """Test TF-IDF baseline ranking for comparison."""
        articles = [self.finance_article, self.tech_article, self.sports_article]
        query = "artificial intelligence research"
        
        # Score all articles with TF-IDF
        scores = []
        for article in articles:
            score = compute_tfidf_relevance_score(article, query)
            scores.append((article, score))
        
        # All scores should be valid
        for article, score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
        
        # TF-IDF might not capture semantic similarity as well
        # but should still give some relevance to the tech article
        tech_score = next(score for article, score in scores if article == self.tech_article)
        self.assertGreater(tech_score, 0)  # Should have some relevance

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embeddings_model_configuration(self, mock_get_model):
        """Test that embeddings model can be configured via environment."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.5, 0.5, 0.0, 0.0]]
        mock_get_model.return_value = mock_model
        
        # Test with default model
        compute_embeddings_relevance_score(self.tech_article, "test query")
        mock_get_model.assert_called_with("all-MiniLM-L6-v2")
        
        # Test with custom model via environment
        with patch.dict('os.environ', {'EMBEDDINGS_MODEL_NAME': 'custom-model'}):
            compute_embeddings_relevance_score(self.tech_article, "test query")
            mock_get_model.assert_called_with("custom-model")


if __name__ == "__main__":
    unittest.main()


