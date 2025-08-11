"""
Unit tests for analyzer module.

Tests core scoring functions and orchestration with mocks to avoid
heavy external model downloads.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from models import Article, ScoringConfig
from curator.services.analyzer import (
    compute_readability_score,
    compute_ner_density_score,
    compute_sentiment_score,
    compute_tfidf_relevance_score,
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


if __name__ == "__main__":
    unittest.main()


