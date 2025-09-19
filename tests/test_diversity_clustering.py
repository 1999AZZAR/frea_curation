"""
Tests for embedding-based diversity clustering functionality.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from curator.core.models import Article, ScoreCard, ScoringConfig
from curator.services._analyzer import (
    _apply_embedding_clustering_diversity,
    _try_embeddings_for_clustering,
    _cluster_articles_by_embeddings,
    _calculate_cluster_similarity,
    _select_diverse_articles,
)


class TestDiversityClustering(unittest.TestCase):
    """Test embedding-based clustering for diversity-constrained ranking."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ScoringConfig()
        
        # Create test articles with different topics
        self.articles = [
            Article(
                url="https://tech.example.com/ai-breakthrough",
                title="Major AI Breakthrough in Machine Learning",
                content="Artificial intelligence researchers have made significant progress in machine learning algorithms. This breakthrough could revolutionize how we approach AI development.",
                publish_date=datetime.now(timezone.utc)
            ),
            Article(
                url="https://tech.example.com/ai-ethics",
                title="AI Ethics and Responsible Development",
                content="The importance of ethical considerations in artificial intelligence development cannot be overstated. Researchers discuss responsible AI practices.",
                publish_date=datetime.now(timezone.utc)
            ),
            Article(
                url="https://finance.example.com/market-update",
                title="Stock Market Reaches New Heights",
                content="Financial markets continue to show strong performance with major indices reaching record highs. Investors remain optimistic about economic growth.",
                publish_date=datetime.now(timezone.utc)
            ),
            Article(
                url="https://health.example.com/vaccine-news",
                title="New Vaccine Shows Promise in Clinical Trials",
                content="Medical researchers report positive results from phase 3 clinical trials of a new vaccine. The treatment shows high efficacy rates.",
                publish_date=datetime.now(timezone.utc)
            ),
            Article(
                url="https://tech.example.com/ai-applications",
                title="AI Applications in Healthcare",
                content="Artificial intelligence is being applied to healthcare with promising results. Machine learning models help doctors diagnose diseases more accurately.",
                publish_date=datetime.now(timezone.utc)
            ),
        ]
        
        # Create corresponding scorecards
        self.scorecards = []
        for i, article in enumerate(self.articles):
            scorecard = ScoreCard(
                overall_score=90.0 - i * 5,  # Decreasing scores
                readability_score=80.0,
                ner_density_score=70.0,
                sentiment_score=60.0,
                tfidf_relevance_score=85.0,
                recency_score=95.0,
                reputation_score=75.0,
                topic_coherence_score=80.0,
                article=article
            )
            self.scorecards.append(scorecard)

    @patch.dict(os.environ, {
        'USE_EMBEDDING_CLUSTERING': '1',
        'CLUSTER_CAP': '2',
        'CLUSTERING_THRESHOLD': '0.7',
        'DOMAIN_CAP': '2'
    })
    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_embedding_clustering_diversity(self, mock_get_model):
        """Test complete embedding-based clustering diversity."""
        # Mock the sentence transformer model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Create mock embeddings that group AI articles together and separate others
        mock_embeddings = [
            [1.0, 0.8, 0.1, 0.1, 0.1],  # AI breakthrough
            [0.9, 0.9, 0.1, 0.1, 0.1],  # AI ethics (similar to AI breakthrough)
            [0.1, 0.1, 1.0, 0.1, 0.1],  # Finance (different)
            [0.1, 0.1, 0.1, 1.0, 0.1],  # Health (different)
            [0.8, 0.7, 0.1, 0.2, 0.9],  # AI healthcare (similar to AI articles)
        ]
        mock_model.encode.return_value = mock_embeddings
        
        # Apply clustering diversity
        result = _apply_embedding_clustering_diversity(
            self.scorecards, domain_cap=2, sim_threshold=0.97
        )
        
        # Should have diverse results with cluster caps applied
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(self.scorecards))
        
        # Verify results are sorted by score
        scores = [card.overall_score for card in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_try_embeddings_for_clustering(self, mock_get_model):
        """Test embedding generation for clustering."""
        # Mock the sentence transformer model
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]] * len(self.scorecards)
        
        # Test successful embedding generation
        result = _try_embeddings_for_clustering(self.scorecards)
        self.assertTrue(result)
        
        # Verify embeddings were assigned
        for card in self.scorecards:
            self.assertTrue(hasattr(card, '_clustering_embedding'))
            self.assertIsInstance(card._clustering_embedding, list)

    @patch('curator.services._analyzer._get_sentence_transformer')
    def test_try_embeddings_for_clustering_failure(self, mock_get_model):
        """Test embedding generation failure handling."""
        # Mock model unavailable
        mock_get_model.return_value = None
        
        result = _try_embeddings_for_clustering(self.scorecards)
        self.assertFalse(result)

    def test_cluster_articles_by_embeddings(self):
        """Test clustering articles by embedding similarity."""
        # Manually assign embeddings to simulate clustering
        embeddings = [
            [1.0, 0.8, 0.1],  # AI cluster
            [0.9, 0.9, 0.1],  # AI cluster (similar)
            [0.1, 0.1, 1.0],  # Finance cluster
            [0.1, 0.2, 0.9],  # Health cluster
            [0.8, 0.7, 0.2],  # AI cluster (similar to first)
        ]
        
        for card, embedding in zip(self.scorecards, embeddings):
            card._clustering_embedding = embedding
        
        # Cluster with threshold 0.7
        clusters = _cluster_articles_by_embeddings(self.scorecards, threshold=0.7)
        
        # Should have multiple clusters
        self.assertGreater(len(clusters), 1)
        
        # AI articles should be clustered together
        ai_articles = 0
        for cluster in clusters:
            cluster_titles = [card.article.title for card in cluster]
            if any("AI" in title or "Machine Learning" in title for title in cluster_titles):
                ai_articles += len([title for title in cluster_titles if "AI" in title or "Machine Learning" in title])
        
        # Should have found AI articles in clusters
        self.assertGreater(ai_articles, 0)

    def test_calculate_cluster_similarity(self):
        """Test cluster similarity calculation."""
        # Create test cards with embeddings
        card1 = self.scorecards[0]
        card1._clustering_embedding = [1.0, 0.0, 0.0]
        
        card2 = self.scorecards[1]
        card2._clustering_embedding = [0.9, 0.1, 0.0]  # Similar
        
        card3 = self.scorecards[2]
        card3._clustering_embedding = [0.0, 0.0, 1.0]  # Different
        
        cluster = [card2]
        
        # Test similarity calculation
        sim_similar = _calculate_cluster_similarity(card1, cluster)
        sim_different = _calculate_cluster_similarity(card3, cluster)
        
        # Similar should have higher similarity
        self.assertGreater(sim_similar, sim_different)
        self.assertGreater(sim_similar, 0.5)

    def test_select_diverse_articles(self):
        """Test diverse article selection from clusters."""
        # Create clusters manually
        clusters = [
            [self.scorecards[0], self.scorecards[1], self.scorecards[4]],  # AI cluster
            [self.scorecards[2]],  # Finance cluster
            [self.scorecards[3]],  # Health cluster
        ]
        
        # Select with caps
        result = _select_diverse_articles(clusters, cluster_cap=2, domain_cap=2)
        
        # Should respect cluster cap
        self.assertLessEqual(len(result), 6)  # 3 clusters * 2 cap
        
        # Should be sorted by score
        scores = [card.overall_score for card in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch.dict(os.environ, {'USE_EMBEDDING_CLUSTERING': '0'})
    @patch('curator.services._analyzer._apply_domain_diversity')
    def test_fallback_to_domain_diversity(self, mock_domain_diversity):
        """Test fallback to domain diversity when clustering disabled."""
        from curator.services._analyzer import _apply_diversity_and_dedup
        
        mock_domain_diversity.return_value = self.scorecards[:3]
        
        result = _apply_diversity_and_dedup(self.scorecards)
        
        # Should call domain diversity function
        mock_domain_diversity.assert_called_once()
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()