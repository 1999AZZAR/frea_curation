"""
Unit tests for data models and configuration.

Tests the core data structures and validation logic
for the AI Content Curator application.
"""

import unittest
from datetime import datetime, timedelta
from models import Article, Entity, ScoreCard, ScoringConfig


class TestEntity(unittest.TestCase):
    """Test cases for the Entity data model."""
    
    def test_valid_entity_creation(self):
        """Test creating a valid entity."""
        entity = Entity(text="Apple Inc.", label="ORG", confidence=0.95)
        self.assertEqual(entity.text, "Apple Inc.")
        self.assertEqual(entity.label, "ORG")
        self.assertEqual(entity.confidence, 0.95)
    
    def test_entity_confidence_validation(self):
        """Test entity confidence validation."""
        # Valid confidence values
        Entity(text="Test", label="PERSON", confidence=0.0)
        Entity(text="Test", label="PERSON", confidence=1.0)
        Entity(text="Test", label="PERSON", confidence=0.5)
        
        # Invalid confidence values
        with self.assertRaises(ValueError):
            Entity(text="Test", label="PERSON", confidence=-0.1)
        
        with self.assertRaises(ValueError):
            Entity(text="Test", label="PERSON", confidence=1.1)
        
        with self.assertRaises(ValueError):
            Entity(text="Test", label="PERSON", confidence="invalid")
    
    def test_entity_text_validation(self):
        """Test entity text validation."""
        with self.assertRaises(ValueError):
            Entity(text="", label="PERSON", confidence=0.5)
        
        with self.assertRaises(ValueError):
            Entity(text="   ", label="PERSON", confidence=0.5)
    
    def test_entity_label_validation(self):
        """Test entity label validation."""
        with self.assertRaises(ValueError):
            Entity(text="Test", label="", confidence=0.5)
        
        with self.assertRaises(ValueError):
            Entity(text="Test", label="   ", confidence=0.5)


class TestArticle(unittest.TestCase):
    """Test cases for the Article data model."""
    
    def test_valid_article_creation(self):
        """Test creating a valid article."""
        article = Article(
            url="https://example.com/article",
            title="Test Article",
            author="John Doe",
            publish_date=datetime.now(),
            content="This is test content.",
            summary="Test summary"
        )
        self.assertEqual(article.url, "https://example.com/article")
        self.assertEqual(article.title, "Test Article")
        self.assertEqual(article.author, "John Doe")
        self.assertEqual(article.content, "This is test content.")
        self.assertEqual(article.summary, "Test summary")
        self.assertIsInstance(article.entities, list)
    
    def test_article_minimal_creation(self):
        """Test creating article with minimal required fields."""
        article = Article(url="https://example.com/article")
        self.assertEqual(article.url, "https://example.com/article")
        self.assertEqual(article.title, "")
        self.assertEqual(article.author, "")
        self.assertIsNone(article.publish_date)
        self.assertEqual(article.content, "")
        self.assertEqual(article.summary, "")
        self.assertEqual(article.entities, [])
    
    def test_article_url_validation(self):
        """Test article URL validation."""
        with self.assertRaises(ValueError):
            Article(url="")
        
        with self.assertRaises(ValueError):
            Article(url="   ")
    
    def test_article_with_entities(self):
        """Test article with entities."""
        entities = [
            Entity(text="Apple", label="ORG", confidence=0.9),
            Entity(text="Tim Cook", label="PERSON", confidence=0.8)
        ]
        article = Article(
            url="https://example.com/article",
            entities=entities
        )
        self.assertEqual(len(article.entities), 2)
        self.assertEqual(article.entities[0].text, "Apple")
        self.assertEqual(article.entities[1].text, "Tim Cook")


class TestScoreCard(unittest.TestCase):
    """Test cases for the ScoreCard data model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_article = Article(url="https://example.com/test")
    
    def test_valid_scorecard_creation(self):
        """Test creating a valid scorecard."""
        scorecard = ScoreCard(
            overall_score=85.5,
            readability_score=80.0,
            ner_density_score=90.0,
            sentiment_score=75.0,
            tfidf_relevance_score=88.0,
            recency_score=92.0,
            article=self.sample_article
        )
        self.assertEqual(scorecard.overall_score, 85.5)
        self.assertEqual(scorecard.readability_score, 80.0)
        self.assertEqual(scorecard.article, self.sample_article)
    
    def test_scorecard_score_validation(self):
        """Test scorecard score validation."""
        # Valid scores (0-100 range)
        ScoreCard(
            overall_score=0.0,
            readability_score=50.0,
            ner_density_score=100.0,
            sentiment_score=25.5,
            tfidf_relevance_score=75.8,
            recency_score=99.9,
            article=self.sample_article
        )
        
        # Invalid scores (outside 0-100 range)
        with self.assertRaises(ValueError):
            ScoreCard(
                overall_score=-1.0,
                readability_score=50.0,
                ner_density_score=50.0,
                sentiment_score=50.0,
                tfidf_relevance_score=50.0,
                recency_score=50.0,
                article=self.sample_article
            )
        
        with self.assertRaises(ValueError):
            ScoreCard(
                overall_score=101.0,
                readability_score=50.0,
                ner_density_score=50.0,
                sentiment_score=50.0,
                tfidf_relevance_score=50.0,
                recency_score=50.0,
                article=self.sample_article
            )
        
        # Invalid score types
        with self.assertRaises(ValueError):
            ScoreCard(
                overall_score="invalid",
                readability_score=50.0,
                ner_density_score=50.0,
                sentiment_score=50.0,
                tfidf_relevance_score=50.0,
                recency_score=50.0,
                article=self.sample_article
            )


class TestScoringConfig(unittest.TestCase):
    """Test cases for the ScoringConfig data model."""
    
    def test_valid_config_creation(self):
        """Test creating a valid scoring configuration."""
        config = ScoringConfig()
        self.assertEqual(config.readability_weight, 0.2)
        self.assertEqual(config.ner_density_weight, 0.2)
        self.assertEqual(config.sentiment_weight, 0.15)
        self.assertEqual(config.tfidf_relevance_weight, 0.25)
        self.assertEqual(config.recency_weight, 0.2)
        self.assertEqual(config.min_word_count, 300)
        self.assertEqual(config.max_articles_per_topic, 20)
    
    def test_custom_config_creation(self):
        """Test creating a custom scoring configuration."""
        config = ScoringConfig(
            readability_weight=0.3,
            ner_density_weight=0.2,
            sentiment_weight=0.1,
            tfidf_relevance_weight=0.2,
            recency_weight=0.2,
            min_word_count=500,
            max_articles_per_topic=10
        )
        self.assertEqual(config.readability_weight, 0.3)
        self.assertEqual(config.min_word_count, 500)
        self.assertEqual(config.max_articles_per_topic, 10)
    
    def test_config_weight_sum_validation(self):
        """Test that weights must sum to 1.0."""
        # Valid weight sum (exactly 1.0)
        ScoringConfig(
            readability_weight=0.2,
            ner_density_weight=0.2,
            sentiment_weight=0.2,
            tfidf_relevance_weight=0.2,
            recency_weight=0.2
        )
        
        # Invalid weight sum (too high)
        with self.assertRaises(ValueError):
            ScoringConfig(
                readability_weight=0.3,
                ner_density_weight=0.3,
                sentiment_weight=0.3,
                tfidf_relevance_weight=0.3,
                recency_weight=0.3
            )
        
        # Invalid weight sum (too low)
        with self.assertRaises(ValueError):
            ScoringConfig(
                readability_weight=0.1,
                ner_density_weight=0.1,
                sentiment_weight=0.1,
                tfidf_relevance_weight=0.1,
                recency_weight=0.1
            )
    
    def test_config_individual_weight_validation(self):
        """Test individual weight validation."""
        # Invalid negative weight
        with self.assertRaises(ValueError):
            ScoringConfig(
                readability_weight=-0.1,
                ner_density_weight=0.3,
                sentiment_weight=0.3,
                tfidf_relevance_weight=0.3,
                recency_weight=0.2
            )
        
        # Invalid weight > 1.0
        with self.assertRaises(ValueError):
            ScoringConfig(
                readability_weight=1.1,
                ner_density_weight=0.0,
                sentiment_weight=0.0,
                tfidf_relevance_weight=0.0,
                recency_weight=0.0
            )
    
    def test_config_parameter_validation(self):
        """Test parameter validation."""
        # Invalid min_word_count
        with self.assertRaises(ValueError):
            ScoringConfig(min_word_count=-1)
        
        with self.assertRaises(ValueError):
            ScoringConfig(min_word_count="invalid")
        
        # Invalid max_articles_per_topic
        with self.assertRaises(ValueError):
            ScoringConfig(max_articles_per_topic=0)
        
        with self.assertRaises(ValueError):
            ScoringConfig(max_articles_per_topic="invalid")


if __name__ == '__main__':
    unittest.main()