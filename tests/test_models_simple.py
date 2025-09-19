#!/usr/bin/env python3
"""
Simple test script for updated models.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_scoring_config():
    """Test that ScoringConfig works with the new topic coherence weight."""
    from curator.core.models import ScoringConfig
    
    print("Testing default ScoringConfig...")
    config = ScoringConfig()
    
    # Check that all weights are present
    assert hasattr(config, 'topic_coherence_weight'), "Missing topic_coherence_weight"
    
    # Check that weights sum to 1.0
    total_weight = (
        config.readability_weight +
        config.ner_density_weight +
        config.sentiment_weight +
        config.tfidf_relevance_weight +
        config.recency_weight +
        config.reputation_weight +
        config.topic_coherence_weight
    )
    
    print(f"Individual weights:")
    print(f"  readability_weight: {config.readability_weight}")
    print(f"  ner_density_weight: {config.ner_density_weight}")
    print(f"  sentiment_weight: {config.sentiment_weight}")
    print(f"  tfidf_relevance_weight: {config.tfidf_relevance_weight}")
    print(f"  recency_weight: {config.recency_weight}")
    print(f"  reputation_weight: {config.reputation_weight}")
    print(f"  topic_coherence_weight: {config.topic_coherence_weight}")
    print(f"  Total weight: {total_weight}")
    
    assert 0.99 <= total_weight <= 1.01, f"Weights should sum to 1.0, got {total_weight}"
    print("✓ ScoringConfig weights sum correctly")

def test_scorecard():
    """Test that ScoreCard works with the new topic coherence score."""
    from curator.core.models import ScoreCard, Article
    from datetime import datetime
    
    print("Testing ScoreCard with topic coherence score...")
    
    article = Article(
        url="https://example.com/test",
        title="Test Article",
        author="Test Author",
        publish_date=datetime.now(),
        content="This is test content for the article."
    )
    
    scorecard = ScoreCard(
        overall_score=85.5,
        readability_score=80.0,
        ner_density_score=75.0,
        sentiment_score=70.0,
        tfidf_relevance_score=90.0,
        recency_score=95.0,
        reputation_score=75.0,
        topic_coherence_score=85.0,
        article=article
    )
    
    # Check that all scores are accessible
    assert scorecard.overall_score == 85.5
    assert scorecard.readability_score == 80.0
    assert scorecard.ner_density_score == 75.0
    assert scorecard.sentiment_score == 70.0
    assert scorecard.tfidf_relevance_score == 90.0
    assert scorecard.recency_score == 95.0
    assert scorecard.reputation_score == 75.0
    assert scorecard.topic_coherence_score == 85.0
    assert scorecard.article == article
    
    print("✓ ScoreCard with topic coherence score works correctly")

def test_scorecard_validation():
    """Test ScoreCard validation with invalid scores."""
    from curator.core.models import ScoreCard, Article
    from datetime import datetime
    
    print("Testing ScoreCard validation...")
    
    article = Article(url="https://example.com/test", title="Test")
    
    # Test invalid score (> 100)
    try:
        ScoreCard(
            overall_score=150.0,  # Invalid
            readability_score=80.0,
            ner_density_score=75.0,
            sentiment_score=70.0,
            tfidf_relevance_score=90.0,
            recency_score=95.0,
            reputation_score=75.0,
            topic_coherence_score=85.0,
            article=article
        )
        assert False, "Should have raised ValueError for invalid score"
    except ValueError as e:
        print(f"✓ Correctly caught invalid score: {e}")
    
    # Test invalid score (< 0)
    try:
        ScoreCard(
            overall_score=85.0,
            readability_score=-10.0,  # Invalid
            ner_density_score=75.0,
            sentiment_score=70.0,
            tfidf_relevance_score=90.0,
            recency_score=95.0,
            reputation_score=75.0,
            topic_coherence_score=85.0,
            article=article
        )
        assert False, "Should have raised ValueError for negative score"
    except ValueError as e:
        print(f"✓ Correctly caught negative score: {e}")

if __name__ == "__main__":
    try:
        test_scoring_config()
        test_scorecard()
        test_scorecard_validation()
        print("\n🎉 All model tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)