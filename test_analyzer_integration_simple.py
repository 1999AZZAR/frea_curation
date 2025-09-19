#!/usr/bin/env python3
"""
Simple test script for analyzer integration with topic coherence.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_calculate_composite_score():
    """Test that composite score calculation includes topic coherence."""
    from curator.services._analyzer import calculate_composite_score
    from curator.core.models import ScoringConfig
    
    print("Testing composite score calculation with topic coherence...")
    
    config = ScoringConfig()
    metrics = {
        "readability": 80.0,
        "ner_density": 75.0,
        "sentiment": 70.0,
        "tfidf_relevance": 90.0,
        "recency": 95.0,
        "reputation": 75.0,
        "topic_coherence": 85.0
    }
    
    overall = calculate_composite_score(metrics, config)
    
    # Calculate expected score manually
    expected = (
        80.0 * config.readability_weight +
        75.0 * config.ner_density_weight +
        70.0 * config.sentiment_weight +
        90.0 * config.tfidf_relevance_weight +
        95.0 * config.recency_weight +
        75.0 * config.reputation_weight +
        85.0 * config.topic_coherence_weight
    )
    expected = round(expected, 2)
    
    print(f"Calculated overall score: {overall}")
    print(f"Expected overall score: {expected}")
    
    assert overall == expected, f"Expected {expected}, got {overall}"
    assert 0 <= overall <= 100, f"Score should be 0-100, got {overall}"
    
    print("✓ Composite score calculation includes topic coherence")

def test_analyze_article_mock():
    """Test analyze_article with mocked dependencies."""
    from curator.services._analyzer import analyze_article
    from curator.core.models import Article, ScoringConfig
    from datetime import datetime
    
    print("Testing analyze_article with topic coherence...")
    
    # Mock the parse_article function
    mock_article = Article(
        url="https://example.com/test",
        title="Machine Learning Guide",
        author="Test Author",
        publish_date=datetime.now(),
        content="This comprehensive guide covers machine learning algorithms and AI applications."
    )
    
    with patch('curator.services._analyzer.parse_article') as mock_parse:
        mock_parse.return_value = mock_article
        
        # Mock NLP components
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = [Mock()]  # Mock entity
        mock_nlp.return_value = mock_doc
        
        mock_vader = Mock()
        mock_vader.polarity_scores.return_value = {"compound": 0.1}
        
        # Call analyze_article
        result = analyze_article(
            url="https://example.com/test",
            query="machine learning AI",
            config=ScoringConfig(),
            nlp=mock_nlp,
            vader_analyzer=mock_vader
        )
        
        # Check that result has topic coherence score
        assert hasattr(result, 'topic_coherence_score'), "Missing topic_coherence_score"
        assert 0 <= result.topic_coherence_score <= 100, f"Invalid topic coherence score: {result.topic_coherence_score}"
        
        # Check that overall score is calculated correctly
        assert 0 <= result.overall_score <= 100, f"Invalid overall score: {result.overall_score}"
        
        print(f"✓ analyze_article returned topic coherence score: {result.topic_coherence_score}")
        print(f"✓ Overall score: {result.overall_score}")

if __name__ == "__main__":
    try:
        test_calculate_composite_score()
        test_analyze_article_mock()
        print("\n🎉 All analyzer integration tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)