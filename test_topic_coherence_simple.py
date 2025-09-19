#!/usr/bin/env python3
"""
Simple test script for topic coherence functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_topic_coherence_basic():
    """Test basic topic coherence functionality without YAKE."""
    from curator.core.topic_coherence import (
        normalize_text_for_matching,
        extract_query_keywords,
        calculate_keyword_coverage_ratio,
        compute_topic_coherence_score
    )
    
    print("Testing text normalization...")
    normalized = normalize_text_for_matching("Hello World! AI & Machine-Learning.")
    expected = "hello world ai machine learning"
    assert normalized == expected, f"Expected '{expected}', got '{normalized}'"
    print("✓ Text normalization works")
    
    print("Testing query keyword extraction...")
    keywords = extract_query_keywords("machine learning AI")
    expected = {"machine", "learning", "ai"}
    assert keywords == expected, f"Expected {expected}, got {keywords}"
    print("✓ Query keyword extraction works")
    
    print("Testing keyword coverage ratio...")
    article = "This article discusses machine learning and AI applications."
    query = "machine learning AI"
    coverage = calculate_keyword_coverage_ratio(article, query)
    # Should find all keywords: machine, learning, ai
    assert coverage == 1.0, f"Expected 1.0, got {coverage}"
    print("✓ Keyword coverage calculation works")
    
    print("Testing topic coherence score...")
    score = compute_topic_coherence_score(article, "Machine Learning Guide", query)
    # Should get some score > 0 even without YAKE
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"
    assert score > 0, f"Should get some score for matching keywords, got {score}"
    print(f"✓ Topic coherence score: {score}")
    
    print("All basic topic coherence tests passed!")

def test_models_with_topic_coherence():
    """Test that models work with the new topic coherence score."""
    from curator.core.models import ScoreCard, ScoringConfig, Article
    from datetime import datetime
    
    print("Testing ScoringConfig with topic coherence weight...")
    config = ScoringConfig()
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
    assert 0.99 <= total_weight <= 1.01, f"Weights should sum to 1.0, got {total_weight}"
    print(f"✓ ScoringConfig weights sum to {total_weight}")
    
    print("Testing ScoreCard with topic coherence score...")
    article = Article(url="https://example.com", title="Test", content="Test content")
    scorecard = ScoreCard(
        overall_score=85.0,
        readability_score=80.0,
        ner_density_score=75.0,
        sentiment_score=70.0,
        tfidf_relevance_score=90.0,
        recency_score=95.0,
        reputation_score=75.0,
        topic_coherence_score=85.0,
        article=article
    )
    assert scorecard.topic_coherence_score == 85.0
    print("✓ ScoreCard with topic coherence score works")
    
    print("All model tests passed!")

if __name__ == "__main__":
    try:
        test_topic_coherence_basic()
        test_models_with_topic_coherence()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)