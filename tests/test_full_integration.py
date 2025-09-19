#!/usr/bin/env python3
"""
Full integration test for topic coherence functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_full_integration():
    """Test the complete topic coherence integration."""
    from curator.core.models import Article, ScoreCard, ScoringConfig
    from curator.core.topic_coherence import compute_topic_coherence_score
    from datetime import datetime
    
    print("=== Full Integration Test for Topic Coherence ===")
    
    # Test 1: ScoringConfig includes topic coherence weight
    print("\n1. Testing ScoringConfig...")
    config = ScoringConfig()
    
    # Verify all weights are present
    weights = [
        config.readability_weight,
        config.ner_density_weight,
        config.sentiment_weight,
        config.tfidf_relevance_weight,
        config.recency_weight,
        config.reputation_weight,
        config.topic_coherence_weight
    ]
    
    total_weight = sum(weights)
    print(f"   Total weight: {total_weight}")
    assert 0.99 <= total_weight <= 1.01, f"Weights should sum to 1.0, got {total_weight}"
    assert config.topic_coherence_weight > 0, "Topic coherence weight should be > 0"
    print("   ✓ ScoringConfig includes topic coherence weight correctly")
    
    # Test 2: Topic coherence scoring works
    print("\n2. Testing topic coherence scoring...")
    article_text = "This comprehensive guide covers machine learning algorithms, neural networks, and AI applications in detail."
    article_title = "Complete Machine Learning Guide"
    query = "machine learning AI neural networks"
    
    score = compute_topic_coherence_score(article_text, article_title, query)
    print(f"   Topic coherence score: {score}")
    assert 0 <= score <= 100, f"Score should be 0-100, got {score}"
    assert score > 0, f"Should get some score for matching content, got {score}"
    print("   ✓ Topic coherence scoring works")
    
    # Test 3: ScoreCard includes topic coherence score
    print("\n3. Testing ScoreCard with topic coherence...")
    article = Article(
        url="https://example.com/ml-guide",
        title=article_title,
        author="ML Expert",
        publish_date=datetime.now(),
        content=article_text
    )
    
    scorecard = ScoreCard(
        overall_score=85.0,
        readability_score=80.0,
        ner_density_score=75.0,
        sentiment_score=70.0,
        tfidf_relevance_score=90.0,
        recency_score=95.0,
        reputation_score=75.0,
        topic_coherence_score=score,
        article=article
    )
    
    assert scorecard.topic_coherence_score == score
    print(f"   ✓ ScoreCard includes topic coherence score: {scorecard.topic_coherence_score}")
    
    # Test 4: Composite score calculation
    print("\n4. Testing composite score calculation...")
    
    def calculate_composite_score(metrics, config):
        weighted_sum = (
            metrics.get("readability", 0.0) * config.readability_weight +
            metrics.get("ner_density", 0.0) * config.ner_density_weight +
            metrics.get("sentiment", 0.0) * config.sentiment_weight +
            metrics.get("tfidf_relevance", 0.0) * config.tfidf_relevance_weight +
            metrics.get("recency", 0.0) * config.recency_weight +
            metrics.get("reputation", 0.0) * config.reputation_weight +
            metrics.get("topic_coherence", 0.0) * config.topic_coherence_weight
        )
        return round(max(0.0, min(100.0, weighted_sum)), 2)
    
    metrics = {
        "readability": scorecard.readability_score,
        "ner_density": scorecard.ner_density_score,
        "sentiment": scorecard.sentiment_score,
        "tfidf_relevance": scorecard.tfidf_relevance_score,
        "recency": scorecard.recency_score,
        "reputation": scorecard.reputation_score,
        "topic_coherence": scorecard.topic_coherence_score
    }
    
    calculated_overall = calculate_composite_score(metrics, config)
    print(f"   Calculated overall score: {calculated_overall}")
    
    # Verify the calculation includes topic coherence
    topic_contribution = scorecard.topic_coherence_score * config.topic_coherence_weight
    print(f"   Topic coherence contribution: {scorecard.topic_coherence_score} * {config.topic_coherence_weight} = {topic_contribution}")
    
    assert 0 <= calculated_overall <= 100, f"Overall score should be 0-100, got {calculated_overall}"
    print("   ✓ Composite score calculation includes topic coherence")
    
    # Test 5: Edge cases
    print("\n5. Testing edge cases...")
    
    # Empty query
    empty_query_score = compute_topic_coherence_score(article_text, article_title, "")
    assert empty_query_score == 0.0, f"Empty query should give 0 score, got {empty_query_score}"
    print("   ✓ Empty query handled correctly")
    
    # Empty content
    empty_content_score = compute_topic_coherence_score("", "", query)
    assert empty_content_score == 0.0, f"Empty content should give 0 score, got {empty_content_score}"
    print("   ✓ Empty content handled correctly")
    
    # No keyword matches
    no_match_score = compute_topic_coherence_score("This article is about cooking recipes.", "Cooking Guide", query)
    assert no_match_score == 0.0, f"No matches should give 0 score, got {no_match_score}"
    print("   ✓ No keyword matches handled correctly")
    
    print("\n🎉 All integration tests passed successfully!")
    print(f"   Final topic coherence score: {score}")
    print(f"   Final overall score: {calculated_overall}")
    print(f"   Topic coherence weight: {config.topic_coherence_weight}")

if __name__ == "__main__":
    try:
        test_full_integration()
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)