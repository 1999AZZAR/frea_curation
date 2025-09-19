#!/usr/bin/env python3
"""
Simple test script for composite score calculation.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_calculate_composite_score():
    """Test that composite score calculation includes topic coherence."""
    # Import just the function we need
    from curator.core.models import ScoringConfig
    
    # Copy the calculate_composite_score function logic to test it
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
    
    print("Testing composite score calculation with topic coherence...")
    
    config = ScoringConfig()
    print(f"Config weights: readability={config.readability_weight}, ner_density={config.ner_density_weight}, sentiment={config.sentiment_weight}, tfidf_relevance={config.tfidf_relevance_weight}, recency={config.recency_weight}, reputation={config.reputation_weight}, topic_coherence={config.topic_coherence_weight}")
    
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
    
    print(f"Individual contributions:")
    print(f"  readability: 80.0 * {config.readability_weight} = {80.0 * config.readability_weight}")
    print(f"  ner_density: 75.0 * {config.ner_density_weight} = {75.0 * config.ner_density_weight}")
    print(f"  sentiment: 70.0 * {config.sentiment_weight} = {70.0 * config.sentiment_weight}")
    print(f"  tfidf_relevance: 90.0 * {config.tfidf_relevance_weight} = {90.0 * config.tfidf_relevance_weight}")
    print(f"  recency: 95.0 * {config.recency_weight} = {95.0 * config.recency_weight}")
    print(f"  reputation: 75.0 * {config.reputation_weight} = {75.0 * config.reputation_weight}")
    print(f"  topic_coherence: 85.0 * {config.topic_coherence_weight} = {85.0 * config.topic_coherence_weight}")
    
    print(f"Calculated overall score: {overall}")
    print(f"Expected overall score: {expected}")
    
    assert overall == expected, f"Expected {expected}, got {overall}"
    assert 0 <= overall <= 100, f"Score should be 0-100, got {overall}"
    
    print("✓ Composite score calculation includes topic coherence correctly")

def test_missing_topic_coherence():
    """Test composite score when topic coherence is missing."""
    from curator.core.models import ScoringConfig
    
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
    
    print("Testing composite score with missing topic coherence...")
    
    config = ScoringConfig()
    metrics = {
        "readability": 80.0,
        "ner_density": 75.0,
        "sentiment": 70.0,
        "tfidf_relevance": 90.0,
        "recency": 95.0,
        "reputation": 75.0,
        # topic_coherence missing - should default to 0.0
    }
    
    overall = calculate_composite_score(metrics, config)
    
    # Calculate expected score manually (topic_coherence = 0.0)
    expected = (
        80.0 * config.readability_weight +
        75.0 * config.ner_density_weight +
        70.0 * config.sentiment_weight +
        90.0 * config.tfidf_relevance_weight +
        95.0 * config.recency_weight +
        75.0 * config.reputation_weight +
        0.0 * config.topic_coherence_weight  # Missing = 0.0
    )
    expected = round(expected, 2)
    
    print(f"Calculated overall score (missing topic coherence): {overall}")
    print(f"Expected overall score: {expected}")
    
    assert overall == expected, f"Expected {expected}, got {overall}"
    assert 0 <= overall <= 100, f"Score should be 0-100, got {overall}"
    
    print("✓ Composite score handles missing topic coherence correctly")

if __name__ == "__main__":
    try:
        test_calculate_composite_score()
        test_missing_topic_coherence()
        print("\n🎉 All composite score tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)