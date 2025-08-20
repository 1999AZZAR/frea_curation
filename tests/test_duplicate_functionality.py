#!/usr/bin/env python3
"""
Simple test script to verify duplicate detection and diversity controls functionality.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, '.')

from curator.core.models import Article, ScoreCard, ScoringConfig
from curator.services._analyzer import _apply_diversity_and_dedup, _extract_domain
from curator.services._news_source import NewsSource


def test_url_canonicalization():
    """Test URL canonicalization functionality."""
    print("Testing URL canonicalization...")
    
    test_cases = [
        ("https://WWW.Example.com/article", "https://example.com/article"),
        ("https://example.com/article?utm_source=test&id=123", "https://example.com/article?id=123"),
        ("https://m.example.com/article/", "https://example.com/article"),
        ("https://example.com/article/amp", "https://example.com/article"),
        ("example.com/article", "https://example.com/article"),
    ]
    
    for input_url, expected in test_cases:
        result = NewsSource._canonicalize_url(input_url)
        print(f"  {input_url} -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ URL canonicalization tests passed")


def test_domain_extraction():
    """Test domain extraction functionality."""
    print("\nTesting domain extraction...")
    
    test_cases = [
        ("https://example.com/article", "example.com"),
        ("https://www.example.com/article", "example.com"),
        ("https://news.example.com/article", "news.example.com"),
        ("invalid-url", ""),
    ]
    
    for url, expected in test_cases:
        result = _extract_domain(url)
        print(f"  {url} -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Domain extraction tests passed")


def test_diversity_controls():
    """Test diversity controls and deduplication."""
    print("\nTesting diversity controls...")
    
    # Create test articles from same domain
    articles = []
    for i in range(5):
        article = Article(
            url=f"https://example.com/article{i}",
            title=f"Article {i}",
            content=f"Content about technology {i}",
            publish_date=datetime.now()
        )
        scorecard = ScoreCard(
            overall_score=90.0 - i,
            readability_score=80.0,
            ner_density_score=70.0,
            sentiment_score=60.0,
            tfidf_relevance_score=85.0,
            recency_score=95.0,
            article=article
        )
        articles.append(scorecard)
    
    # Test domain cap
    result = _apply_diversity_and_dedup(articles, domain_cap=2)
    print(f"  Domain cap test: {len(articles)} articles -> {len(result)} articles (cap=2)")
    assert len(result) == 2, f"Expected 2 articles, got {len(result)}"
    
    # Should keep highest scoring articles
    assert result[0].overall_score == 90.0
    assert result[1].overall_score == 89.0
    
    print("✓ Diversity controls tests passed")


def test_basic_deduplication():
    """Test basic deduplication functionality."""
    print("\nTesting basic deduplication...")
    
    # Create articles with same title
    articles = [
        ScoreCard(
            overall_score=90.0,
            readability_score=80.0,
            ner_density_score=70.0,
            sentiment_score=60.0,
            tfidf_relevance_score=85.0,
            recency_score=95.0,
            article=Article(
                url="https://example.com/article1",
                title="Same Title",
                content="Different content 1"
            )
        ),
        ScoreCard(
            overall_score=85.0,
            readability_score=80.0,
            ner_density_score=70.0,
            sentiment_score=60.0,
            tfidf_relevance_score=85.0,
            recency_score=95.0,
            article=Article(
                url="https://different.com/article2",
                title="Same Title",
                content="Different content 2"
            )
        )
    ]
    
    result = _apply_diversity_and_dedup(articles, domain_cap=10)
    print(f"  Title dedup test: {len(articles)} articles -> {len(result)} articles")
    assert len(result) == 1, f"Expected 1 article after dedup, got {len(result)}"
    assert result[0].overall_score == 90.0, "Should keep higher scoring article"
    
    print("✓ Basic deduplication tests passed")


if __name__ == "__main__":
    try:
        test_url_canonicalization()
        test_domain_extraction()
        test_diversity_controls()
        test_basic_deduplication()
        print("\n🎉 All tests passed! Duplicate detection and diversity controls are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)