#!/usr/bin/env python3
"""
Example demonstrating Redis caching functionality in the AI Content Curator.

This script shows how the caching layer improves performance by avoiding
redundant article parsing and scoring operations.
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curator.services._analyzer import analyze_article, batch_analyze
from curator.core.cache import cache_manager, get_cached_article, get_cached_scorecard
from curator.core.config import load_scoring_config


def demonstrate_single_article_caching():
    """Demonstrate caching for single article analysis."""
    print("=== Single Article Caching Demo ===")
    
    # Use a real article URL for demonstration
    url = "https://www.bbc.com/news/technology"
    query = "artificial intelligence"
    
    print(f"Analyzing article: {url}")
    print(f"Query: {query}")
    
    # First analysis - will parse and cache
    print("\n1. First analysis (cold cache):")
    start_time = time.time()
    try:
        result1 = analyze_article(url, query=query)
        elapsed1 = time.time() - start_time
        print(f"   Overall Score: {result1.overall_score}")
        print(f"   Time taken: {elapsed1:.2f} seconds")
        print(f"   Article title: {result1.article.title[:50]}...")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Second analysis - should use cache
    print("\n2. Second analysis (warm cache):")
    start_time = time.time()
    try:
        result2 = analyze_article(url, query=query)
        elapsed2 = time.time() - start_time
        print(f"   Overall Score: {result2.overall_score}")
        print(f"   Time taken: {elapsed2:.2f} seconds")
        print(f"   Speed improvement: {elapsed1/elapsed2:.1f}x faster")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Verify results are identical
    if 'result1' in locals() and 'result2' in locals():
        print(f"\n3. Cache verification:")
        print(f"   Scores identical: {result1.overall_score == result2.overall_score}")
        print(f"   Articles identical: {result1.article.url == result2.article.url}")


def demonstrate_batch_caching():
    """Demonstrate caching for batch analysis."""
    print("\n=== Batch Analysis Caching Demo ===")
    
    # Use multiple URLs for demonstration
    urls = [
        "https://www.bbc.com/news/technology",
        "https://www.reuters.com/technology/",
        "https://techcrunch.com/"
    ]
    query = "machine learning"
    
    print(f"Analyzing {len(urls)} articles with query: {query}")
    
    # First batch analysis
    print("\n1. First batch analysis (cold cache):")
    start_time = time.time()
    try:
        results1 = batch_analyze(urls, query=query)
        elapsed1 = time.time() - start_time
        print(f"   Successfully analyzed: {len(results1)} articles")
        print(f"   Time taken: {elapsed1:.2f} seconds")
        if results1:
            avg_score = sum(r.overall_score for r in results1) / len(results1)
            print(f"   Average score: {avg_score:.1f}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Second batch analysis - should use cache
    print("\n2. Second batch analysis (warm cache):")
    start_time = time.time()
    try:
        results2 = batch_analyze(urls, query=query)
        elapsed2 = time.time() - start_time
        print(f"   Successfully analyzed: {len(results2)} articles")
        print(f"   Time taken: {elapsed2:.2f} seconds")
        if elapsed2 > 0:
            print(f"   Speed improvement: {elapsed1/elapsed2:.1f}x faster")
    except Exception as e:
        print(f"   Error: {e}")


def demonstrate_cache_stats():
    """Show cache statistics and health information."""
    print("\n=== Cache Statistics ===")
    
    if not cache_manager.is_available():
        print("Redis cache is not available.")
        print("To enable caching:")
        print("1. Install Redis: apt-get install redis-server")
        print("2. Start Redis: redis-server")
        print("3. Set REDIS_HOST=localhost in .env")
        return
    
    stats = cache_manager.get_cache_stats()
    
    print("Cache Status:")
    print(f"  Available: {stats.get('available', False)}")
    
    if stats.get('available'):
        print(f"  Cached articles: {stats.get('article_count', 0)}")
        print(f"  Cached scorecards: {stats.get('scorecard_count', 0)}")
        print(f"  Total cache keys: {stats.get('total_keys', 0)}")
        print(f"  Memory usage: {stats.get('memory_usage', 'unknown')}")
        print(f"  Connected clients: {stats.get('connected_clients', 0)}")
        print(f"  Uptime: {stats.get('uptime_seconds', 0)} seconds")
    else:
        print(f"  Error: {stats.get('error', 'Unknown error')}")


def demonstrate_cache_keys():
    """Show how cache keys work with different queries."""
    print("\n=== Cache Key Demonstration ===")
    
    url = "https://example.com/test-article"
    
    # Show how different queries create different cache keys
    queries = ["", "AI", "machine learning", "artificial intelligence"]
    
    print("Cache key behavior:")
    for query in queries:
        # Check if scorecard is cached (will be None for this demo URL)
        cached = get_cached_scorecard(url, query)
        query_display = query if query else "(empty)"
        print(f"  Query '{query_display}': {'Cached' if cached else 'Not cached'}")
    
    print("\nNote: Different queries create separate cache entries")
    print("This allows relevance scores to be cached per query.")


def main():
    """Run all cache demonstrations."""
    print("AI Content Curator - Redis Caching Demo")
    print("=" * 50)
    
    # Show cache configuration
    print(f"Cache configuration:")
    print(f"  Redis available: {cache_manager.is_available()}")
    if cache_manager.is_available():
        print(f"  Article TTL: {cache_manager.article_ttl} seconds")
        print(f"  Scorecard TTL: {cache_manager.scorecard_ttl} seconds")
    
    # Run demonstrations
    demonstrate_cache_stats()
    demonstrate_cache_keys()
    
    # Only run network-dependent demos if Redis is available
    if cache_manager.is_available():
        print("\nNote: The following demos require internet access and may take time...")
        
        # Uncomment to run network-dependent demos
        # demonstrate_single_article_caching()
        # demonstrate_batch_caching()
        
        print("\nTo run full network demos, uncomment the demo calls in main()")
    else:
        print("\nTo see full caching benefits:")
        print("1. Install and start Redis")
        print("2. Configure Redis connection in .env")
        print("3. Re-run this script")


if __name__ == "__main__":
    main()