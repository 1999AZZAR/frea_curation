#!/usr/bin/env python3
"""
Debug script for keyword coverage.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curator.core.topic_coherence import (
    normalize_text_for_matching,
    extract_query_keywords,
    calculate_keyword_coverage_ratio
)

article = "This article discusses machine learning and artificial intelligence applications."
query = "machine learning AI"

print(f"Article: {article}")
print(f"Query: {query}")

article_normalized = normalize_text_for_matching(article)
print(f"Article normalized: '{article_normalized}'")

query_keywords = extract_query_keywords(query)
print(f"Query keywords: {query_keywords}")

print("\nChecking each keyword:")
for keyword in query_keywords:
    found = keyword in article_normalized
    print(f"  '{keyword}' in article: {found}")

coverage = calculate_keyword_coverage_ratio(article, query)
print(f"\nCoverage ratio: {coverage}")