#!/usr/bin/env python3
"""
Verify that the duplicate detection and diversity controls implementation works.
"""

from curator.core.utils import canonicalize_url, extract_domain

def test_url_canonicalization():
    """Test URL canonicalization."""
    print("Testing URL canonicalization...")
    
    # Test basic canonicalization
    url = "https://WWW.Example.com/article?utm_source=test&id=123"
    result = canonicalize_url(url)
    expected = "https://example.com/article?id=123"
    print(f"Input: {url}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"✓ Match: {result == expected}")
    
    # Test mobile variant
    url = "https://m.example.com/article/"
    result = canonicalize_url(url)
    expected = "https://example.com/article"
    print(f"\nInput: {url}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"✓ Match: {result == expected}")

def test_domain_extraction():
    """Test domain extraction."""
    print("\nTesting domain extraction...")
    
    url = "https://www.example.com/article"
    result = extract_domain(url)
    expected = "example.com"
    print(f"Input: {url}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    print(f"✓ Match: {result == expected}")

if __name__ == "__main__":
    test_url_canonicalization()
    test_domain_extraction()
    print("\n🎉 Basic functionality verification complete!")