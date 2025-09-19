#!/usr/bin/env python3
"""
Verification script for parser resilience improvements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from curator.services._parser import (
            parse_article, 
            ArticleParsingError, 
            ContentValidationError,
            USER_AGENTS,
            READABILITY_AVAILABLE,
            parse_with_readability_fallback,
            parse_with_fallback
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_enhanced_error_classes():
    """Test enhanced error classes."""
    try:
        from curator.services._parser import ArticleParsingError, ContentValidationError
        
        # Test ArticleParsingError
        error = ArticleParsingError(
            "Test error", 
            url="https://example.com", 
            attempt=1, 
            method="newspaper3k"
        )
        
        if hasattr(error, 'get_user_friendly_message'):
            friendly_msg = error.get_user_friendly_message()
            print(f"✓ ArticleParsingError user-friendly message: {friendly_msg[:50]}...")
        
        # Test ContentValidationError
        validation_error = ContentValidationError(
            "Content too short", 
            url="https://example.com", 
            word_count=50, 
            min_required=300
        )
        
        if hasattr(validation_error, 'get_user_friendly_message'):
            friendly_msg = validation_error.get_user_friendly_message()
            print(f"✓ ContentValidationError user-friendly message: {friendly_msg[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error class test failed: {e}")
        return False

def test_user_agent_expansion():
    """Test that user agents have been expanded."""
    try:
        from curator.services._parser import USER_AGENTS
        
        print(f"✓ User agents available: {len(USER_AGENTS)}")
        
        # Check for diversity
        chrome_count = sum(1 for ua in USER_AGENTS if 'Chrome' in ua)
        firefox_count = sum(1 for ua in USER_AGENTS if 'Firefox' in ua)
        safari_count = sum(1 for ua in USER_AGENTS if 'Safari' in ua)
        
        print(f"  - Chrome variants: {chrome_count}")
        print(f"  - Firefox variants: {firefox_count}")
        print(f"  - Safari variants: {safari_count}")
        
        if len(USER_AGENTS) >= 5:  # Should have at least 5 user agents
            print("✓ User agent diversity looks good")
            return True
        else:
            print("✗ Not enough user agent diversity")
            return False
            
    except Exception as e:
        print(f"✗ User agent test failed: {e}")
        return False

def test_readability_availability():
    """Test readability availability."""
    try:
        from curator.services._parser import READABILITY_AVAILABLE
        
        if READABILITY_AVAILABLE:
            print("✓ Readability-lxml is available")
            
            # Test the fallback function exists
            from curator.services._parser import parse_with_readability_fallback
            print("✓ Readability fallback function available")
        else:
            print("⚠ Readability-lxml not available (install readability-lxml)")
        
        return True
    except Exception as e:
        print(f"✗ Readability test failed: {e}")
        return False

def test_empty_url_handling():
    """Test enhanced empty URL handling."""
    try:
        from curator.services._parser import parse_article, ArticleParsingError
        
        try:
            parse_article("")
            print("✗ Empty URL should raise error")
            return False
        except ArticleParsingError as e:
            if "URL cannot be empty" in str(e):
                print("✓ Empty URL properly handled")
                return True
            else:
                print(f"✗ Unexpected error message: {e}")
                return False
    except Exception as e:
        print(f"✗ Empty URL test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Parser Resilience Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_enhanced_error_classes,
        test_user_agent_expansion,
        test_readability_availability,
        test_empty_url_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"✗ {test.__name__} failed")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All parser resilience improvements verified!")
        return True
    else:
        print("✗ Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)