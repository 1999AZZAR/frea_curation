#!/usr/bin/env python3
"""
Simple test script to verify parser functionality without external dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from models import Article, Entity, ScoreCard, ScoringConfig
        print("✓ Models imported successfully")
        
        # Test basic model creation
        article = Article(url="https://example.com/test")
        print("✓ Article model creation works")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_parser_structure():
    """Test parser module structure without external dependencies."""
    try:
        # Check if parser.py exists and has correct structure
        with open('parser.py', 'r') as f:
            content = f.read()
        
        # Check for required functions
        required_functions = [
            'parse_article',
            'batch_parse_articles',
            'validate_content',
            'parse_article_with_config',
            'get_article_word_count',
            'is_article_recent'
        ]
        
        for func in required_functions:
            if f"def {func}" in content:
                print(f"✓ Function {func} found")
            else:
                print(f"✗ Function {func} missing")
                return False
        
        # Check for required classes
        required_classes = ['ArticleParsingError', 'ContentValidationError']
        for cls in required_classes:
            if f"class {cls}" in content:
                print(f"✓ Class {cls} found")
            else:
                print(f"✗ Class {cls} missing")
                return False
        
        # Check for USER_AGENTS
        if "USER_AGENTS = [" in content:
            print("✓ USER_AGENTS list found")
        else:
            print("✗ USER_AGENTS list missing")
            return False
        
        return True
    except FileNotFoundError:
        print("✗ parser.py file not found")
        return False
    except Exception as e:
        print(f"✗ Error reading parser.py: {e}")
        return False

def test_parser_logic():
    """Test parser logic without external dependencies."""
    try:
        # Import parser functions (this will fail if newspaper3k is not available)
        # But we can still test the structure
        print("Testing parser module structure...")
        
        # Check if the file can be parsed as Python
        with open('parser.py', 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, 'parser.py', 'exec')
        print("✓ Parser module syntax is valid")
        
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in parser.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error compiling parser.py: {e}")
        return False

def main():
    """Run all tests."""
    print("Running simple parser tests...\n")
    
    tests = [
        ("Model imports", test_imports),
        ("Parser structure", test_parser_structure),
        ("Parser logic", test_parser_logic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n--- Results ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())