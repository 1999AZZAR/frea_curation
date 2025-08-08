"""
Unit tests for validation functions.

Tests URL validation, input sanitization, and other
validation utilities.
"""

import unittest
from validation import (
    validate_url, 
    validate_topic_keywords, 
    validate_url_list, 
    sanitize_input
)


class TestUrlValidation(unittest.TestCase):
    """Test cases for URL validation."""
    
    def test_valid_urls(self):
        """Test validation of valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://www.example.com/article",
            "https://subdomain.example.com/path/to/article",
            "https://example.com/article?param=value",
            "https://example.com:8080/article",
            "https://example-site.com/article-title"
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                is_valid, error = validate_url(url)
                self.assertTrue(is_valid, f"URL {url} should be valid, got error: {error}")
                self.assertIsNone(error)
    
    def test_invalid_urls(self):
        """Test validation of invalid URLs."""
        invalid_urls = [
            "",  # Empty string
            "   ",  # Whitespace only
            "not-a-url",  # No protocol
            "ftp://example.com",  # Wrong protocol
            "https://",  # No domain
            "https:///path",  # No domain
            "javascript:alert('xss')",  # Dangerous protocol
            "https://" + "a" * 2050,  # Too long
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                is_valid, error = validate_url(url)
                self.assertFalse(is_valid, f"URL {url} should be invalid")
                self.assertIsNotNone(error)
    
    def test_url_with_whitespace(self):
        """Test URL validation with surrounding whitespace."""
        url_with_spaces = "  https://example.com/article  "
        is_valid, error = validate_url(url_with_spaces)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_none_and_non_string_urls(self):
        """Test URL validation with None and non-string inputs."""
        invalid_inputs = [None, 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                is_valid, error = validate_url(invalid_input)
                self.assertFalse(is_valid)
                self.assertIsNotNone(error)


class TestTopicKeywordsValidation(unittest.TestCase):
    """Test cases for topic keywords validation."""
    
    def test_valid_keywords(self):
        """Test validation of valid topic keywords."""
        valid_keywords = [
            "technology",
            "artificial intelligence",
            "machine learning AI",
            "climate change 2024",
            "sports news",
            "business finance",
            "health medical research"
        ]
        
        for keywords in valid_keywords:
            with self.subTest(keywords=keywords):
                is_valid, error = validate_topic_keywords(keywords)
                self.assertTrue(is_valid, f"Keywords '{keywords}' should be valid, got error: {error}")
                self.assertIsNone(error)
    
    def test_invalid_keywords(self):
        """Test validation of invalid topic keywords."""
        invalid_keywords = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Too short
            "a" * 201,  # Too long
            "test<script>",  # Contains forbidden character
            "test>alert",  # Contains forbidden character
            'test"quote',  # Contains forbidden character
            "test'quote",  # Contains forbidden character
            "test&amp;",  # Contains forbidden character
            "test;drop",  # Contains forbidden character
        ]
        
        for keywords in invalid_keywords:
            with self.subTest(keywords=keywords):
                is_valid, error = validate_topic_keywords(keywords)
                self.assertFalse(is_valid, f"Keywords '{keywords}' should be invalid")
                self.assertIsNotNone(error)
    
    def test_keywords_with_whitespace(self):
        """Test keywords validation with surrounding whitespace."""
        keywords_with_spaces = "  artificial intelligence  "
        is_valid, error = validate_topic_keywords(keywords_with_spaces)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_none_and_non_string_keywords(self):
        """Test keywords validation with None and non-string inputs."""
        invalid_inputs = [None, 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                is_valid, error = validate_topic_keywords(invalid_input)
                self.assertFalse(is_valid)
                self.assertIsNotNone(error)


class TestUrlListValidation(unittest.TestCase):
    """Test cases for URL list validation."""
    
    def test_mixed_url_list(self):
        """Test validation of a mixed list of valid and invalid URLs."""
        urls = [
            "https://example.com/valid1",
            "invalid-url",
            "https://example.com/valid2",
            "",
            "https://example.com/valid3"
        ]
        
        valid_urls, invalid_urls = validate_url_list(urls)
        
        self.assertEqual(len(valid_urls), 3)
        self.assertEqual(len(invalid_urls), 2)
        
        self.assertIn("https://example.com/valid1", valid_urls)
        self.assertIn("https://example.com/valid2", valid_urls)
        self.assertIn("https://example.com/valid3", valid_urls)
        
        # Check that invalid URLs include error messages
        invalid_url_strings = [item.split(':')[0] for item in invalid_urls]
        self.assertIn("invalid-url", invalid_url_strings)
        self.assertIn("", invalid_url_strings)
    
    def test_all_valid_urls(self):
        """Test validation of a list with all valid URLs."""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        
        valid_urls, invalid_urls = validate_url_list(urls)
        
        self.assertEqual(len(valid_urls), 3)
        self.assertEqual(len(invalid_urls), 0)
    
    def test_all_invalid_urls(self):
        """Test validation of a list with all invalid URLs."""
        urls = [
            "invalid-url-1",
            "invalid-url-2",
            ""
        ]
        
        valid_urls, invalid_urls = validate_url_list(urls)
        
        self.assertEqual(len(valid_urls), 0)
        self.assertEqual(len(invalid_urls), 3)
    
    def test_empty_url_list(self):
        """Test validation of an empty URL list."""
        urls = []
        
        valid_urls, invalid_urls = validate_url_list(urls)
        
        self.assertEqual(len(valid_urls), 0)
        self.assertEqual(len(invalid_urls), 0)


class TestInputSanitization(unittest.TestCase):
    """Test cases for input sanitization."""
    
    def test_normal_input_sanitization(self):
        """Test sanitization of normal input."""
        input_text = "This is a normal input string."
        result = sanitize_input(input_text)
        self.assertEqual(result, "This is a normal input string.")
    
    def test_whitespace_trimming(self):
        """Test that leading and trailing whitespace is trimmed."""
        input_text = "   This has whitespace   "
        result = sanitize_input(input_text)
        self.assertEqual(result, "This has whitespace")
    
    def test_length_limiting(self):
        """Test that input is limited to maximum length."""
        long_input = "a" * 1500
        result = sanitize_input(long_input, max_length=100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result, "a" * 100)
    
    def test_control_character_removal(self):
        """Test removal of control characters."""
        input_with_control = "Normal text\x00\x01\x02with control chars"
        result = sanitize_input(input_with_control)
        self.assertEqual(result, "Normal textwith control chars")
    
    def test_preserve_allowed_whitespace(self):
        """Test that newlines, carriage returns, and tabs are preserved."""
        input_text = "Line 1\nLine 2\rLine 3\tTabbed"
        result = sanitize_input(input_text)
        self.assertEqual(result, "Line 1\nLine 2\rLine 3\tTabbed")
    
    def test_none_and_non_string_input(self):
        """Test sanitization with None and non-string inputs."""
        invalid_inputs = [None, 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                result = sanitize_input(invalid_input)
                self.assertEqual(result, "")
    
    def test_empty_string_input(self):
        """Test sanitization of empty string."""
        result = sanitize_input("")
        self.assertEqual(result, "")
    
    def test_whitespace_only_input(self):
        """Test sanitization of whitespace-only input."""
        result = sanitize_input("   ")
        self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()