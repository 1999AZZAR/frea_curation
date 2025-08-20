"""
Validation functions for the AI Content Curator application.

This module provides validation utilities for user input,
particularly URL validation and data integrity checks.
"""

import validators
from urllib.parse import urlparse
from typing import Optional, List


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate if a URL is properly formatted and accessible.
    
    Args:
        url: The URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
        If invalid, error_message contains the reason
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty"
    
    # Strip whitespace
    url = url.strip()
    
    if not url:
        return False, "URL cannot be empty"
    
    # Check basic URL format using validators library
    if not validators.url(url):
        return False, "Invalid URL format"
    
    # Parse URL to check components
    try:
        parsed = urlparse(url)
        
        # Check for required components
        if not parsed.scheme:
            return False, "URL must include protocol (http:// or https://)"
        
        if parsed.scheme not in ['http', 'https']:
            return False, "URL must use HTTP or HTTPS protocol"
        
        if not parsed.netloc:
            return False, "URL must include a valid domain"
        
        # Check for suspicious patterns
        if len(url) > 2048:
            return False, "URL is too long (maximum 2048 characters)"
        
        return True, None
        
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"


def validate_topic_keywords(keywords: str) -> tuple[bool, Optional[str]]:
    """
    Validate topic keywords for news search.
    
    Args:
        keywords: The topic keywords string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not keywords or not isinstance(keywords, str):
        return False, "Keywords cannot be empty"
    
    # Strip whitespace
    keywords = keywords.strip()
    
    if not keywords:
        return False, "Keywords cannot be empty"
    
    if len(keywords) < 2:
        return False, "Keywords must be at least 2 characters long"
    
    if len(keywords) > 200:
        return False, "Keywords are too long (maximum 200 characters)"
    
    # Check for potentially problematic characters
    forbidden_chars = ['<', '>', '"', "'", '&', ';']
    for char in forbidden_chars:
        if char in keywords:
            return False, f"Keywords cannot contain '{char}' character"
    
    return True, None


def validate_url_list(urls: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate a list of URLs and return valid and invalid ones separately.
    
    Args:
        urls: List of URL strings to validate
        
    Returns:
        Tuple of (valid_urls, invalid_urls_with_reasons)
    """
    valid_urls = []
    invalid_urls = []
    
    for url in urls:
        is_valid, error_message = validate_url(url)
        if is_valid:
            valid_urls.append(url)
        else:
            invalid_urls.append(f"{url}: {error_message}")
    
    return valid_urls, invalid_urls


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize user input by removing potentially harmful characters.
    
    Args:
        input_string: The input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not input_string or not isinstance(input_string, str):
        return ""
    
    # Strip whitespace and limit length
    sanitized = input_string.strip()[:max_length]
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in ['\n', '\r', '\t'])
    
    return sanitized

