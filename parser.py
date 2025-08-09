"""
Article parsing and content extraction module.

This module provides functionality to parse articles from URLs using newspaper3k,
with comprehensive error handling, retry logic, and content validation.
"""

import time
import logging
from typing import Optional, List
from datetime import datetime
import requests
from newspaper import Article as NewspaperArticle, Config
from models import Article

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for retry attempts
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
]

class ArticleParsingError(Exception):
    """Custom exception for article parsing errors."""
    pass

class ContentValidationError(Exception):
    """Custom exception for content validation errors."""
    pass

def validate_content(article: NewspaperArticle, min_word_count: int = 300) -> None:
    """
    Validate parsed article content meets minimum requirements.
    
    Args:
        article: Parsed newspaper article object
        min_word_count: Minimum required word count
        
    Raises:
        ContentValidationError: If content doesn't meet requirements
    """
    if not article.text or not article.text.strip():
        raise ContentValidationError("Article content is empty")
    
    word_count = len(article.text.split())
    if word_count < min_word_count:
        raise ContentValidationError(
            f"Article too short: {word_count} words (minimum: {min_word_count})"
        )
    
    if not article.title or not article.title.strip():
        raise ContentValidationError("Article title is missing")

def parse_article_with_config(url: str, user_agent: str, timeout: int = 30) -> NewspaperArticle:
    """
    Parse article with specific configuration and user agent.
    
    Args:
        url: Article URL to parse
        user_agent: User agent string to use
        timeout: Request timeout in seconds
        
    Returns:
        Parsed newspaper article object
        
    Raises:
        ArticleParsingError: If parsing fails
    """
    try:
        # Configure newspaper3k with custom settings
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = timeout
        config.number_threads = 1
        config.thread_timeout_seconds = timeout
        
        # Create and download article
        article = NewspaperArticle(url, config=config)
        article.download()
        
        # Check if download was successful
        if not article.html:
            raise ArticleParsingError("Failed to download article HTML")
        
        # Parse the article
        article.parse()
        
        return article
        
    except requests.exceptions.Timeout:
        raise ArticleParsingError(f"Request timeout after {timeout} seconds")
    except requests.exceptions.ConnectionError:
        raise ArticleParsingError("Network connection error")
    except requests.exceptions.RequestException as e:
        raise ArticleParsingError(f"Request failed: {str(e)}")
    except Exception as e:
        raise ArticleParsingError(f"Parsing failed: {str(e)}")

def parse_article(url: str, min_word_count: int = 300, max_retries: int = 3, 
                 retry_delay: float = 1.0) -> Optional[Article]:
    """
    Parse article from URL with retry logic and comprehensive error handling.
    
    Args:
        url: Article URL to parse
        min_word_count: Minimum required word count for content validation
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retry attempts in seconds
        
    Returns:
        Parsed Article object or None if parsing fails
        
    Raises:
        ArticleParsingError: If all retry attempts fail
        ContentValidationError: If content doesn't meet validation requirements
    """
    if not url or not url.strip():
        raise ArticleParsingError("URL cannot be empty")
    
    last_error = None
    
    # Try parsing with different user agents
    for attempt in range(max_retries):
        user_agent = USER_AGENTS[attempt % len(USER_AGENTS)]
        
        try:
            logger.info(f"Parsing attempt {attempt + 1}/{max_retries} for URL: {url}")
            
            # Parse article with current user agent
            newspaper_article = parse_article_with_config(url, user_agent)
            
            # Validate content
            validate_content(newspaper_article, min_word_count)
            
            # Convert to our Article model
            article = Article(
                url=url,
                title=newspaper_article.title or "",
                author=", ".join(newspaper_article.authors) if newspaper_article.authors else "",
                publish_date=newspaper_article.publish_date,
                content=newspaper_article.text or "",
                summary=newspaper_article.summary or "",
                entities=[]  # Will be populated by NLP processing
            )
            
            logger.info(f"Successfully parsed article: {article.title[:50]}...")
            return article
            
        except (ArticleParsingError, ContentValidationError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
    
    # All attempts failed
    error_msg = f"Failed to parse article after {max_retries} attempts. Last error: {str(last_error)}"
    logger.error(error_msg)
    raise ArticleParsingError(error_msg)

def batch_parse_articles(urls: List[str], min_word_count: int = 300, 
                        max_retries: int = 3) -> List[Article]:
    """
    Parse multiple articles with error handling for individual failures.
    
    Args:
        urls: List of article URLs to parse
        min_word_count: Minimum required word count for content validation
        max_retries: Maximum number of retry attempts per article
        
    Returns:
        List of successfully parsed Article objects
    """
    if not urls:
        return []
    
    parsed_articles = []
    
    for i, url in enumerate(urls):
        try:
            logger.info(f"Parsing article {i + 1}/{len(urls)}: {url}")
            article = parse_article(url, min_word_count, max_retries)
            if article:
                parsed_articles.append(article)
        except (ArticleParsingError, ContentValidationError) as e:
            logger.error(f"Failed to parse {url}: {str(e)}")
            # Continue with next article instead of failing entire batch
            continue
    
    logger.info(f"Successfully parsed {len(parsed_articles)}/{len(urls)} articles")
    return parsed_articles

def get_article_word_count(article: Article) -> int:
    """
    Get word count for an article.
    
    Args:
        article: Article object
        
    Returns:
        Number of words in article content
    """
    if not article.content:
        return 0
    return len(article.content.split())

def is_article_recent(article: Article, max_age_days: int = 30) -> bool:
    """
    Check if article is recent based on publication date.
    
    Args:
        article: Article object
        max_age_days: Maximum age in days to consider recent
        
    Returns:
        True if article is recent or has no publication date
    """
    if not article.publish_date:
        return True  # Assume recent if no date available
    
    age_days = (datetime.now() - article.publish_date).days
    return age_days <= max_age_days