#!/usr/bin/env python3
"""
Example usage of the article parser module.

This demonstrates how to use the parser functionality for both single
articles and batch processing.
"""

from parser import (
    parse_article, batch_parse_articles, 
    ArticleParsingError, ContentValidationError
)

def example_single_article():
    """Example of parsing a single article."""
    url = "https://example.com/article"
    
    try:
        article = parse_article(url, min_word_count=300, max_retries=3)
        
        if article:
            print(f"Successfully parsed: {article.title}")
            print(f"Author: {article.author}")
            print(f"Word count: {len(article.content.split())}")
            print(f"Published: {article.publish_date}")
            return article
        else:
            print("Failed to parse article")
            return None
            
    except ArticleParsingError as e:
        print(f"Parsing error: {e}")
        return None
    except ContentValidationError as e:
        print(f"Validation error: {e}")
        return None

def example_batch_articles():
    """Example of parsing multiple articles."""
    urls = [
        "https://example.com/article1",
        "https://example.com/article2", 
        "https://example.com/article3"
    ]
    
    try:
        articles = batch_parse_articles(urls, min_word_count=300, max_retries=2)
        
        print(f"Successfully parsed {len(articles)} out of {len(urls)} articles")
        
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.title} ({len(article.content.split())} words)")
            
        return articles
        
    except Exception as e:
        print(f"Batch parsing error: {e}")
        return []

def main():
    """Main example function."""
    print("Article Parser Examples")
    print("=" * 50)
    
    print("\n1. Single Article Parsing:")
    print("-" * 30)
    example_single_article()
    
    print("\n2. Batch Article Parsing:")
    print("-" * 30)
    example_batch_articles()
    
    print("\nNote: These examples will fail without actual URLs and network access.")
    print("The parser module is ready for integration with real article URLs.")

if __name__ == "__main__":
    main()