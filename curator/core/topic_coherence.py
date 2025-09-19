"""
Topic coherence and coverage analysis using YAKE keyphrase extraction.

Implements keyword coverage ratio and keyphrase extraction to measure how well
an article covers the requested topic/query terms.
"""

from typing import List, Dict, Set, Optional, Tuple
import re


def get_yake_extractor(language: str = "en", max_ngram_size: int = 3, deduplication_threshold: float = 0.9):
    """Get YAKE keyphrase extractor with graceful fallback."""
    try:
        import yake
        return yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=deduplication_threshold,
            top=20,  # Extract top 20 keyphrases
            features=None
        )
    except ImportError:
        return None
    except Exception:
        return None


def extract_keyphrases(text: str, max_keyphrases: int = 20) -> List[Tuple[str, float]]:
    """
    Extract keyphrases from text using YAKE.
    
    Args:
        text: Input text to extract keyphrases from
        max_keyphrases: Maximum number of keyphrases to return
        
    Returns:
        List of (keyphrase, score) tuples, sorted by relevance (lower score = better)
        Returns empty list if YAKE is unavailable
    """
    if not text or not text.strip():
        return []
    
    extractor = get_yake_extractor()
    if extractor is None:
        return []
    
    try:
        keyphrases = extractor.extract_keywords(text)
        # YAKE returns (keyphrase, score) where lower score = better
        # Limit to max_keyphrases and ensure we have valid results
        result = []
        for phrase, score in keyphrases[:max_keyphrases]:
            if isinstance(phrase, str) and phrase.strip() and isinstance(score, (int, float)):
                result.append((phrase.strip(), float(score)))
        return result
    except Exception:
        return []


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for keyword matching by lowercasing and removing punctuation."""
    if not text:
        return ""
    # Convert to lowercase and replace non-alphanumeric with spaces
    normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def extract_query_keywords(query: str) -> Set[str]:
    """
    Extract individual keywords from a query string.
    
    Args:
        query: Query string to extract keywords from
        
    Returns:
        Set of normalized keywords
    """
    if not query or not query.strip():
        return set()
    
    normalized = normalize_text_for_matching(query)
    # Split into individual words and filter out short words
    words = [word for word in normalized.split() if len(word) >= 2]
    return set(words)


def calculate_keyword_coverage_ratio(article_text: str, query: str) -> float:
    """
    Calculate the ratio of query keywords that appear in the article text.
    
    Args:
        article_text: Full article text to search in
        query: Query string containing keywords to look for
        
    Returns:
        Coverage ratio between 0.0 and 1.0 (1.0 = all keywords found)
    """
    if not article_text or not query:
        return 0.0
    
    query_keywords = extract_query_keywords(query)
    if not query_keywords:
        return 0.0
    
    article_normalized = normalize_text_for_matching(article_text)
    
    # Count how many query keywords appear in the article (substring matching)
    found_keywords = set()
    for keyword in query_keywords:
        if keyword in article_normalized:
            found_keywords.add(keyword)
    
    coverage_ratio = len(found_keywords) / len(query_keywords)
    
    return min(1.0, max(0.0, coverage_ratio))


def calculate_keyphrase_relevance(article_keyphrases: List[Tuple[str, float]], query: str) -> float:
    """
    Calculate how relevant the article's keyphrases are to the query.
    
    Args:
        article_keyphrases: List of (keyphrase, yake_score) from article
        query: Query string to compare against
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not article_keyphrases or not query:
        return 0.0
    
    query_normalized = normalize_text_for_matching(query)
    query_words = set(query_normalized.split())
    
    if not query_words:
        return 0.0
    
    relevance_scores = []
    
    for keyphrase, yake_score in article_keyphrases:
        if not keyphrase:
            continue
            
        phrase_normalized = normalize_text_for_matching(keyphrase)
        phrase_words = set(phrase_normalized.split())
        
        # Calculate word overlap between keyphrase and query
        overlap = len(phrase_words.intersection(query_words))
        if overlap > 0:
            # Convert YAKE score to relevance (lower YAKE score = higher relevance)
            # YAKE scores typically range from 0 to ~10, with 0 being best
            yake_relevance = max(0.0, 1.0 - min(1.0, yake_score / 10.0))
            
            # Weight by word overlap ratio
            overlap_ratio = overlap / len(phrase_words) if phrase_words else 0.0
            combined_relevance = yake_relevance * overlap_ratio
            relevance_scores.append(combined_relevance)
    
    if not relevance_scores:
        return 0.0
    
    # Return the average relevance of matching keyphrases
    return sum(relevance_scores) / len(relevance_scores)


def compute_topic_coherence_score(article_text: str, article_title: str, query: str) -> float:
    """
    Compute overall topic coherence score combining keyword coverage and keyphrase relevance.
    
    Args:
        article_text: Full article content
        article_title: Article title (used for keyphrase extraction)
        query: Query/topic to measure coherence against
        
    Returns:
        Topic coherence score between 0.0 and 100.0
    """
    if not query or not query.strip():
        return 0.0
    
    # Combine title and content for analysis, with title weighted more heavily
    combined_text = ""
    if article_title and article_title.strip():
        combined_text += article_title.strip() + " "
    if article_text and article_text.strip():
        combined_text += article_text.strip()
    
    if not combined_text.strip():
        return 0.0
    
    # Calculate keyword coverage ratio (40% weight)
    coverage_ratio = calculate_keyword_coverage_ratio(combined_text, query)
    coverage_score = coverage_ratio * 40.0
    
    # Calculate keyphrase relevance (60% weight)
    keyphrases = extract_keyphrases(combined_text)
    keyphrase_relevance = calculate_keyphrase_relevance(keyphrases, query)
    keyphrase_score = keyphrase_relevance * 60.0
    
    # Combine scores
    total_score = coverage_score + keyphrase_score
    
    return round(min(100.0, max(0.0, total_score)), 2)


def get_article_keyphrases(article_text: str, article_title: str, max_keyphrases: int = 10) -> List[str]:
    """
    Extract keyphrases from article for display purposes.
    
    Args:
        article_text: Full article content
        article_title: Article title
        max_keyphrases: Maximum number of keyphrases to return
        
    Returns:
        List of keyphrase strings (without scores)
    """
    combined_text = ""
    if article_title and article_title.strip():
        combined_text += article_title.strip() + " "
    if article_text and article_text.strip():
        combined_text += article_text.strip()
    
    if not combined_text.strip():
        return []
    
    keyphrases = extract_keyphrases(combined_text, max_keyphrases)
    return [phrase for phrase, _ in keyphrases]