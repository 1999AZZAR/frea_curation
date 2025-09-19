"""
Summarization utilities for article content.

Provides extractive and lightweight abstractive summarization capabilities
to generate summaries when newspaper3k summary is missing or low quality.
"""

import re
import logging
from typing import Optional, List, Tuple
from collections import Counter
import math

logger = logging.getLogger(__name__)


def is_summary_low_quality(summary: str, content: str, min_length: int = 30, 
                          max_ratio: float = 0.8) -> bool:
    """
    Determine if an existing summary is low quality and needs replacement.
    
    Args:
        summary: Existing summary text
        content: Full article content
        min_length: Minimum acceptable summary length
        max_ratio: Maximum ratio of summary to content length
        
    Returns:
        True if summary is low quality and should be replaced
    """
    if not summary or not summary.strip():
        return True
    
    summary = summary.strip()
    
    # Too short
    if len(summary) < min_length:
        return True
    
    # Too long relative to content (likely not a real summary)
    if content and len(summary) / len(content) > max_ratio:
        return True
    
    # Check for common low-quality patterns
    low_quality_patterns = [
        r'^(click here|read more|continue reading)',
        r'(subscribe|sign up|newsletter)',
        r'^(advertisement|sponsored)',
        r'^\s*\.\.\.\s*$',  # Just ellipsis
        r'^.{1,20}$',  # Very short generic text
        r'\.\.\.\s*$',  # Ends with ellipsis (incomplete)
        r'^[^.!?]*\.\.\.\s*$'  # Single sentence ending with ellipsis
    ]
    
    for pattern in low_quality_patterns:
        if re.search(pattern, summary.lower()):
            return True
    
    return False


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex patterns."""
    if not text:
        return []
    
    # Simple sentence splitting - handles most cases
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter very short fragments
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def calculate_sentence_scores(sentences: List[str], content: str) -> List[Tuple[str, float]]:
    """
    Calculate importance scores for sentences using extractive methods.
    
    Uses a combination of:
    - Word frequency (TF)
    - Sentence position
    - Sentence length
    """
    if not sentences:
        return []
    
    # Calculate word frequencies
    words = re.findall(r'\b\w+\b', content.lower())
    word_freq = Counter(words)
    
    # Remove common stop words (simple list)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Filter stop words and calculate normalized frequencies
    filtered_freq = {word: freq for word, freq in word_freq.items() 
                    if word not in stop_words and len(word) > 2}
    
    if not filtered_freq:
        return [(sent, 0.0) for sent in sentences]
    
    max_freq = max(filtered_freq.values())
    normalized_freq = {word: freq / max_freq for word, freq in filtered_freq.items()}
    
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Word frequency score
        word_score = sum(normalized_freq.get(word, 0) for word in sentence_words)
        if sentence_words:
            word_score /= len(sentence_words)
        
        # Position score (earlier sentences get higher scores)
        position_score = 1.0 - (i / len(sentences)) * 0.5
        
        # Length score (prefer medium-length sentences)
        length = len(sentence.split())
        if 10 <= length <= 30:
            length_score = 1.0
        elif length < 10:
            length_score = length / 10.0
        else:
            length_score = max(0.5, 30.0 / length)
        
        # Combined score
        total_score = (word_score * 0.6 + position_score * 0.3 + length_score * 0.1)
        scored_sentences.append((sentence, total_score))
    
    return scored_sentences


def extractive_summarize(content: str, max_sentences: int = 3, 
                        target_length: int = 200) -> str:
    """
    Generate extractive summary by selecting top-scoring sentences.
    
    Args:
        content: Full article content
        max_sentences: Maximum number of sentences in summary
        target_length: Target summary length in characters
        
    Returns:
        Extractive summary text
    """
    if not content or not content.strip():
        return ""
    
    content = clean_text(content)
    sentences = split_into_sentences(content)
    
    if not sentences:
        return ""
    
    # If content is already short, return first part
    if len(content) <= target_length:
        return content[:target_length].strip()
    
    # Score sentences
    scored_sentences = calculate_sentence_scores(sentences, content)
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select sentences while staying under target length
    selected_sentences = []
    current_length = 0
    
    for sentence, score in scored_sentences:
        if len(selected_sentences) >= max_sentences:
            break
        
        sentence_length = len(sentence)
        if current_length + sentence_length <= target_length:
            selected_sentences.append(sentence)
            current_length += sentence_length
        elif not selected_sentences:  # Always include at least one sentence
            selected_sentences.append(sentence[:target_length])
            break
    
    if not selected_sentences:
        # Fallback: return first part of content
        return content[:target_length].strip() + "..."
    
    # Join sentences and clean up
    summary = ". ".join(selected_sentences)
    if not summary.endswith('.'):
        summary += "."
    
    return summary.strip()


def simple_abstractive_summarize(content: str, target_length: int = 200) -> str:
    """
    Generate simple abstractive summary using keyword extraction and templates.
    
    This is a lightweight approach that doesn't require heavy NLP models.
    
    Args:
        content: Full article content
        target_length: Target summary length in characters
        
    Returns:
        Simple abstractive summary
    """
    if not content or not content.strip():
        return ""
    
    content = clean_text(content)
    
    # Extract key information
    sentences = split_into_sentences(content)
    if not sentences:
        return content[:target_length].strip()
    
    # Find the most informative sentence (usually first or highest scoring)
    scored_sentences = calculate_sentence_scores(sentences, content)
    if scored_sentences:
        best_sentence = max(scored_sentences, key=lambda x: x[1])[0]
    else:
        best_sentence = sentences[0] if sentences else content[:100]
    
    # Extract key entities/topics (simple approach)
    words = re.findall(r'\b[A-Z][a-z]+\b', content)  # Capitalized words
    word_freq = Counter(words)
    key_terms = [word for word, freq in word_freq.most_common(5) if freq > 1]
    
    # Create simple abstractive summary
    if key_terms:
        key_terms_str = ", ".join(key_terms[:3])
        summary = f"This article discusses {key_terms_str}. {best_sentence}"
    else:
        summary = best_sentence
    
    # Trim to target length
    if len(summary) > target_length:
        summary = summary[:target_length].rsplit(' ', 1)[0] + "..."
    
    return summary.strip()


def generate_summary(content: str, existing_summary: str = "", 
                    method: str = "extractive", target_length: int = 200) -> str:
    """
    Generate article summary using specified method.
    
    Args:
        content: Full article content
        existing_summary: Existing summary (if any)
        method: Summarization method ("extractive" or "abstractive")
        target_length: Target summary length in characters
        
    Returns:
        Generated summary text
    """
    try:
        # Check if existing summary is good enough
        if existing_summary and not is_summary_low_quality(existing_summary, content):
            logger.debug("Existing summary is acceptable, using as-is")
            return existing_summary.strip()
        
        if not content or not content.strip():
            logger.warning("No content available for summarization")
            return existing_summary.strip() if existing_summary else ""
        
        # Generate new summary based on method
        if method == "abstractive":
            summary = simple_abstractive_summarize(content, target_length)
            logger.debug(f"Generated abstractive summary: {len(summary)} chars")
        else:  # Default to extractive
            summary = extractive_summarize(content, target_length=target_length)
            logger.debug(f"Generated extractive summary: {len(summary)} chars")
        
        # Fallback to existing summary if generation failed
        if not summary or len(summary.strip()) < 20:
            logger.warning("Generated summary too short, falling back to existing")
            return existing_summary.strip() if existing_summary else content[:target_length].strip()
        
        return summary
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Return existing summary or truncated content as fallback
        if existing_summary:
            return existing_summary.strip()
        elif content:
            return content[:target_length].strip() + "..."
        else:
            return ""


def enhance_article_summary(article, method: str = "extractive", 
                          target_length: int = 200) -> None:
    """
    Enhance article summary in-place if needed.
    
    Args:
        article: Article object to enhance
        method: Summarization method to use
        target_length: Target summary length
    """
    try:
        if not hasattr(article, 'summary') or not hasattr(article, 'content'):
            logger.warning("Article missing summary or content attributes")
            return
        
        original_summary = getattr(article, 'summary', '') or ''
        content = getattr(article, 'content', '') or ''
        
        # Generate enhanced summary
        enhanced_summary = generate_summary(
            content=content,
            existing_summary=original_summary,
            method=method,
            target_length=target_length
        )
        
        # Update article summary
        article.summary = enhanced_summary
        
        logger.debug(f"Enhanced summary: {len(original_summary)} -> {len(enhanced_summary)} chars")
        
    except Exception as e:
        logger.error(f"Failed to enhance article summary: {e}")
        # Leave original summary unchanged on error