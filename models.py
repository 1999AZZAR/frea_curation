"""
Data models for the AI Content Curator application.

This module defines the core data structures used throughout the application
for representing articles, entities, scoring results, and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Entity:
    """Represents a named entity extracted from article content."""
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    confidence: float
    
    def __post_init__(self):
        """Validate entity data after initialization."""
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("Confidence must be a numeric value")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.text.strip():
            raise ValueError("Entity text cannot be empty")
        if not self.label.strip():
            raise ValueError("Entity label cannot be empty")


@dataclass
class Article:
    """Represents a parsed article with metadata and content."""
    url: str
    title: str = ""
    author: str = ""
    publish_date: Optional[datetime] = None
    content: str = ""
    summary: str = ""
    entities: List[Entity] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate article data after initialization."""
        if not self.url.strip():
            raise ValueError("Article URL cannot be empty")
        # Ensure entities is always a list
        if self.entities is None:
            self.entities = []


@dataclass
class ScoreCard:
    """Represents the complete scoring analysis for an article."""
    overall_score: float
    readability_score: float
    ner_density_score: float
    sentiment_score: float
    tfidf_relevance_score: float
    recency_score: float
    article: Article
    
    def __post_init__(self):
        """Validate score data after initialization."""
        scores = [
            self.overall_score,
            self.readability_score,
            self.ner_density_score,
            self.sentiment_score,
            self.tfidf_relevance_score,
            self.recency_score
        ]
        
        for score in scores:
            if not isinstance(score, (int, float)):
                raise ValueError("All scores must be numeric values")
            if not 0.0 <= score <= 100.0:
                raise ValueError("All scores must be between 0.0 and 100.0")


@dataclass
class ScoringConfig:
    """Configuration for the scoring algorithm weights and parameters."""
    readability_weight: float = 0.2
    ner_density_weight: float = 0.2
    sentiment_weight: float = 0.15
    tfidf_relevance_weight: float = 0.25
    recency_weight: float = 0.2
    min_word_count: int = 300
    max_articles_per_topic: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check that weights sum to 1.0 (with small tolerance for floating point)
        total_weight = (
            self.readability_weight +
            self.ner_density_weight +
            self.sentiment_weight +
            self.tfidf_relevance_weight +
            self.recency_weight
        )
        
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Validate individual weights
        weights = [
            self.readability_weight,
            self.ner_density_weight,
            self.sentiment_weight,
            self.tfidf_relevance_weight,
            self.recency_weight
        ]
        
        for weight in weights:
            if not isinstance(weight, (int, float)):
                raise ValueError("All weights must be numeric values")
            if not 0.0 <= weight <= 1.0:
                raise ValueError("All weights must be between 0.0 and 1.0")
        
        # Validate other parameters
        if not isinstance(self.min_word_count, int) or self.min_word_count < 0:
            raise ValueError("min_word_count must be a non-negative integer")
        
        if not isinstance(self.max_articles_per_topic, int) or self.max_articles_per_topic < 1:
            raise ValueError("max_articles_per_topic must be a positive integer")