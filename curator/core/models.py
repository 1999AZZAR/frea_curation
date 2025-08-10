"""
Canonical data models for Curator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Entity:
    text: str
    label: str  # PERSON, ORG, GPE, etc.
    confidence: float

    def __post_init__(self):
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("Confidence must be a numeric value")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.text.strip():
            raise ValueError("Entity text cannot be empty")
        if not self.label.strip():
            raise ValueError("Entity label cannot be empty")


@dataclass(eq=True, frozen=False)
class Article:
    url: str
    title: str = ""
    author: str = ""
    publish_date: Optional[datetime] = None
    content: str = ""
    summary: str = ""
    entities: List[Entity] = field(default_factory=list)

    def __post_init__(self):
        if not self.url.strip():
            raise ValueError("Article URL cannot be empty")
        if self.entities is None:
            self.entities = []


@dataclass
class ScoreCard:
    overall_score: float
    readability_score: float
    ner_density_score: float
    sentiment_score: float
    tfidf_relevance_score: float
    recency_score: float
    article: Article

    def __post_init__(self):
        scores = [
            self.overall_score,
            self.readability_score,
            self.ner_density_score,
            self.sentiment_score,
            self.tfidf_relevance_score,
            self.recency_score,
        ]
        for score in scores:
            if not isinstance(score, (int, float)):
                raise ValueError("All scores must be numeric values")
            if not 0.0 <= score <= 100.0:
                raise ValueError("All scores must be between 0.0 and 100.0")


@dataclass
class ScoringConfig:
    readability_weight: float = 0.2
    ner_density_weight: float = 0.2
    sentiment_weight: float = 0.15
    tfidf_relevance_weight: float = 0.25
    recency_weight: float = 0.2
    min_word_count: int = 300
    max_articles_per_topic: int = 20

    def __post_init__(self):
        total_weight = (
            self.readability_weight
            + self.ner_density_weight
            + self.sentiment_weight
            + self.tfidf_relevance_weight
            + self.recency_weight
        )
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")

        weights = [
            self.readability_weight,
            self.ner_density_weight,
            self.sentiment_weight,
            self.tfidf_relevance_weight,
            self.recency_weight,
        ]
        for weight in weights:
            if not isinstance(weight, (int, float)):
                raise ValueError("All weights must be numeric values")
            if not 0.0 <= weight <= 1.0:
                raise ValueError("All weights must be between 0.0 and 1.0")

        if not isinstance(self.min_word_count, int) or self.min_word_count < 0:
            raise ValueError("min_word_count must be a non-negative integer")

        if not isinstance(self.max_articles_per_topic, int) or self.max_articles_per_topic < 1:
            raise ValueError("max_articles_per_topic must be a positive integer")


