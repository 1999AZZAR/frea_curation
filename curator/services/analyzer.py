"""Public analyzer API that forwards to `._analyzer` at call time.

This avoids duplication and ensures that tests which patch
`curator.services._analyzer.*` affect calls made via this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from curator.core.models import Article, ScoreCard, ScoringConfig
from . import _analyzer as _impl

__all__ = [
    "compute_readability_score",
    "compute_ner_density_score",
    "compute_sentiment_score",
    "compute_tfidf_relevance_score",
    "compute_embeddings_relevance_score",
    "compute_relevance_score",
    "compute_recency_score",
    "calculate_composite_score",
    "analyze_article",
    "batch_analyze",
]


def compute_readability_score(article: Article, min_word_count: int = 300) -> float:
    return _impl.compute_readability_score(article, min_word_count=min_word_count)


def compute_ner_density_score(
    article: Article,
    nlp=None,
    max_entities_per_100_words: float = 10.0,
) -> float:
    return _impl.compute_ner_density_score(
        article, nlp=nlp, max_entities_per_100_words=max_entities_per_100_words
    )


def compute_sentiment_score(article: Article, vader_analyzer=None) -> float:
    return _impl.compute_sentiment_score(article, vader_analyzer=vader_analyzer)


def compute_tfidf_relevance_score(article: Article, query: str) -> float:
    return _impl.compute_tfidf_relevance_score(article, query)


def compute_embeddings_relevance_score(article: Article, query: str) -> float:
    return _impl.compute_embeddings_relevance_score(article, query)


def compute_relevance_score(article: Article, query: str) -> float:
    return _impl.compute_relevance_score(article, query)


def compute_recency_score(
    article: Article, now: Optional[object] = None, half_life_days: float = 7.0
) -> float:
    return _impl.compute_recency_score(article, now=now, half_life_days=half_life_days)


def calculate_composite_score(metrics: Dict[str, float], config: ScoringConfig) -> float:
    return _impl.calculate_composite_score(metrics, config)


def analyze_article(
    url: str,
    query: Optional[str] = None,
    config: Optional[ScoringConfig] = None,
    nlp=None,
    vader_analyzer=None,
) -> ScoreCard:
    return _impl.analyze_article(
        url=url,
        query=query,
        config=config,
        nlp=nlp,
        vader_analyzer=vader_analyzer,
    )


def batch_analyze(
    urls: List[str],
    query: Optional[str] = None,
    config: Optional[ScoringConfig] = None,
    nlp=None,
    vader_analyzer=None,
    apply_diversity: Optional[bool] = None,
) -> List[ScoreCard]:
    return _impl.batch_analyze(
        urls=urls,
        query=query,
        config=config,
        nlp=nlp,
        vader_analyzer=vader_analyzer,
        apply_diversity=apply_diversity,
    )
