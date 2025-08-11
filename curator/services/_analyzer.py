"""
Content analysis and scoring engine.

Implements individual scoring components and composite scoring per design:
- Readability (word count based)
- NER density (spaCy)
- Sentiment (NLTK VADER)
- TF-IDF relevance (scikit-learn)
- Recency (publication date)

Provides orchestrator functions for single and batch analysis.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from curator.core.models import Article, ScoreCard, ScoringConfig
from curator.services.parser import parse_article, batch_parse_articles, get_article_word_count


def compute_readability_score(article: Article, min_word_count: int = 300) -> float:
    word_count = get_article_word_count(article)
    if min_word_count <= 0:
        return 100.0
    ratio = min(1.0, word_count / float(min_word_count))
    return round(ratio * 100.0, 2)


def compute_ner_density_score(
    article: Article,
    nlp=None,
    max_entities_per_100_words: float = 10.0,
) -> float:
    if nlp is None or not article.content:
        return 0.0
    try:
        doc = nlp(article.content)
        num_entities = len(getattr(doc, "ents", []))
        words = max(1, get_article_word_count(article))
        entities_per_100 = (num_entities / float(words)) * 100.0
        ratio = min(1.0, entities_per_100 / max_entities_per_100_words)
        return round(ratio * 100.0, 2)
    except Exception:
        return 0.0


def compute_sentiment_score(article: Article, vader_analyzer=None) -> float:
    if vader_analyzer is None or not article.content:
        return 50.0
    try:
        compound = float(vader_analyzer.polarity_scores(article.content).get("compound", 0.0))
        score = (1.0 - min(1.0, abs(compound))) * 100.0
        return round(score, 2)
    except Exception:
        return 50.0


def compute_tfidf_relevance_score(article: Article, query: str) -> float:
    if not article.content or not query or not query.strip():
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [article.content, query]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(corpus)
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return round(max(0.0, min(1.0, sim)) * 100.0, 2)
    except ValueError:
        return 0.0
    except ModuleNotFoundError:
        return 0.0


def compute_recency_score(article: Article, now: Optional[datetime] = None, half_life_days: float = 7.0) -> float:
    if article.publish_date is None:
        return 100.0
    # Normalize datetimes to UTC-aware to avoid naive/aware subtraction errors
    pub = article.publish_date
    if pub.tzinfo is None or pub.tzinfo.utcoffset(pub) is None:
        pub_utc = pub.replace(tzinfo=timezone.utc)
    else:
        pub_utc = pub.astimezone(timezone.utc)

    if now is None:
        now_utc = datetime.now(timezone.utc)
    else:
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)

    age_days = max(0.0, (now_utc - pub_utc).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 100.0 if age_days == 0 else 0.0
    decay = math.pow(2.0, -age_days / half_life_days)
    return round(max(0.0, min(1.0, decay)) * 100.0, 2)


def calculate_composite_score(metrics: Dict[str, float], config: ScoringConfig) -> float:
    weighted_sum = (
        metrics.get("readability", 0.0) * config.readability_weight +
        metrics.get("ner_density", 0.0) * config.ner_density_weight +
        metrics.get("sentiment", 0.0) * config.sentiment_weight +
        metrics.get("tfidf_relevance", 0.0) * config.tfidf_relevance_weight +
        metrics.get("recency", 0.0) * config.recency_weight
    )
    return round(max(0.0, min(100.0, weighted_sum)), 2)


def analyze_article(
    url: str,
    query: Optional[str] = None,
    config: Optional[ScoringConfig] = None,
    nlp=None,
    vader_analyzer=None,
) -> ScoreCard:
    if config is None:
        config = ScoringConfig()
    parsed = parse_article(url, min_word_count=config.min_word_count)
    metrics = {
        "readability": compute_readability_score(parsed, min_word_count=config.min_word_count),
        "ner_density": compute_ner_density_score(parsed, nlp=nlp),
        "sentiment": compute_sentiment_score(parsed, vader_analyzer=vader_analyzer),
        "tfidf_relevance": compute_tfidf_relevance_score(parsed, query or ""),
        "recency": compute_recency_score(parsed),
    }
    overall = calculate_composite_score(metrics, config)
    return ScoreCard(
        overall_score=overall,
        readability_score=metrics["readability"],
        ner_density_score=metrics["ner_density"],
        sentiment_score=metrics["sentiment"],
        tfidf_relevance_score=metrics["tfidf_relevance"],
        recency_score=metrics["recency"],
        article=parsed,
    )


def batch_analyze(
    urls: List[str],
    query: Optional[str] = None,
    config: Optional[ScoringConfig] = None,
    nlp=None,
    vader_analyzer=None,
) -> List[ScoreCard]:
    if not urls:
        return []
    if config is None:
        config = ScoringConfig()
    results: List[ScoreCard] = []
    parsed_articles = batch_parse_articles(urls, min_word_count=config.min_word_count)
    for article in parsed_articles:
        metrics = {
            "readability": compute_readability_score(article, min_word_count=config.min_word_count),
            "ner_density": compute_ner_density_score(article, nlp=nlp),
            "sentiment": compute_sentiment_score(article, vader_analyzer=vader_analyzer),
            "tfidf_relevance": compute_tfidf_relevance_score(article, query or ""),
            "recency": compute_recency_score(article),
        }
        overall = calculate_composite_score(metrics, config)
        results.append(
            ScoreCard(
                overall_score=overall,
                readability_score=metrics["readability"],
                ner_density_score=metrics["ner_density"],
                sentiment_score=metrics["sentiment"],
                tfidf_relevance_score=metrics["tfidf_relevance"],
                recency_score=metrics["recency"],
                article=article,
            )
        )
    return results
