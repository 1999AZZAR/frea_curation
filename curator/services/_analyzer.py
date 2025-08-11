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
import os
from typing import Dict, List, Optional, Tuple

from curator.core.models import Article, ScoreCard, ScoringConfig, Entity
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


def extract_entities(article: Article, nlp=None, max_entities: int = 50) -> None:
    """Populate article.entities using spaCy NER if available.

    Confidence is set to 1.0 as spaCy does not expose per-entity confidence by default.
    """
    if nlp is None or not article.content:
        return
    try:
        doc = nlp(article.content)
        ents = []
        for ent in getattr(doc, "ents", [])[:max_entities]:
            text = (ent.text or "").strip()
            label = (ent.label_ or "").strip()
            if not text or not label:
                continue
            try:
                ents.append(Entity(text=text, label=label, confidence=1.0))
            except Exception:
                continue
        if ents:
            article.entities = ents
    except Exception:
        # Leave entities as-is on failure
        return


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


_EMBED_MODEL = None  # Lazy global cache for sentence-transformers model


def _get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        _EMBED_MODEL = SentenceTransformer(model_name)
        return _EMBED_MODEL
    except Exception:
        return None


def compute_embeddings_relevance_score(article: Article, query: str) -> float:
    """Compute semantic relevance via sentence-transformers cosine similarity.

    Returns 0.0 if model or inputs are unavailable.
    """
    if not article.content or not query or not query.strip():
        return 0.0
    model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    model = _get_sentence_transformer(model_name)
    if model is None:
        return 0.0
    try:
        # Encode and compute cosine similarity
        emb_article = model.encode([article.content], normalize_embeddings=True)
        emb_query = model.encode([query], normalize_embeddings=True)
        # Cosine similarity is dot product after normalization
        sim = float((emb_article @ emb_query.T)[0][0])
        return round(max(0.0, min(1.0, sim)) * 100.0, 2)
    except Exception:
        return 0.0


def compute_relevance_score(article: Article, query: str) -> float:
    """Select relevance scoring method based on environment configuration.

    If USE_EMBEDDINGS_RELEVANCE is set to a truthy value and a sentence-transformers
    model is available, use embeddings; otherwise fall back to TF-IDF.
    """
    use_embeddings = os.environ.get("USE_EMBEDDINGS_RELEVANCE", "").strip().lower() in {"1", "true", "yes", "on"}
    if use_embeddings:
        score = compute_embeddings_relevance_score(article, query)
        if score > 0.0:
            return score
    return compute_tfidf_relevance_score(article, query)


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
    # Enrich with entities if possible
    extract_entities(parsed, nlp=nlp)
    metrics = {
        "readability": compute_readability_score(parsed, min_word_count=config.min_word_count),
        "ner_density": compute_ner_density_score(parsed, nlp=nlp),
        "sentiment": compute_sentiment_score(parsed, vader_analyzer=vader_analyzer),
        # Keep key name for backward compatibility in API even if using embeddings under the hood
        "tfidf_relevance": compute_relevance_score(parsed, query or ""),
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
    apply_diversity: Optional[bool] = None,
) -> List[ScoreCard]:
    if not urls:
        return []
    if config is None:
        config = ScoringConfig()
    results: List[ScoreCard] = []
    parsed_articles = batch_parse_articles(urls, min_word_count=config.min_word_count)
    for article in parsed_articles:
        extract_entities(article, nlp=nlp)
        metrics = {
            "readability": compute_readability_score(article, min_word_count=config.min_word_count),
            "ner_density": compute_ner_density_score(article, nlp=nlp),
            "sentiment": compute_sentiment_score(article, vader_analyzer=vader_analyzer),
            # Keep key name for backward compatibility in API even if using embeddings under the hood
            "tfidf_relevance": compute_relevance_score(article, query or ""),
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
    # Optionally apply duplicate collapse and domain diversity caps
    if apply_diversity is None:
        apply_diversity = os.environ.get("DIVERSIFY_RESULTS", "1").strip().lower() in {"1", "true", "yes", "on"}
    if apply_diversity:
        return _apply_diversity_and_dedup(results)
    return results


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith('www.') else netloc
    except Exception:
        return ""


def _apply_diversity_and_dedup(cards: List[ScoreCard], domain_cap: Optional[int] = None, sim_threshold: Optional[float] = None) -> List[ScoreCard]:
    """Collapse near-duplicates and cap per-domain items to improve diversity.

    - Domain cap: limit how many items from the same domain appear
    - Near-duplicate collapse: if embeddings are available, suppress items whose
      cosine similarity to a kept item exceeds sim_threshold.
    Fallback to simple URL-based dedup if embeddings are unavailable.
    """
    if not cards:
        return []

    # Read caps from environment if not provided
    if domain_cap is None:
        try:
            domain_cap = int(os.environ.get("DOMAIN_CAP", 2))
        except Exception:
            domain_cap = 2
    if sim_threshold is None:
        try:
            sim_threshold = float(os.environ.get("DUP_SIM_THRESHOLD", 0.97))
        except Exception:
            sim_threshold = 0.97

    kept: List[ScoreCard] = []
    domain_counts: Dict[str, int] = {}
    # Prepare embeddings if available
    model = _get_sentence_transformer(os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2"))
    embeddings: List[Tuple[ScoreCard, List[float]]] = []
    if model is not None:
        try:
            texts = [c.article.content or c.article.title or c.article.url for c in cards]
            vecs = model.encode(texts, normalize_embeddings=True)
            embeddings = list(zip(cards, vecs))
        except Exception:
            embeddings = []

    # Iterate in descending overall score order for stable suppression
    cards_sorted = sorted(cards, key=lambda c: c.overall_score, reverse=True)
    kept_vecs: List[List[float]] = []
    for card in cards_sorted:
        domain = _extract_domain(card.article.url)
        if domain and domain_counts.get(domain, 0) >= domain_cap:
            continue
        # Duplicate suppression
        is_duplicate = False
        if embeddings:
            try:
                idx = cards.index(card)
                vec = embeddings[idx][1]
                # Compare with kept vectors
                for kv in kept_vecs:
                    # cosine sim for normalized vectors is dot product
                    sim = float(sum(a*b for a, b in zip(vec, kv)))
                    if sim >= sim_threshold:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
            except Exception:
                # Fallback to simple check by title/url
                pass
        else:
            # Basic fallback: duplicate by exact title or URL
            if any(k.article.url == card.article.url or k.article.title == card.article.title for k in kept):
                continue

        kept.append(card)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if embeddings:
            try:
                idx = cards.index(card)
                kept_vecs.append(embeddings[idx][1])
            except Exception:
                pass
    return kept
