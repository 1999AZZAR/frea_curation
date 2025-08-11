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
from curator.services._parser import ArticleParsingError
from curator.core.nlp import get_sentence_transformer


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
    """Get cached sentence transformer model with lazy initialization."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    
    _EMBED_MODEL = get_sentence_transformer(model_name)
    return _EMBED_MODEL


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
    """
    Main orchestrator function for analyzing a single article.
    
    Implements comprehensive error handling and graceful degradation:
    - If parsing fails, raises ArticleParsingError
    - If individual scoring components fail, uses fallback scores
    - Ensures all scores are valid (0-100 range)
    
    Args:
        url: Article URL to analyze
        query: Optional query for relevance scoring
        config: Scoring configuration (uses defaults if None)
        nlp: Optional spaCy model for NER
        vader_analyzer: Optional VADER sentiment analyzer
        
    Returns:
        ScoreCard with all scoring components
        
    Raises:
        ArticleParsingError: If article cannot be parsed
        ValueError: If URL is invalid
    """
    if config is None:
        config = ScoringConfig()
    
    # Parse article - this can raise ArticleParsingError
    parsed = parse_article(url, min_word_count=config.min_word_count)
    
    # Enrich with entities if possible (graceful degradation)
    try:
        extract_entities(parsed, nlp=nlp)
    except Exception as e:
        # Log but don't fail - entities are optional
        import logging
        logging.getLogger(__name__).warning(f"Entity extraction failed for {url}: {e}")
    
    # Calculate individual scores with graceful degradation
    metrics = {}
    
    # Readability score (should always work)
    try:
        metrics["readability"] = compute_readability_score(parsed, min_word_count=config.min_word_count)
    except Exception:
        metrics["readability"] = 0.0
    
    # NER density score (graceful degradation if spaCy unavailable)
    try:
        metrics["ner_density"] = compute_ner_density_score(parsed, nlp=nlp)
    except Exception:
        metrics["ner_density"] = 0.0
    
    # Sentiment score (graceful degradation if VADER unavailable)
    try:
        metrics["sentiment"] = compute_sentiment_score(parsed, vader_analyzer=vader_analyzer)
    except Exception:
        metrics["sentiment"] = 50.0  # Neutral fallback
    
    # Relevance score (graceful degradation)
    try:
        metrics["tfidf_relevance"] = compute_relevance_score(parsed, query or "")
    except Exception:
        metrics["tfidf_relevance"] = 0.0
    
    # Recency score (should always work)
    try:
        metrics["recency"] = compute_recency_score(parsed)
    except Exception:
        metrics["recency"] = 100.0  # Assume recent if date parsing fails
    
    # Ensure all scores are valid
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 100.0):
            metrics[key] = 0.0
    
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
    """
    Batch processing function for analyzing multiple articles efficiently.
    
    Implements comprehensive error handling and graceful degradation:
    - Continues processing even if individual articles fail
    - Uses fallback scores for failed scoring components
    - Logs failures for debugging without stopping the batch
    
    Args:
        urls: List of article URLs to analyze
        query: Optional query for relevance scoring
        config: Scoring configuration (uses defaults if None)
        nlp: Optional spaCy model for NER
        vader_analyzer: Optional VADER sentiment analyzer
        apply_diversity: Whether to apply diversity filtering (auto-detect if None)
        
    Returns:
        List of ScoreCard objects for successfully processed articles
    """
    if not urls:
        return []
    if config is None:
        config = ScoringConfig()
    
    import logging
    logger = logging.getLogger(__name__)
    
    results: List[ScoreCard] = []
    
    # Parse articles in batch - this handles individual failures gracefully
    parsed_articles = batch_parse_articles(urls, min_word_count=config.min_word_count)
    
    logger.info(f"Successfully parsed {len(parsed_articles)}/{len(urls)} articles for batch analysis")
    
    for i, article in enumerate(parsed_articles):
        try:
            # Enrich with entities if possible (graceful degradation)
            try:
                extract_entities(article, nlp=nlp)
            except Exception as e:
                logger.warning(f"Entity extraction failed for article {i+1}: {e}")
            
            # Calculate individual scores with graceful degradation
            metrics = {}
            
            # Readability score (should always work)
            try:
                metrics["readability"] = compute_readability_score(article, min_word_count=config.min_word_count)
            except Exception as e:
                logger.warning(f"Readability scoring failed for article {i+1}: {e}")
                metrics["readability"] = 0.0
            
            # NER density score (graceful degradation if spaCy unavailable)
            try:
                metrics["ner_density"] = compute_ner_density_score(article, nlp=nlp)
            except Exception as e:
                logger.warning(f"NER density scoring failed for article {i+1}: {e}")
                metrics["ner_density"] = 0.0
            
            # Sentiment score (graceful degradation if VADER unavailable)
            try:
                metrics["sentiment"] = compute_sentiment_score(article, vader_analyzer=vader_analyzer)
            except Exception as e:
                logger.warning(f"Sentiment scoring failed for article {i+1}: {e}")
                metrics["sentiment"] = 50.0  # Neutral fallback
            
            # Relevance score (graceful degradation)
            try:
                metrics["tfidf_relevance"] = compute_relevance_score(article, query or "")
            except Exception as e:
                logger.warning(f"Relevance scoring failed for article {i+1}: {e}")
                metrics["tfidf_relevance"] = 0.0
            
            # Recency score (should always work)
            try:
                metrics["recency"] = compute_recency_score(article)
            except Exception as e:
                logger.warning(f"Recency scoring failed for article {i+1}: {e}")
                metrics["recency"] = 100.0  # Assume recent if date parsing fails
            
            # Ensure all scores are valid
            for key, value in metrics.items():
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 100.0):
                    logger.warning(f"Invalid score {key}={value} for article {i+1}, using 0.0")
                    metrics[key] = 0.0
            
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
            
        except Exception as e:
            # Log the error but continue with other articles
            logger.error(f"Failed to analyze article {i+1} ({article.url}): {e}")
            continue
    
    logger.info(f"Successfully analyzed {len(results)}/{len(parsed_articles)} parsed articles")
    
    # Optionally apply duplicate collapse and domain diversity caps
    if apply_diversity is None:
        apply_diversity = os.environ.get("DIVERSIFY_RESULTS", "1").strip().lower() in {"1", "true", "yes", "on"}
    if apply_diversity:
        try:
            return _apply_diversity_and_dedup(results)
        except Exception as e:
            logger.warning(f"Diversity filtering failed: {e}, returning unfiltered results")
            return results
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
