"""
Configuration management for AI Content Curator (package module)
Handles environment-specific settings and scoring parameters
"""

import os
from dotenv import load_dotenv
from curator.core.models import ScoringConfig


# Load environment variables
load_dotenv()


def load_scoring_config() -> ScoringConfig:
    """Load scoring configuration from environment variables with validation.

    If only a subset of weights are overridden via environment variables and the
    total does not sum to 1.0, automatically adjust one non-overridden weight
    (preferring recency) so the total equals 1.0. This keeps caller-provided
    weights intact while maintaining a valid configuration.
    """
    try:
        # Read raw values (with defaults)
        readability_weight = float(os.environ.get('READABILITY_WEIGHT', 0.15))
        ner_density_weight = float(os.environ.get('NER_DENSITY_WEIGHT', 0.15))
        sentiment_weight = float(os.environ.get('SENTIMENT_WEIGHT', 0.15))
        tfidf_relevance_weight = float(os.environ.get('TFIDF_RELEVANCE_WEIGHT', 0.25))
        recency_weight = float(os.environ.get('RECENCY_WEIGHT', 0.15))
        reputation_weight = float(os.environ.get('REPUTATION_WEIGHT', 0.15))

        # Detect which weights were overridden by env
        overridden = {
            'readability_weight': os.environ.get('READABILITY_WEIGHT') is not None,
            'ner_density_weight': os.environ.get('NER_DENSITY_WEIGHT') is not None,
            'sentiment_weight': os.environ.get('SENTIMENT_WEIGHT') is not None,
            'tfidf_relevance_weight': os.environ.get('TFIDF_RELEVANCE_WEIGHT') is not None,
            'recency_weight': os.environ.get('RECENCY_WEIGHT') is not None,
            'reputation_weight': os.environ.get('REPUTATION_WEIGHT') is not None,
        }

        # Compute total and adjust if necessary only when not all weights are overridden
        weights_total = (
            readability_weight +
            ner_density_weight +
            sentiment_weight +
            tfidf_relevance_weight +
            recency_weight +
            reputation_weight
        )

        # If sum deviates from 1.0 and at least one weight is not overridden,
        # adjust a preferred non-overridden weight to compensate.
        if not all(overridden.values()) and abs(weights_total - 1.0) > 1e-6:
            delta = 1.0 - weights_total
            # Preference order for adjustment
            candidates = [
                ('recency_weight', recency_weight),
                ('reputation_weight', reputation_weight),
                ('tfidf_relevance_weight', tfidf_relevance_weight),
                ('ner_density_weight', ner_density_weight),
                ('sentiment_weight', sentiment_weight),
                ('readability_weight', readability_weight),
            ]

            for name, value in candidates:
                if not overridden[name]:
                    new_value = max(0.0, min(1.0, value + delta))
                    # Apply the adjustment
                    if name == 'recency_weight':
                        recency_weight = new_value
                    elif name == 'reputation_weight':
                        reputation_weight = new_value
                    elif name == 'tfidf_relevance_weight':
                        tfidf_relevance_weight = new_value
                    elif name == 'ner_density_weight':
                        ner_density_weight = new_value
                    elif name == 'sentiment_weight':
                        sentiment_weight = new_value
                    elif name == 'readability_weight':
                        readability_weight = new_value
                    break

        # Topic-aware recency calibration
        # Example env usage:
        #   DEFAULT_RECENCY_HALF_LIFE_DAYS=7
        #   TOPIC_HALF_LIFE_DAYS=finance:1,tech:3,research:14,ai:5
        try:
            default_half_life = float(os.environ.get('DEFAULT_RECENCY_HALF_LIFE_DAYS', 7.0))
        except Exception:
            default_half_life = 7.0

        topic_half_life_days = {}
        raw = os.environ.get('TOPIC_HALF_LIFE_DAYS', '')
        if isinstance(raw, str) and raw.strip():
            for pair in raw.split(','):
                if not pair.strip():
                    continue
                if ':' not in pair:
                    continue
                k, v = pair.split(':', 1)
                k = k.strip()
                try:
                    days = float(v.strip())
                    if k:
                        topic_half_life_days[k] = days
                except Exception:
                    continue

        return ScoringConfig(
            readability_weight=readability_weight,
            ner_density_weight=ner_density_weight,
            sentiment_weight=sentiment_weight,
            tfidf_relevance_weight=tfidf_relevance_weight,
            recency_weight=recency_weight,
            reputation_weight=reputation_weight,
            min_word_count=int(os.environ.get('MIN_WORD_COUNT', 300)),
            max_articles_per_topic=int(os.environ.get('MAX_ARTICLES_PER_TOPIC', 20)),
            default_recency_half_life_days=default_half_life,
            topic_half_life_days=topic_half_life_days,
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid scoring configuration: {e}")


def validate_required_env_vars():
    """Validate that required environment variables are present."""
    required_vars = []  # Currently none mandatory
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


