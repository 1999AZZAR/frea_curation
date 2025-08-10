"""
Configuration management for AI Content Curator
Handles environment-specific settings and scoring parameters
"""

import os
from dotenv import load_dotenv
from models import ScoringConfig

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
        readability_weight = float(os.environ.get('READABILITY_WEIGHT', 0.2))
        ner_density_weight = float(os.environ.get('NER_DENSITY_WEIGHT', 0.2))
        sentiment_weight = float(os.environ.get('SENTIMENT_WEIGHT', 0.15))
        tfidf_relevance_weight = float(os.environ.get('TFIDF_RELEVANCE_WEIGHT', 0.25))
        recency_weight = float(os.environ.get('RECENCY_WEIGHT', 0.2))

        # Detect which weights were overridden by env
        overridden = {
            'readability_weight': os.environ.get('READABILITY_WEIGHT') is not None,
            'ner_density_weight': os.environ.get('NER_DENSITY_WEIGHT') is not None,
            'sentiment_weight': os.environ.get('SENTIMENT_WEIGHT') is not None,
            'tfidf_relevance_weight': os.environ.get('TFIDF_RELEVANCE_WEIGHT') is not None,
            'recency_weight': os.environ.get('RECENCY_WEIGHT') is not None,
        }

        # Compute total and adjust if necessary only when not all weights are overridden
        weights_total = (
            readability_weight +
            ner_density_weight +
            sentiment_weight +
            tfidf_relevance_weight +
            recency_weight
        )

        # If sum deviates from 1.0 and at least one weight is not overridden,
        # adjust a preferred non-overridden weight to compensate.
        if not all(overridden.values()) and abs(weights_total - 1.0) > 1e-6:
            delta = 1.0 - weights_total
            # Preference order for adjustment
            candidates = [
                ('recency_weight', recency_weight),
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
                    elif name == 'tfidf_relevance_weight':
                        tfidf_relevance_weight = new_value
                    elif name == 'ner_density_weight':
                        ner_density_weight = new_value
                    elif name == 'sentiment_weight':
                        sentiment_weight = new_value
                    elif name == 'readability_weight':
                        readability_weight = new_value
                    break

        return ScoringConfig(
            readability_weight=readability_weight,
            ner_density_weight=ner_density_weight,
            sentiment_weight=sentiment_weight,
            tfidf_relevance_weight=tfidf_relevance_weight,
            recency_weight=recency_weight,
            min_word_count=int(os.environ.get('MIN_WORD_COUNT', 300)),
            max_articles_per_topic=int(os.environ.get('MAX_ARTICLES_PER_TOPIC', 20))
        )
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid scoring configuration: {e}")


def validate_required_env_vars():
    """Validate that required environment variables are present."""
    required_vars = ['NEWS_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return True

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    NEWS_API_KEY = 'test-api-key'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])