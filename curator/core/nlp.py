"""
NLP utilities for spaCy, NLTK VADER, and SentenceTransformers initialization under package namespace.
"""

from __future__ import annotations

from typing import Any, Optional


def get_spacy_model(model_name: str = "en_core_web_sm") -> Optional[Any]:
    """Return a spaCy language model instance or None if unavailable."""
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    try:
        return spacy.load(model_name)
    except Exception:
        return None


def get_vader_analyzer() -> Optional[Any]:
    """Return an NLTK VADER SentimentIntensityAnalyzer or None if unavailable."""
    try:
        import nltk  # type: ignore
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception:
        return None

    try:
        # Avoid downloads in constrained environments
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except Exception:
            return None
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Optional[Any]:
    """Return a SentenceTransformer model instance or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


