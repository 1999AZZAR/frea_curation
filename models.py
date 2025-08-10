"""
Compatibility re-exports for models under `curator.core.models`.

Tests and legacy code that import from `models` will get the same class
objects as the canonical package to preserve isinstance semantics.
"""

from curator.core.models import (  # noqa: F401
    Article,
    Entity,
    ScoreCard,
    ScoringConfig,
)