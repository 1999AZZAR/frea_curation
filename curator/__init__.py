"""Curator application package."""


__all__ = [
    "core",
    "services",
    "web",
]

# Convenience re-export for config loader to ease imports
try:
    from curator.core.config import load_scoring_config  # noqa: F401
except Exception:
    pass

