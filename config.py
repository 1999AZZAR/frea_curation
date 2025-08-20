"""
Backwards-compatibility shim for configuration.

Prefer importing from `curator.core.config` going forward.
This module delegates to the package implementation.
"""

from curator.core.config import *  # noqa: F401,F403