"""Public parser API that forwards to `._parser` implementation.

This ensures single source of truth and consistent patch points in tests.
"""

from ._parser import *  # noqa: F401,F403