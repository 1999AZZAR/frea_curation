import os
import sys


def _ensure_project_root_on_path() -> None:
    """Prepend the repository root to sys.path for test imports.

    Allows tests to import `app`, `models`, `config`, and `curator.*`
    without requiring editable installs.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_path()


