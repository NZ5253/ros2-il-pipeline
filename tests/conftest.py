"""
Shared pytest fixtures and path setup.

Adds the il_pipeline package root to sys.path so tests can
`import il_pipeline.*` without first running `colcon build`.

After `colcon build && source install/setup.bash`, the package is on
PYTHONPATH automatically and this conftest is a no-op.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_PARENT = REPO_ROOT / "il_pipeline"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))
