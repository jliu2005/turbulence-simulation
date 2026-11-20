"""
Pytest configuration.

Putting this file at the repository root, together with pytest's rootdir
detection, makes ``import postprocessing`` work in tests no matter which
directory ``pytest`` is launched from - no editable install required.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
