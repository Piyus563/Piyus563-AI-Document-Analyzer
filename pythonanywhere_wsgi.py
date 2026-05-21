"""
PythonAnywhere WSGI entry point.

On PythonAnywhere, copy this file's contents into the WSGI configuration file
or import it from there after replacing YOUR_USERNAME with your username.
"""

import sys
from pathlib import Path

USERNAME = "YOUR_USERNAME"
PROJECT_DIR = Path(f"/home/{USERNAME}/Piyus563-AI-Document-Analyzer")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app import app as application  # noqa: E402
