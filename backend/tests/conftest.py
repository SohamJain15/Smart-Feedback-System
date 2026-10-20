"""Run every test against a throw-away database and models folder.

This module is imported by pytest before any test module, so the environment is
set before ``app.database`` / ``app.ml_service`` read it.
"""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

_TMP = Path(tempfile.mkdtemp(prefix="feedback-tests-"))
os.environ["SMART_FEEDBACK_DB"] = str(_TMP / "test.db")
os.environ["SMART_FEEDBACK_MODELS"] = str(_TMP / "models")
