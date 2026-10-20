import os
from pathlib import Path

from app import database, ml_service

BACKEND = Path(__file__).resolve().parents[1]


def test_database_and_models_use_temp_dirs():
    assert BACKEND not in database.DATABASE_PATH.resolve().parents
    assert BACKEND not in ml_service.MODELS.resolve().parents
    assert str(database.DATABASE_PATH) == os.environ["SMART_FEEDBACK_DB"]
    assert str(ml_service.MODELS) == os.environ["SMART_FEEDBACK_MODELS"]
