"""Pipeline gate: the candidate model produced by Jenkins must load and predict.

Skipped in normal runs; Jenkinsfile.retrain sets CANDIDATE_MODEL_DIR.
"""
import os
from pathlib import Path

import joblib
import pytest

FOLDER = os.environ.get("CANDIDATE_MODEL_DIR")


@pytest.mark.skipif(not FOLDER, reason="CANDIDATE_MODEL_DIR not set (only used by the retrain pipeline)")
def test_candidate_loads_and_predicts():
    folder = Path(FOLDER)
    for target in ("sentiment", "topic", "priority"):
        model = joblib.load(folder / f"{target}.joblib")
        text = "category_library " * 8 + "The library wifi keeps dropping" if target == "topic" else "The library wifi keeps dropping"
        probabilities = model.predict_proba([text])[0]
        assert abs(sum(probabilities) - 1) < 1e-6
        assert str(model.classes_[int(probabilities.argmax())])
