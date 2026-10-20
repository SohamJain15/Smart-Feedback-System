import json

import pandas as pd
import pytest

from app.ml_service import base_dataset
from ml import gate, train

FILES = ["sentiment.joblib", "topic.joblib", "priority.joblib"]


def _write_reviewed(path, rows):
    pd.DataFrame(rows, columns=["text", "sentiment", "topic", "priority"]).to_csv(path, index=False)


def _first_labels():
    row = base_dataset().iloc[0]
    return row["sentiment"], row["topic"], row["priority"]


def test_train_base_only(tmp_path):
    result = train.train_candidate(None, tmp_path, "v50")
    folder = tmp_path / "v50"
    assert all((folder / name).exists() for name in FILES)
    saved = json.loads((folder / "metrics.json").read_text())
    assert saved == result
    assert result["version"] == "v50"
    assert result["training_samples"] == len(base_dataset())
    assert 0 < result["aggregate_f1"] <= 1
    assert set(result["metrics"]) == {"sentiment", "topic", "priority"}
    assert set(result["metrics"]["topic"]) == {"accuracy", "precision", "recall", "f1_score"}


def test_train_with_reviewed_rows(tmp_path):
    s, t, p = _first_labels()
    csv_path = tmp_path / "reviewed.csv"
    _write_reviewed(csv_path, [(f"extra feedback number {i}", s, t, p) for i in range(3)])
    result = train.train_candidate(csv_path, tmp_path / "out", "v51")
    assert result["training_samples"] == len(base_dataset()) + 3


def test_header_only_csv_equals_base_only(tmp_path):
    csv_path = tmp_path / "reviewed.csv"
    _write_reviewed(csv_path, [])
    result = train.train_candidate(csv_path, tmp_path / "out", "v52")
    assert result["training_samples"] == len(base_dataset())


def test_missing_column_raises_value_error(tmp_path):
    csv_path = tmp_path / "reviewed.csv"
    pd.DataFrame({"text": ["x"], "sentiment": ["negative"], "topic": ["library"]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="priority"):
        train.train_candidate(csv_path, tmp_path / "out", "v53")


def test_blank_text_main_returns_1(tmp_path, capsys):
    s, t, p = _first_labels()
    csv_path = tmp_path / "reviewed.csv"
    _write_reviewed(csv_path, [("", s, t, p)])
    code = train.main(["--reviewed-csv", str(csv_path), "--out-dir", str(tmp_path / "out"), "--version", "v54"])
    assert code == 1
    assert "ERROR" in capsys.readouterr().out


def test_main_success_returns_0(tmp_path):
    code = train.main(["--out-dir", str(tmp_path), "--version", "v55"])
    assert code == 0
    assert (tmp_path / "v55" / "metrics.json").exists()


def test_gate_decide():
    assert gate.decide(0.9, 0.8)[0] is True
    assert gate.decide(0.8, 0.8)[0] is False
    assert gate.decide(0.7, 0.8)[0] is False


def _metrics_file(tmp_path, f1):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"aggregate_f1": f1}))
    return path


def test_gate_main_exit_codes(tmp_path):
    path = _metrics_file(tmp_path, 0.9)
    out = tmp_path / "gate.txt"
    assert gate.main(["--metrics", str(path), "--current-f1", "0.8", "--out", str(out)]) == 0
    assert out.read_text().strip()
    assert gate.main(["--metrics", str(path), "--current-f1", "0.9"]) == 10
    assert gate.main(["--metrics", str(tmp_path / "missing.json"), "--current-f1", "0.8"]) == 1


def test_reviewed_text_that_looks_like_null_is_kept(tmp_path):
    s, t, p = _first_labels()
    csv_path = tmp_path / "reviewed.csv"
    _write_reviewed(csv_path, [("None", s, t, p), ("NA", s, t, p)])
    result = train.train_candidate(csv_path, tmp_path / "out", "v56")
    assert result["training_samples"] == len(base_dataset()) + 2
