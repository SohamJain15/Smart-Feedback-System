import csv
import io

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline_api import rows_to_csv


def _save_and_label(client, text, category="Library", rating=2, labels=("negative", "library", "high")):
    saved = client.post("/api/feedback", json={"text": text, "category": category, "rating": rating}).json()
    sentiment, topic, priority = labels
    assert client.put(
        f"/api/feedback/{saved['id']}/label",
        json={"actual_sentiment": sentiment, "actual_topic": topic, "actual_priority": priority},
    ).status_code == 200
    return saved


def _numbers(history):
    return [int(row["version"][1:]) for row in history if row["version"][1:].isdigit()]


def test_pipeline_info_shape():
    with TestClient(app) as client:
        info = client.get("/api/models/pipeline-info").json()
        assert set(info) == {"active_version", "aggregate_f1", "next_version", "newly_labelled", "threshold", "jenkins_managed"}
        assert info["active_version"] == client.get("/api/health").json()["active_version"]
        history = client.get("/api/models/history").json()
        assert int(info["next_version"][1:]) > max(_numbers(history))
        assert 0 < info["aggregate_f1"] <= 1


def test_existing_retraining_status_unchanged():
    with TestClient(app) as client:
        status = client.get("/api/models/retraining-status").json()
        assert set(status) == {"newly_labelled", "threshold", "recommended", "active_version"}
        assert status["threshold"] == 20


JENKINS_VARS = {"JENKINS_URL": "http://localhost:8080/", "JENKINS_USER": "gyan", "JENKINS_TOKEN": "tok123", "JENKINS_JOB": "feedback-retrain"}


def _set_jenkins_env(monkeypatch, **overrides):
    for name, value in {**JENKINS_VARS, **overrides}.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_jenkins_managed_needs_all_four_variables(monkeypatch):
    with TestClient(app) as client:
        for name in JENKINS_VARS:
            monkeypatch.delenv(name, raising=False)
        assert client.get("/api/models/pipeline-info").json()["jenkins_managed"] is False
        monkeypatch.setenv("JENKINS_URL", "http://localhost:8080")  # URL alone is not enough
        assert client.get("/api/models/pipeline-info").json()["jenkins_managed"] is False
        _set_jenkins_env(monkeypatch)
        assert client.get("/api/models/pipeline-info").json()["jenkins_managed"] is True


def test_export_only_reviewed():
    with TestClient(app) as client:
        client.post("/api/feedback", json={"text": "unreviewed-only-marker feedback", "category": "Library", "rating": 3})
        _save_and_label(client, "reviewed-marker feedback about the library")
        response = client.get("/api/feedback/export")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        rows = list(csv.reader(io.StringIO(response.text)))
        assert rows[0] == ["text", "sentiment", "topic", "priority"]
        texts = [row[0] for row in rows[1:]]
        assert "reviewed-marker feedback about the library" in texts
        assert "unreviewed-only-marker feedback" not in texts


def test_export_round_trips_awkward_text():
    awkward = 'Wifi, "slow"\nagain'
    with TestClient(app) as client:
        _save_and_label(client, awkward)
        rows = list(csv.reader(io.StringIO(client.get("/api/feedback/export").text)))
        assert awkward in [row[0] for row in rows[1:]]


def test_rows_to_csv_empty_is_header_only():
    assert rows_to_csv([]) == "text,sentiment,topic,priority\n"


# ---- activate / rollback / report -------------------------------------------------
import pytest

from app.ml_service import MODELS
from ml.train import train_candidate

ACTIVATE = "/api/models/pipeline-activate"
REPORT = "/api/models/pipeline-report"


def _candidate(version):
    train_candidate(None, MODELS, version)
    return MODELS / version


def _active_rows(client):
    return [row for row in client.get("/api/models/history").json() if row["status"] == "ACTIVE"]


def test_activate_switches_live_model():
    _candidate("v901")
    with TestClient(app) as client:
        response = client.post(ACTIVATE, json={"version": "v901", "note": "Jenkins build #1"})
        assert response.status_code == 200
        assert response.json()["active_version"] == "v901"
        assert client.get("/api/health").json()["active_version"] == "v901"
        assert client.get("/api/models/current").json()["version"] == "v901"
        active = _active_rows(client)
        assert len(active) == 3 and {row["version"] for row in active} == {"v901"}
        runs = client.get("/api/models/training-runs").json()
        run = next(r for r in runs if r["version"] == "v901")
        assert run["deployed"] is True and run["status"] == "COMPLETED"
        assert run["stage"] == "Jenkins build #1"
        assert run["new_f1"] is not None and run["old_f1"] is not None
        predictions = client.post(
            "/api/feedback/analyze", json={"text": "The lab computers are slow", "category": "Laboratory", "rating": 2}
        ).json()["predictions"]
        assert set(predictions) == {"sentiment", "topic", "priority"}


def test_rollback_restores_previous_and_records_no_run():
    _candidate("v902")
    with TestClient(app) as client:
        previous = client.get("/api/health").json()["active_version"]
        assert client.post(ACTIVATE, json={"version": "v902"}).status_code == 200
        runs_before = len(client.get("/api/models/training-runs").json())
        response = client.post(ACTIVATE, json={"version": previous, "kind": "rollback"})
        assert response.status_code == 200
        assert client.get("/api/health").json()["active_version"] == previous
        active = _active_rows(client)
        assert len(active) == 3 and {row["version"] for row in active} == {previous}
        assert len(client.get("/api/models/training-runs").json()) == runs_before
        undone = next(r for r in client.get("/api/models/training-runs").json() if r["version"] == "v902")
        assert undone["status"] == "ROLLED_BACK" and undone["deployed"] is False
        assert undone["stage"].endswith("(rolled back)")


def test_activate_is_idempotent():
    _candidate("v903")
    with TestClient(app) as client:
        assert client.post(ACTIVATE, json={"version": "v903"}).status_code == 200
        assert client.post(ACTIVATE, json={"version": "v903"}).status_code == 200
        rows = [r for r in client.get("/api/models/history").json() if r["version"] == "v903"]
        assert len(rows) == 3


@pytest.mark.parametrize("bad", ["../x", "latest", "v1;x", "V1", "v", ""])
def test_activate_rejects_bad_versions(bad):
    with TestClient(app) as client:
        assert client.post(ACTIVATE, json={"version": bad}).status_code == 422


def test_activate_missing_folder_404():
    with TestClient(app) as client:
        assert client.post(ACTIVATE, json={"version": "v999"}).status_code == 404


def test_activate_incomplete_folder_422():
    folder = _candidate("v904")
    (folder / "topic.joblib").unlink()
    with TestClient(app) as client:
        before = client.get("/api/health").json()["active_version"]
        assert client.post(ACTIVATE, json={"version": "v904"}).status_code == 422
        assert client.get("/api/health").json()["active_version"] == before


def test_activate_corrupt_joblib_422_and_state_unchanged():
    folder = _candidate("v905")
    (folder / "sentiment.joblib").write_bytes(b"not a model")
    with TestClient(app) as client:
        before = client.get("/api/health").json()["active_version"]
        history_before = client.get("/api/models/history").json()
        assert client.post(ACTIVATE, json={"version": "v905"}).status_code == 422
        assert client.get("/api/health").json()["active_version"] == before
        assert client.get("/api/models/history").json() == history_before


@pytest.mark.parametrize("status", ["REJECTED", "FAILED"])
def test_report_records_rejected_and_failed(status):
    with TestClient(app) as client:
        body = {"version": "v950", "status": status, "old_f1": 0.9, "new_f1": 0.8, "training_samples": 120, "note": "Jenkins build #7"}
        assert client.post(REPORT, json=body).status_code == 200
        runs = client.get("/api/models/training-runs").json()
        run = next(r for r in runs if r["version"] == "v950" and r["status"] == status)
        assert run["deployed"] is False and run["stage"] == "Jenkins build #7"
        assert run["old_f1"] == 0.9 and run["new_f1"] == 0.8 and run["training_samples"] == 120


def test_report_rejects_other_status_422():
    with TestClient(app) as client:
        assert client.post(REPORT, json={"version": "v951", "status": "COMPLETED"}).status_code == 422


def test_activated_model_survives_backend_restart():
    from app.main import initialize_models
    from app.ml_service import ml

    _candidate("v907")
    with TestClient(app) as client:
        assert client.post(ACTIVATE, json={"version": "v907"}).status_code == 200
        ml.version = None
        ml.models = {}
        initialize_models()  # what the app does at startup
        assert ml.version == "v907"
        assert client.get("/api/health").json()["active_version"] == "v907"


def test_rollback_to_app_created_version_without_metrics_json():
    # v1 is trained by the app itself at first startup and has no metrics.json.
    _candidate("v908")
    with TestClient(app) as client:
        assert not (MODELS / "v1" / "metrics.json").exists()
        assert client.post(ACTIVATE, json={"version": "v908"}).status_code == 200
        response = client.post(ACTIVATE, json={"version": "v1", "kind": "rollback"})
        assert response.status_code == 200
        assert client.get("/api/health").json()["active_version"] == "v1"
        assert response.json()["aggregate_f1"] > 0
        assert {row["version"] for row in _active_rows(client)} == {"v1"}


def test_activate_new_version_requires_metrics_json():
    folder = _candidate("v909")
    (folder / "metrics.json").unlink()
    with TestClient(app) as client:
        before = client.get("/api/health").json()["active_version"]
        response = client.post(ACTIVATE, json={"version": "v909"})
        assert response.status_code == 422 and "metrics.json" in response.text
        assert client.get("/api/health").json()["active_version"] == before


# ---- retrain button -> Jenkins ---------------------------------------------------
import base64
import urllib.error


class _FakeResponse:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_retrain_triggers_jenkins(monkeypatch):
    _set_jenkins_env(monkeypatch)
    seen = []
    monkeypatch.setattr("app.pipeline_api.urllib.request.urlopen", lambda request, timeout=None: seen.append(request) or _FakeResponse())
    with TestClient(app) as client:
        runs_before = len(client.get("/api/models/training-runs").json())
        response = client.post("/api/models/retrain")
        assert response.status_code == 200
        assert response.json() == {"status": "REQUESTED", "job": "feedback-retrain"}
        assert len(client.get("/api/models/training-runs").json()) == runs_before  # nothing trained in-process
    request = seen[0]
    assert request.full_url == "http://localhost:8080/job/feedback-retrain/buildWithParameters?FORCE=true"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Basic " + base64.b64encode(b"gyan:tok123").decode()


def test_retrain_quotes_job_name_with_spaces(monkeypatch):
    _set_jenkins_env(monkeypatch, JENKINS_JOB="feedback retrain")
    seen = []
    monkeypatch.setattr("app.pipeline_api.urllib.request.urlopen", lambda request, timeout=None: seen.append(request) or _FakeResponse())
    with TestClient(app) as client:
        assert client.post("/api/models/retrain").status_code == 200
    assert "/job/feedback%20retrain/buildWithParameters" in seen[0].full_url


# Explicit ids + a real (empty) body: on Python 3.10 pytest crashes while auto-naming an HTTPError whose fp is None.
@pytest.mark.parametrize("error", [
    urllib.error.URLError("connection refused"),
    urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, io.BytesIO(b"")),
], ids=["url-error", "http-401"])
def test_retrain_jenkins_error_returns_502(monkeypatch, error):
    _set_jenkins_env(monkeypatch)

    def boom(request, timeout=None):
        raise error
    monkeypatch.setattr("app.pipeline_api.urllib.request.urlopen", boom)
    with TestClient(app) as client:
        response = client.post("/api/models/retrain")
        assert response.status_code == 502
        assert "Could not reach Jenkins" in response.json()["detail"]


def test_deploy_of_already_known_version_is_rejected_409():
    # e.g. the in-app trainer and Jenkins both produced the same version number
    _candidate("v910")
    _candidate("v911")
    with TestClient(app) as client:
        assert client.post(ACTIVATE, json={"version": "v910"}).status_code == 200
        assert client.post(ACTIVATE, json={"version": "v911"}).status_code == 200
        response = client.post(ACTIVATE, json={"version": "v910"})  # kind defaults to "deploy"
        assert response.status_code == 409 and "rollback" in response.json()["detail"]
        assert client.get("/api/health").json()["active_version"] == "v911"
