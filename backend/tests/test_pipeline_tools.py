import zipfile

import pytest

from ml import pipeline_tools as tools
from ml.train import train_candidate

APP = "http://app.test"


@pytest.fixture
def candidate(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate")
    train_candidate(None, root, "v70")
    return root / "v70"


def test_info_writes_key_value_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "http_json", lambda method, url, payload=None, timeout=30: {
        "active_version": "v7", "aggregate_f1": 0.8125, "next_version": "v8",
        "newly_labelled": 21, "threshold": 20, "jenkins_managed": True,
    })
    out = tmp_path / "info.env"
    assert tools.cmd_info(APP, out) == 0
    lines = out.read_text().splitlines()
    assert lines == ["active_version=v7", "aggregate_f1=0.8125", "next_version=v8", "newly_labelled=21", "threshold=20"]


def test_export_saves_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "http_bytes", lambda url, timeout=60: b"text,sentiment,topic,priority\r\n")
    out = tmp_path / "reviewed.csv"
    assert tools.cmd_export(APP, out) == 0
    assert out.read_bytes() == b"text,sentiment,topic,priority\r\n"


def test_package_zip_contents(tmp_path, candidate):
    zip_path = tools.cmd_package(candidate, "v70", tmp_path)
    assert zip_path == tmp_path / "model-v70.zip"
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == sorted(
            f"v70/{name}" for name in ["sentiment.joblib", "topic.joblib", "priority.joblib", "metrics.json"]
        )


def test_package_rejects_incomplete_folder(tmp_path, candidate):
    (candidate / "topic.joblib").unlink()
    with pytest.raises(RuntimeError, match="topic.joblib"):
        tools.cmd_package(candidate, "v70", tmp_path)


def test_deploy_copies_then_activates(tmp_path, monkeypatch, candidate):
    calls = []
    monkeypatch.setattr(tools, "http_json", lambda method, url, payload=None, timeout=30: calls.append((method, url, payload)) or {})
    app_models = tmp_path / "app-models"
    assert tools.cmd_deploy(APP, "v70", app_models, "Jenkins build #3", model_dir=candidate) == 0
    assert (app_models / "v70" / "metrics.json").exists()
    assert not (app_models / "v70.partial").exists()
    assert calls == [("POST", f"{APP}/api/models/pipeline-activate", {"version": "v70", "kind": "deploy", "note": "Jenkins build #3"})]


def test_deploy_replaces_stale_folder(tmp_path, monkeypatch, candidate):
    monkeypatch.setattr(tools, "http_json", lambda *a, **k: {})
    app_models = tmp_path / "app-models"
    (app_models / "v70").mkdir(parents=True)
    (app_models / "v70" / "stale.txt").write_text("old")
    assert tools.cmd_deploy(APP, "v70", app_models, "n", model_dir=candidate) == 0
    assert not (app_models / "v70" / "stale.txt").exists()


def test_deploy_activation_failure_returns_1_without_partial(tmp_path, monkeypatch, candidate, capsys):
    def fail(*a, **k):
        raise RuntimeError("HTTP 422: model not usable")
    monkeypatch.setattr(tools, "http_json", fail)
    app_models = tmp_path / "app-models"
    assert tools.cmd_deploy(APP, "v70", app_models, "n", model_dir=candidate) == 1
    assert "ERROR: HTTP 422" in capsys.readouterr().out
    assert not list(app_models.glob("*.partial"))


def test_deploy_copy_failure_leaves_no_partial_and_keeps_old_folder(tmp_path, monkeypatch, candidate, capsys):
    app_models = tmp_path / "app-models"
    (app_models / "v70").mkdir(parents=True)
    (app_models / "v70" / "keep.txt").write_text("keep")

    def broken_copy(src, dst, *a, **k):
        dst.mkdir()
        raise OSError("disk full")
    monkeypatch.setattr(tools.shutil, "copytree", broken_copy)
    monkeypatch.setattr(tools, "http_json", lambda *a, **k: pytest.fail("must not activate"))
    assert tools.cmd_deploy(APP, "v70", app_models, "n", model_dir=candidate) == 1
    assert "ERROR" in capsys.readouterr().out
    assert not list(app_models.glob("*.partial"))
    assert (app_models / "v70" / "keep.txt").exists()


def test_rollback_does_not_copy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "http_json", lambda method, url, payload=None, timeout=30: calls.append(payload) or {})
    app_models = tmp_path / "app-models"
    assert tools.cmd_deploy(APP, "v6", app_models, "rollback", kind="rollback") == 0
    assert not app_models.exists()
    assert calls == [{"version": "v6", "kind": "rollback", "note": "rollback"}]


def _health_script(monkeypatch, versions, predictions=("sentiment", "topic", "priority")):
    seen = iter(versions)

    def fake(method, url, payload=None, timeout=30):
        if url.endswith("/api/health"):
            return {"active_version": next(seen, versions[-1])}
        return {"predictions": {name: {} for name in predictions}}
    monkeypatch.setattr(tools, "http_json", fake)


def test_wait_healthy_succeeds_after_restart_delay(monkeypatch):
    _health_script(monkeypatch, ["v6", "v6", "v9"])
    assert tools.cmd_wait_healthy(APP, "v9", timeout=5, interval=0.01) == 0


def test_wait_healthy_times_out(monkeypatch, capsys):
    _health_script(monkeypatch, ["v6"])
    assert tools.cmd_wait_healthy(APP, "v9", timeout=0.2, interval=0.05) == 1
    assert "v9" in capsys.readouterr().out


def test_wait_healthy_requires_three_predictions(monkeypatch):
    _health_script(monkeypatch, ["v9"], predictions=("sentiment",))
    assert tools.cmd_wait_healthy(APP, "v9", timeout=0.2, interval=0.05) == 1


def test_report_posts_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "http_json", lambda method, url, payload=None, timeout=30: calls.append((url, payload)) or {})
    assert tools.cmd_report(APP, "v9", "REJECTED", 0.9, 0.8, 120, "Jenkins build #4") == 0
    assert calls == [(f"{APP}/api/models/pipeline-report", {
        "version": "v9", "status": "REJECTED", "old_f1": 0.9, "new_f1": 0.8, "training_samples": 120, "note": "Jenkins build #4"})]


def test_unreachable_app_returns_1(tmp_path, capsys):
    code = tools.main(["info", "--app-url", "http://127.0.0.1:9", "--out", str(tmp_path / "info.env")])
    assert code == 1
    assert "ERROR:" in capsys.readouterr().out
