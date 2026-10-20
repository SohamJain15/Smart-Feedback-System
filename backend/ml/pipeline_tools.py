"""Helper CLI called by Jenkins (stdlib only, works the same on Windows and Linux).

    python -m ml.pipeline_tools info --app-url URL --out FILE
    python -m ml.pipeline_tools export --app-url URL --out FILE
    python -m ml.pipeline_tools package --model-dir DIR --version vN --out-dir DIR
    python -m ml.pipeline_tools deploy --app-url URL --version vN --app-models-dir DIR --note TEXT [--model-dir DIR | --kind rollback]
    python -m ml.pipeline_tools wait-healthy --app-url URL --version vN [--timeout 60]
    python -m ml.pipeline_tools report --app-url URL --version vN --status REJECTED|FAILED ...

Every command prints "ERROR: <reason>" and exits 1 on failure.
"""
import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REQUIRED_FILES = ["sentiment.joblib", "topic.joblib", "priority.joblib", "metrics.json"]
INFO_KEYS = ["active_version", "aggregate_f1", "next_version", "newly_labelled", "threshold"]
SAMPLE = {"text": "The library wifi keeps dropping during lectures", "category": "Library", "rating": 2}


def _request(method: str, url: str, payload: dict | None, timeout: float) -> bytes:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"{method} {url} failed with HTTP {error.code}: {detail}")
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"cannot reach {url}: {getattr(error, 'reason', error)}")


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    return json.loads(_request(method, url, payload, timeout) or b"{}")


def http_bytes(url: str, timeout: float = 60) -> bytes:
    return _request("GET", url, None, timeout)


def cmd_info(app_url: str, out: Path) -> int:
    info = http_json("GET", f"{app_url}/api/models/pipeline-info")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(f"{key}={info[key]}\n" for key in INFO_KEYS))
    return 0


def cmd_export(app_url: str, out: Path) -> int:
    data = http_bytes(f"{app_url}/api/feedback/export")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    return 0


def cmd_package(model_dir: Path, version: str, out_dir: Path) -> Path:
    model_dir = Path(model_dir)
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"model folder {model_dir} is incomplete: missing {', '.join(missing)}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(out_dir) / f"model-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            archive.write(model_dir / name, f"{version}/{name}")
    return zip_path


def _copy_model(model_dir: Path | None, version: str, app_models_dir: Path) -> None:
    if model_dir is None or not Path(model_dir).is_dir():
        raise RuntimeError(f"model folder {model_dir} does not exist")
    app_models_dir.mkdir(parents=True, exist_ok=True)
    partial, final = app_models_dir / f"{version}.partial", app_models_dir / version
    shutil.rmtree(partial, ignore_errors=True)
    try:
        shutil.copytree(model_dir, partial)
    except OSError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)


def cmd_deploy(app_url: str, version: str, app_models_dir: Path, note: str,
               kind: str = "deploy", model_dir: Path | None = None) -> int:
    try:
        if kind == "deploy":
            _copy_model(model_dir, version, Path(app_models_dir))
        http_json("POST", f"{app_url}/api/models/pipeline-activate", {"version": version, "kind": kind, "note": note})
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"{'Rolled back to' if kind == 'rollback' else 'Activated'} {version}")
    return 0


def cmd_wait_healthy(app_url: str, version: str, timeout: float = 60, interval: float = 2.0) -> int:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if http_json("GET", f"{app_url}/api/health").get("active_version") == version:
                break
        except RuntimeError:
            pass  # the app may still be loading the model
        if time.monotonic() >= deadline:
            print(f"ERROR: app did not report active version {version} within {timeout:g}s")
            return 1
        time.sleep(interval)
    try:
        predictions = http_json("POST", f"{app_url}/api/feedback/analyze", SAMPLE).get("predictions", {})
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1
    if set(predictions) != {"sentiment", "topic", "priority"}:
        print(f"ERROR: smoke prediction returned {sorted(predictions)} instead of 3 classifiers")
        return 1
    print(f"Smoke test passed on {version}")
    return 0


def cmd_report(app_url: str, version: str, status: str, old_f1: float | None,
               new_f1: float | None, samples: int, note: str) -> int:
    try:
        http_json("POST", f"{app_url}/api/models/pipeline-report", {
            "version": version, "status": status, "old_f1": old_f1,
            "new_f1": new_f1, "training_samples": samples, "note": note,
        })
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jenkins helper for the retraining pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info"); info.add_argument("--app-url", required=True); info.add_argument("--out", type=Path, required=True)
    export = sub.add_parser("export"); export.add_argument("--app-url", required=True); export.add_argument("--out", type=Path, required=True)
    package = sub.add_parser("package")
    package.add_argument("--model-dir", type=Path, required=True); package.add_argument("--version", required=True); package.add_argument("--out-dir", type=Path, required=True)
    deploy = sub.add_parser("deploy")
    deploy.add_argument("--app-url", required=True); deploy.add_argument("--version", required=True)
    deploy.add_argument("--app-models-dir", type=Path, required=True); deploy.add_argument("--note", default="")
    deploy.add_argument("--kind", choices=["deploy", "rollback"], default="deploy"); deploy.add_argument("--model-dir", type=Path, default=None)
    wait = sub.add_parser("wait-healthy")
    wait.add_argument("--app-url", required=True); wait.add_argument("--version", required=True)
    wait.add_argument("--timeout", type=float, default=60); wait.add_argument("--interval", type=float, default=2.0)
    report = sub.add_parser("report")
    report.add_argument("--app-url", required=True); report.add_argument("--version", required=True)
    report.add_argument("--status", choices=["REJECTED", "FAILED"], required=True)
    report.add_argument("--old-f1", type=float, default=None); report.add_argument("--new-f1", type=float, default=None)
    report.add_argument("--samples", type=int, default=0); report.add_argument("--note", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "info":
            return cmd_info(args.app_url, args.out)
        if args.command == "export":
            return cmd_export(args.app_url, args.out)
        if args.command == "package":
            print(cmd_package(args.model_dir, args.version, args.out_dir))
            return 0
        if args.command == "deploy":
            return cmd_deploy(args.app_url, args.version, args.app_models_dir, args.note, args.kind, args.model_dir)
        if args.command == "wait-healthy":
            return cmd_wait_healthy(args.app_url, args.version, args.timeout, args.interval)
        return cmd_report(args.app_url, args.version, args.status, args.old_f1, args.new_f1, args.samples, args.note)
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
