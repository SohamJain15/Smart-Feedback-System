# Jenkins Auto-Retraining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Owner rule:** the executor only creates/edits repository files. The owner (Gyaneshwar) runs every terminal command, every `git` command, any environment-variable setup, and all Jenkins UI steps. Each task therefore ends with a **Run (owner)** block; do not run those yourself.

**Goal:** Jenkins retrains the model from reviewed feedback, gates it on F1, tests and archives it, activates it in the running app without a restart, smoke-tests it, and rolls back on failure.

**Architecture:** Training becomes a standalone CLI (`backend/ml`). New FastAPI router `app/pipeline_api.py` exposes info/export/activate/report endpoints; a stdlib-only `ml/pipeline_tools.py` is the single helper Jenkins calls. Two Jenkins Pipeline jobs: `feedback-ci` (`Jenkinsfile`) and `feedback-retrain` (`Jenkinsfile.retrain`). No Docker anywhere in the pipeline.

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy, scikit-learn/pandas/joblib, pytest, Jenkins declarative pipeline (Windows `bat`, Linux `sh`), React/Vite (one small UI change).

**Spec:** `docs/superpowers/specs/2026-10-06-auto-retraining-pipeline-design.md`

## Global Constraints

- No new third-party dependencies; new code uses stdlib plus what is in `backend/requirements.txt`. Python syntax must run on 3.10.
- Version strings match `^v\d+$`. Gate rule: promote only if candidate aggregate F1 is **strictly greater** than the live aggregate F1 (aggregate = mean of the 3 classifier F1s).
- `metrics.json` schema (exact): `{"version": str, "training_samples": int, "aggregate_f1": float, "metrics": {"sentiment"|"topic"|"priority": {"accuracy","precision","recall","f1_score"}}}`.
- Defaults: `MIN_NEW_LABELS=20`, `APP_URL=http://127.0.0.1:8000`, `APP_MODELS_DIR=D:\PS4\backend\models`. Jenkins job names: `feedback-ci`, `feedback-retrain`. Branch: `main`.
- Jenkinsfiles work on Windows and Linux (`isUnix()` helper), use a workspace `.venv`, and quote every path passed to `bat`/`sh` (workspace paths can contain spaces).
- No secrets in the repo. The Jenkins API token exists only in the shell that starts the backend.
- Existing API responses and the Dockerfiles/`docker-compose.yml` stay unchanged.

## Review Focus

1. Feedback text containing commas, quotes or newlines must survive the CSV export → train round trip (Task 3).
2. `version` values like `../x`, `v1; rm`, or `latest` must be rejected by activate (Task 4).
3. A corrupt or incomplete model folder must leave the current model active and the DB unchanged (Task 4).
4. Reviewed CSV that is header-only, has blank text, or lacks a column must give a clear error, not a traceback (Task 2).
5. App unreachable, or activation failing mid-deploy, must exit non-zero with a clear message and never leave a half-copied `vN` folder (Task 5).

---

### Task 1: Test isolation and CI Jenkinsfile

**Files:**
- Create: `backend/tests/conftest.py`, `backend/tests/test_isolation.py`
- Modify (replace content): `Jenkinsfile`

**Interfaces:** Produces: tests run against a temp DB and temp models dir; Jenkins job `feedback-ci`.

- [ ] **Step 1:** Write `test_isolation.py::test_database_and_models_use_temp_dirs`: asserts `app.database.DATABASE_PATH` and `app.ml_service.MODELS` are not inside the `backend/` directory and equal the env values `SMART_FEEDBACK_DB` / `SMART_FEEDBACK_MODELS`.
- [ ] **Step 2:** Create `conftest.py`: at import time (before any `app` import) `tempfile.mkdtemp(prefix="feedback-tests-")`, set `SMART_FEEDBACK_DB=<tmp>/test.db` and `SMART_FEEDBACK_MODELS=<tmp>/models` (overwrite, not `setdefault`), and insert the `backend/` directory at `sys.path[0]` so `ml` and `app` import regardless of cwd.
- [ ] **Step 3:** Replace `Jenkinsfile` with declarative pipeline, `options { timestamps() }`, `parameters { booleanParam(name:'RUN_E2E', defaultValue:false) }`, helpers `run(String cmd)` (`bat` on Windows, `sh` otherwise) and `venvPy()` (`.venv\Scripts\python` / `.venv/bin/python`, absolute via `env.WORKSPACE`). Stages: Checkout (`checkout scm`); Backend tests (`python -m venv .venv`, `pip install -r backend/requirements.txt`, in `backend`: `pytest -q --junitxml=test-results.xml`); Frontend build (`npm ci && npm run build` in `frontend`, then `archiveArtifacts 'frontend/dist/**'`); E2E (`when { expression { params.RUN_E2E } }`, runs `python e2e/run_local_e2e.py` using the venv python). `post { always { junit allowEmptyResults: true, testResults: 'backend/test-results.xml' } }`. No Docker commands.

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest -q          # Expected: all pass; backend\smart_feedback.db is NOT modified
cd D:\PS4; git add backend/tests Jenkinsfile; git commit -m "ci: isolate tests, Jenkins-only CI pipeline"; git push
```
Jenkins: New Item `feedback-ci` → Pipeline → Pipeline script from SCM → Git `https://github.com/SohamJain15/Smart-Feedback-System.git`, branch `*/main`, script path `Jenkinsfile` → Build Now. Expected: green, test results graph, `dist` artifact.

---

### Task 2: Training CLI and quality gate

**Files:**
- Create: `backend/ml/__init__.py`, `backend/ml/train.py`, `backend/ml/gate.py`, `backend/tests/test_train_gate.py`

**Interfaces:**
- Consumes: `app.ml_service.MLService().train(frame) -> (trained: dict, metrics: dict)`, `app.ml_service.base_dataset() -> DataFrame`, `TARGETS`.
- Produces:
  - `ml.train.train_candidate(reviewed_csv: Path | None, out_dir: Path, version: str) -> dict` — writes `<out_dir>/<version>/{sentiment,topic,priority}.joblib` and `metrics.json` (schema in Global Constraints) and returns that dict. Writes joblib directly (does not use `MLService.save`, which targets the global models dir). Raises `ValueError` with a readable message for missing columns / bad data.
  - `ml.train.main(argv: list[str] | None = None) -> int` — args `--reviewed-csv` (optional), `--out-dir`, `--version`; prints the error and returns 1 on `ValueError`/`FileNotFoundError`, else 0.
  - `ml.gate.decide(candidate_f1: float, current_f1: float) -> tuple[bool, str]` — `(True, reason)` only when strictly greater.
  - `ml.gate.main(argv=None) -> int` — args `--metrics <path> --current-f1 <float> [--out <path>]`; exit `0` promote, `10` reject, `1` error; reason printed and written to `--out`.

- [ ] **Step 1:** Write tests in `test_train_gate.py` (tmp_path fixtures):
  - `test_train_base_only`: 3 joblib files + `metrics.json` exist; `training_samples == len(base_dataset())`; `0 < aggregate_f1 <= 1`.
  - `test_train_with_reviewed_rows`: CSV of 3 rows copied from the first base-dataset row's labels → `training_samples == base + 3`.
  - `test_header_only_csv_equals_base_only`.
  - `test_missing_column_raises_value_error` (CSV without `priority`) and `test_blank_text_main_returns_1` (a row with empty `text`; `main` returns 1, no exception).
  - `test_gate_decide`: better → True; equal → False; worse → False.
  - `test_gate_main_exit_codes`: 0 / 10 for better / equal; 1 for a missing metrics file.
- [ ] **Step 2:** Implement `train.py` and `gate.py` per signatures. Reviewed rows are concatenated onto the base frame with `pd.concat(ignore_index=True)` after validating columns `text,sentiment,topic,priority`.

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest tests/test_train_gate.py -q     # Expected: all pass
cd D:\PS4; git add backend/ml backend/tests; git commit -m "feat: standalone training CLI and F1 quality gate"; git push
```

---

### Task 3: Pipeline info and export endpoints

**Files:**
- Create: `backend/app/pipeline_api.py`, `backend/tests/test_pipeline_api.py`
- Modify: `backend/app/main.py` (include router; reuse helpers)

**Interfaces:**
- Produces in `app/pipeline_api.py`: `router = APIRouter(prefix="/api")`; `THRESHOLD = 20`; `next_version(session: Session) -> str` (same rule as today: `v{max numeric version over all ModelVersion rows + 1}`); `newly_labelled(session: Session) -> int` (reviewed rows created after the last deployed `TrainingRun.completed_at`, same logic as today's `retraining_status`); `rows_to_csv(rows: list[Feedback]) -> str` (header `text,sentiment,topic,priority`, uses stdlib `csv` quoting, empty list → header only); `GET /api/models/pipeline-info` → `{active_version, aggregate_f1, next_version, newly_labelled, threshold, jenkins_managed}` (`jenkins_managed = bool(os.environ.get("JENKINS_URL"))`); `GET /api/feedback/export` → `text/csv` of reviewed rows (labels = the `actual_*` fields).
- Modifies `main.py`: `app.include_router(router)`; `retrain` uses `next_version`; `retraining_status` uses `newly_labelled` and `THRESHOLD`; responses unchanged.

- [ ] **Step 1:** Tests in `test_pipeline_api.py` (`TestClient(app)` as a context manager, like `test_flow.py`; assert relatively because the DB is shared across tests):
  - `test_pipeline_info_shape`: all six keys; `active_version == /api/health active_version`; `next_version` number is greater than every number in `/api/models/history`.
  - `test_existing_retraining_status_unchanged`: keys `newly_labelled, threshold, recommended, active_version`, `threshold == 20`.
  - `test_jenkins_managed_follows_env` (`monkeypatch.setenv/delenv("JENKINS_URL")`).
  - `test_export_only_reviewed`: save feedback → absent from CSV; label it → present, with header exactly `text,sentiment,topic,priority`.
  - `test_export_round_trips_awkward_text`: feedback text `'Wifi, "slow"\nagain'` is labelled then parsed back with `csv.reader` to identical text.
  - `test_rows_to_csv_empty_is_header_only`.
- [ ] **Step 2:** Implement per the Interfaces block; move the version/threshold logic out of `main.py` rather than copying it.

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest -q          # Expected: all pass (old tests included)
cd D:\PS4; git add backend/app backend/tests; git commit -m "feat: pipeline info and export endpoints"; git push
```

---

### Task 4: Activate and report endpoints

**Files:**
- Modify: `backend/app/pipeline_api.py`, `backend/app/schemas.py`, `backend/tests/test_pipeline_api.py`

**Interfaces:**
- Consumes: Task 2 `train_candidate`; Task 3 `router`, `next_version`; `app.ml_service.ml`, `MLService`, `MODELS`, `TARGETS`.
- Produces schemas in `schemas.py`: `ActivateRequest(version: str (pattern ^v\d+$), kind: Literal["deploy","rollback"] = "deploy", note: str = "")`; `ReportRequest(version: str (pattern ^v\d+$), status: Literal["REJECTED","FAILED"], old_f1: float | None = None, new_f1: float | None = None, training_samples: int = 0, note: str = "")`.
- Endpoints:
  - `POST /api/models/pipeline-activate` → `{active_version, aggregate_f1}`. Rules: folder `MODELS/<version>` missing → 404; any of the 3 joblib files or `metrics.json` missing → 422; load a fresh `MLService()` probe first and return 422 if it fails (corrupt file) **before** touching the DB or the live model; if already active → 200, no changes; otherwise set all other ACTIVE rows to `SUPERSEDED`, then for this version either set existing rows to `ACTIVE` (rollback) or insert 3 `ModelVersion` rows from `metrics.json` with status `ACTIVE`; then `ml.load(version)`. For `kind="deploy"` also insert `TrainingRun(version, status="COMPLETED", stage=note or "Activated by Jenkins", training_samples, old_f1=<previous active aggregate>, new_f1=<metrics aggregate>, deployed=True, completed_at=utcnow)`.
  - `POST /api/models/pipeline-report` → inserts `TrainingRun` (`deployed=False`, `completed_at=utcnow`, `stage=note`) and returns it.

- [ ] **Step 1:** Tests (versions `v901`+ to avoid collisions; build candidates with `train_candidate(None, MODELS, "v901")`):
  - `test_activate_switches_live_model`: after POST, `/api/health` shows `v901`, `/api/models/current` has version `v901`, exactly 3 ACTIVE rows in `/api/models/history`, a deployed run exists in `/api/models/training-runs`, and `/api/feedback/analyze` still returns 3 predictions.
  - `test_rollback_restores_previous`: capture the active version `p` first; activate `v902`; POST `kind="rollback"` with `p` → health shows `p`, 3 ACTIVE rows, no new TrainingRun created.
  - `test_activate_is_idempotent`.
  - `test_activate_rejects_bad_versions`: `../x`, `latest`, `v1;x` → 422.
  - `test_activate_missing_folder_404` (`v999`), `test_activate_incomplete_folder_422` (delete `topic.joblib`), `test_activate_corrupt_joblib_422_and_state_unchanged` (write garbage into `sentiment.joblib`; health and ACTIVE rows unchanged).
  - `test_report_records_rejected_and_failed` and `test_report_rejects_other_status_422`.
- [ ] **Step 2:** Implement per the Interfaces block. Use one DB transaction for the status changes and inserts; commit only after the probe load succeeded.

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest -q          # Expected: all pass
cd D:\PS4; git add backend; git commit -m "feat: model activate (deploy/rollback) and report endpoints"; git push
```

---

### Task 5: Pipeline helper CLI

**Files:**
- Create: `backend/ml/pipeline_tools.py`, `backend/tests/test_pipeline_tools.py`

**Interfaces:**
- Consumes: Task 3/4 endpoints (HTTP only), Task 2 `metrics.json` layout.
- Produces (stdlib only: `argparse, json, urllib, shutil, zipfile, pathlib, time`):
  - `http_json(method: str, url: str, payload: dict | None = None, timeout: float = 30) -> dict` (raises `RuntimeError` with the server's `detail` text on HTTP errors and a readable "cannot reach <url>" on connection errors).
  - `cmd_info(app_url: str, out: Path) -> int` — writes lines `active_version=…`, `aggregate_f1=…`, `next_version=…`, `newly_labelled=…`, `threshold=…` (one `key=value` per line, no quotes) to `out`.
  - `cmd_export(app_url: str, out: Path) -> int` — saves CSV bytes from `GET /api/feedback/export`.
  - `cmd_package(model_dir: Path, version: str, out_dir: Path) -> Path` — `out_dir/model-<version>.zip` containing `<version>/*` (3 joblib + `metrics.json`).
  - `cmd_deploy(app_url: str, version: str, app_models_dir: Path, note: str, kind: str = "deploy", model_dir: Path | None = None) -> int` — for `kind="deploy"`: copy `model_dir` to `<app_models_dir>/<version>.partial`, remove any existing `<version>` folder, rename to `<version>`; then `POST /api/models/pipeline-activate`. For `kind="rollback"`: no copy, only the POST.
  - `cmd_wait_healthy(app_url: str, version: str, timeout: float, interval: float = 2.0) -> int` — polls `/api/health` until `active_version == version`, then posts a fixed sample to `/api/feedback/analyze` and requires 3 prediction keys; `1` on timeout.
  - `cmd_report(app_url, version, status, old_f1, new_f1, samples, note) -> int`.
  - `main(argv: list[str] | None = None) -> int` — subcommands `info export package deploy wait-healthy report` with the same parameter names as flags (`--app-url`, `--out`, `--version`, …). Every command catches `RuntimeError`/`OSError`, prints `ERROR: <message>` and returns 1.

- [ ] **Step 1:** Tests (monkeypatch `pipeline_tools.http_json`; use `tmp_path`):
  - `test_info_writes_key_value_lines`.
  - `test_package_zip_contents` (names inside the zip).
  - `test_deploy_copies_then_activates` (captured call: `POST <app>/api/models/pipeline-activate`, payload `{"version","kind":"deploy","note"}`; folder `<version>` exists, no `.partial` left).
  - `test_deploy_replaces_stale_folder` (pre-existing `<version>` with an extra stale file → file gone after deploy).
  - `test_deploy_activation_failure_returns_1_without_partial` (http_json raises `RuntimeError`; no `.partial` folder remains).
  - `test_rollback_does_not_copy`.
  - `test_wait_healthy_success_and_timeout` (timeout 0.2, interval 0.05 → returns 1).
  - `test_unreachable_app_returns_1` (real call to `http://127.0.0.1:9` with a short timeout, no monkeypatch; output contains `ERROR:`).
- [ ] **Step 2:** Implement per the Interfaces block.

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest -q          # Expected: all pass
cd D:\PS4; git add backend; git commit -m "feat: pipeline helper CLI for Jenkins"; git push
```

---

### Task 6: `Jenkinsfile.retrain` and first end-to-end run

**Files:**
- Create: `Jenkinsfile.retrain`

**Interfaces:** Consumes Tasks 2 and 5 CLIs (run from `backend` with `-m ml.<module>`).

- [ ] **Step 1:** Write `Jenkinsfile.retrain` (declarative; same `run`/`venvPy` helpers as Task 1; `options { timestamps(); disableConcurrentBuilds() }`; `triggers { cron('H 2 * * *') }`; parameters `FORCE` (bool, false), `MIN_NEW_LABELS` (string, `20`), `APP_URL` (string, `http://127.0.0.1:8000`), `APP_MODELS_DIR` (string, `D:\PS4\backend\models`), `SIMULATE_FAILURE` (bool, false)). Stage contract (artifacts under `${WORKSPACE}\artifacts`; read `info.env` with `readFile` and plain Groovy `split`, no extra plugins):

| Stage | Action |
|---|---|
| Setup | create `.venv`, `pip install -r backend/requirements.txt` |
| Info | `pipeline_tools info --app-url … --out artifacts/info.env`; set `env.PREV`, `env.VERSION` (= `next_version`), `env.LIVE_F1`, `env.PROCEED='true'`; if `newly_labelled < MIN_NEW_LABELS` and not `FORCE` → `currentBuild.result='NOT_BUILT'`, `PROCEED='false'`, description explains why |
| Export | `pipeline_tools export … --out artifacts/reviewed.csv` |
| Train | `ml.train --reviewed-csv … --out-dir artifacts/models --version $VERSION` |
| Gate | `returnStatus` of `ml.gate --metrics artifacts/models/$VERSION/metrics.json --current-f1 $LIVE_F1 --out artifacts/gate.txt`; `10` → `NOT_BUILT`, `PROCEED='false'`, `pipeline_tools report --status REJECTED …`; `1` → fail the build |
| Test | `pytest -q --junitxml=test-results.xml` in `backend`, `junit` |
| Package | `pipeline_tools package` → `artifacts/model-$VERSION.zip`; `archiveArtifacts artifacts: 'artifacts/*.zip, artifacts/gate.txt, artifacts/models/*/metrics.json', fingerprint: true`; set `currentBuild.description = "$PREV -> $VERSION  F1 $LIVE_F1 -> <new>"` |
| Deploy | `pipeline_tools deploy --version $VERSION --model-dir artifacts/models/$VERSION --app-models-dir "$APP_MODELS_DIR" --note "Jenkins build #${BUILD_NUMBER}"`; then `env.DEPLOYED='true'` |
| Smoke | `pipeline_tools wait-healthy --version <$VERSION, or v0 when SIMULATE_FAILURE> --timeout 60` |

Every stage after Info carries `when { expression { env.PROCEED == 'true' } }`. `post { failure }`: if `env.DEPLOYED == 'true'` run `pipeline_tools deploy --kind rollback --version $PREV …`; then `pipeline_tools report --status FAILED --version $VERSION …` wrapped so a failing report only marks the build `UNSTABLE`. `post { success }`: nothing extra (the activate call already recorded the run). `reviewed.csv` is never archived.

**Run (owner):**
```powershell
cd D:\PS4; git add Jenkinsfile.retrain; git commit -m "ci: Jenkins retrain/deploy pipeline"; git push
# start the app (separate terminals)
cd D:\PS4\backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Jenkins: New Item `feedback-retrain` → Pipeline from SCM, script path `Jenkinsfile.retrain`. Label ≥ 1 feedback in the app, then **Build with Parameters** with `FORCE` ticked. Expected: stages green, description `vN -> vN+1`, `model-vN+1.zip` under Artifacts, `/models` page shows `vN+1` active with a history row. Run again with `FORCE` ticked and nothing new: either a new version (if F1 improved) or `NOT_BUILT` with a REJECTED row. Then run once with `SIMULATE_FAILURE` ticked: Smoke fails, build is red, `/models` shows the previous version active again and a FAILED run.

---

### Task 7: Retrain button triggers Jenkins

**Files:**
- Modify: `backend/app/pipeline_api.py`, `backend/app/main.py`, `frontend/src/pages/ModelsPage.tsx`, `backend/tests/test_pipeline_api.py`

**Interfaces:**
- Produces in `pipeline_api.py`: `jenkins_configured() -> bool` (all of `JENKINS_URL`, `JENKINS_USER`, `JENKINS_TOKEN`, `JENKINS_JOB` set and non-empty); `trigger_jenkins() -> dict` — POST to `<JENKINS_URL stripped of trailing "/">/job/<quote(JENKINS_JOB)>/buildWithParameters?FORCE=true` with HTTP Basic auth `JENKINS_USER:JENKINS_TOKEN` via `urllib.request.urlopen`; returns `{"status": "REQUESTED", "job": <JENKINS_JOB>}`; any URL/HTTP error → `HTTPException(502, "Could not reach Jenkins: <reason>")`.
- Modifies `main.retrain`: first line `if jenkins_configured(): return trigger_jenkins()`; otherwise the existing in-process behaviour, untouched.

- [ ] **Step 1:** Tests (monkeypatch `urllib.request.urlopen` in `app.pipeline_api` and set the four env vars): `test_retrain_triggers_jenkins` (response `{"status":"REQUESTED",...}`; captured request URL ends with `/job/<job>/buildWithParameters?FORCE=true` and carries an `Authorization: Basic …` header; no new row in `/api/models/training-runs`); `test_retrain_jenkins_error_returns_502`; `test_jenkins_configured_false_when_any_var_missing`.
- [ ] **Step 2:** Implement. In `ModelsPage.tsx` `start()`: if the response has `status === 'REQUESTED'`, show a notice "Retraining requested in Jenkins (job <job>). Refresh this page when the build finishes." and return without calling `setRunning`/`setRuns`; keep the existing path for normal runs. Fetch `/models/pipeline-info` in `load()` and, when `jenkins_managed` is true, label the button "Request retraining" (otherwise keep "Start retraining").

**Run (owner):**
```powershell
cd D:\PS4\backend; python -m pytest -q
cd D:\PS4\frontend; npm run build               # Expected: no TypeScript errors
# enable the button (same shell that starts the backend); create the token in Jenkins: your user > Security > API Token
$env:JENKINS_URL="http://localhost:8080"; $env:JENKINS_USER="<jenkins-user>"; $env:JENKINS_TOKEN="<api-token>"; $env:JENKINS_JOB="feedback-retrain"
cd D:\PS4\backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
cd D:\PS4; git add backend frontend; git commit -m "feat: retrain button triggers Jenkins job"; git push
```
Expected: clicking the button on `/models` shows the notice and a `feedback-retrain` build starts.

---

### Task 8: Optional local Selenium stage

**Files:**
- Create: `e2e/run_local_e2e.py`
- Modify: `e2e/test_feedback_flow.py` (`create_driver`), `frontend/vite.config.ts` (proxy target), `Jenkinsfile` (add `-r e2e/requirements.txt` to the E2E stage's pip install, inside that stage only)

**Interfaces:**
- `vite.config.ts`: proxy target `process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'`.
- `create_driver()`: when `SELENIUM_URL` is unset/empty (change its default from the docker URL to `""`) use `webdriver.Chrome(options=options)` (same headless options); otherwise the existing `Remote` path, so the compose flow keeps working.
- `e2e/run_local_e2e.py::main() -> int`: creates temp DB/models dirs, starts `uvicorn app.main:app --port 8010` (cwd `backend`, env `SMART_FEEDBACK_DB`, `SMART_FEEDBACK_MODELS`), starts `npx vite preview --port 4173` (cwd `frontend`, env `VITE_API_TARGET=http://127.0.0.1:8010`), waits (≤60 s each) for `/api/health` and `http://127.0.0.1:4173`, runs `pytest -q e2e/test_feedback_flow.py` with `APP_URL=http://127.0.0.1:4173`, always terminates both child processes, returns pytest's exit code.

- [ ] **Step 1:** Make the three changes above (no unit tests; the verification is a Jenkins run).

**Run (owner):**
```powershell
cd D:\PS4; git add e2e frontend/vite.config.ts; git commit -m "test: optional local Selenium run"; git push
```
Jenkins: build `feedback-ci` with `RUN_E2E` ticked (needs Chrome installed on the Jenkins machine). Expected: E2E stage green. If it fails because of Chrome/driver, leave `RUN_E2E` off; the rest of the pipeline is unaffected.

---

### Task 9: Documentation and demo script

**Files:**
- Create: `docs/DEVOPS.md`
- Modify: `README.md` (replace the "Jenkins CI/CD pipeline" section with a short summary and a link; keep the Docker run instructions)

- [ ] **Step 1:** `docs/DEVOPS.md` contains: the architecture diagram from the spec; Jenkins setup (two jobs, parameters, the PATH note, `APP_MODELS_DIR`, optional env vars for the button); a 6-step demo script (label ≥20 → click button → watch stages → show new version in `/models` → show `NOT_BUILT` rejection → show `SIMULATE_FAILURE` rollback); screenshot checklist (stage view, artifacts page with the zip, `/models` before/after, console showing rollback); syllabus mapping table (practicals 1–10, noting Docker practicals 4/5 are covered by the existing Dockerfiles but not executed); the F1-split limitation.
- [ ] **Step 2:** README section: three lines on the two jobs plus a link to `docs/DEVOPS.md`.

**Run (owner):**
```powershell
cd D:\PS4; git add docs README.md; git commit -m "docs: DevOps pipeline guide and demo script"; git push
```
Final acceptance: follow `docs/DEVOPS.md` once from a fresh shell and capture the screenshots.

---

## Self-review notes

- Spec coverage: components 4.1–4.6 → Tasks 2–7; CI job → Task 1; optional Selenium → Task 8; error-handling table → Tasks 4–6 tests and the post block; testing section → conftest (Task 1) plus per-task tests; docs and syllabus mapping → Task 9. Spec section 12's separate "verify tools" step is already done (env-check passed).
- Names used across tasks: `train_candidate`, `decide`, `next_version`, `newly_labelled`, `rows_to_csv`, `ActivateRequest`, `ReportRequest`, `cmd_*`, `jenkins_configured`, `trigger_jenkins` are each defined once and referenced consistently.
