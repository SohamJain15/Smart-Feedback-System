# Auto-Retraining Pipeline with Jenkins — Design (Jenkins-only, no Docker)

Date: 2026-10-06 (revised: Docker removed from the pipeline)
Project: Smart Feedback System with Auto-Retraining Pipeline (DevOps mini project, practical 10, topic 9)

## 1. Goal and success criteria

The existing app (React + FastAPI + SQLite + scikit-learn) runs locally. Add DevOps so that **Jenkins retrains the model, tests it, versions it as a build artifact, activates it in the running app, verifies it, and rolls back automatically on failure**. No Docker is needed to run any of it.

Demo:

1. Admin labels >= 20 feedback records (or clicks "Start retraining").
2. Jenkins job `feedback-retrain` starts: exports labels, trains a candidate, compares F1 with the live model.
3. If better: tests pass, model bundle `model-vN.zip` is archived in Jenkins, copied to the app's models folder, activated without restarting the app, smoke-tested.
4. `/models` shows the new active version and the run in history.
5. If worse: nothing changes; build is "not built" with the reason.
6. If activation or the smoke test fails: the previous version is re-activated automatically.

A second job, `feedback-ci`, runs on every push: backend tests, frontend build, optional Selenium test.

## 2. Context (what exists)

- `backend/app/ml_service.py`: `MLService.train()` (3 TF-IDF + LogisticRegression classifiers) returning metrics.
- `backend/app/main.py`: `run_training()` merges `data/training_data.csv` with reviewed feedback, promotes only if aggregate F1 is strictly higher, saves `models/vN/*.joblib`, writes `ModelVersion`/`TrainingRun` rows; `POST /api/models/retrain` runs it in-process.
- App runs locally: backend `uvicorn` on 127.0.0.1:8000, frontend Vite on 5173 (README).
- `Jenkinsfile` currently depends on Docker Compose and Selenium containers (to be replaced).
- Dockerfiles and `docker-compose.yml` already exist in the repo; they stay as optional deliverables but nothing in the pipeline runs them.
- Repo: https://github.com/SohamJain15/Smart-Feedback-System (branch `main`). Jenkins is on the same Windows machine as the app.

Problems to solve: the labelled data lives in the app's SQLite DB, which Jenkins' workspace cannot see; Jenkins' workspace is a different folder from the app's models folder; two retraining paths (in-process and Jenkins) would race on version numbers.

## 3. Architecture

```
 Running app (local)                 Jenkins (same machine)
 uvicorn :8000                       Job: feedback-retrain
 models/ folder  ◄── copy ────────── 1 Checkout (GitHub)
 SQLite DB                           2 Setup venv + deps
   ▲  GET pipeline-info ───────────► 3 Pipeline info, skip if too few labels
   ▲  GET feedback/export ─────────► 4 Export labelled CSV
                                     5 Train candidate  (ml/train.py)
                                     6 Quality gate     (ml/gate.py)
                                     7 pytest + model check
                                     8 Package model-vN.zip (archived, fingerprinted)
   ▲  POST pipeline-activate ◄─────── 9 Deploy: copy to app models dir + activate
   ▲  GET health + analyze ◄────────── 10 Smoke test
   ▲  POST pipeline-report ◄────────── 11 Report; on failure roll back to previous
                                     Job: feedback-ci  (every push)
```

## 4. Components

### 4.1 `backend/ml/train.py` — standalone training CLI
`--reviewed-csv`, `--out-dir`, `--version`. Reuses `MLService.train()` and `base_dataset()`. Writes `<out-dir>/<version>/{sentiment,topic,priority}.joblib` and `metrics.json` (per-target metrics, aggregate F1, sample count). No network, no DB.

### 4.2 `backend/ml/gate.py`
Candidate `metrics.json` vs current aggregate F1. Exit `0` promote (strictly greater, same rule as today), `10` reject, `1` error. Prints a one-line reason.

### 4.3 `backend/ml/pipeline_tools.py`
Cross-platform helper over `urllib`/`shutil`/`zipfile` (no curl, jq, xcopy): `info`, `export`, `package`, `deploy`, `wait-healthy`, `report`, `rollback`. Keeps the Jenkinsfile identical on Windows and Linux.

### 4.4 New backend endpoints (additive, no schema change)
- `GET /api/models/pipeline-info` -> `{active_version, aggregate_f1, next_version, newly_labelled, threshold, jenkins_managed}`.
- `GET /api/feedback/export` -> CSV of reviewed rows (`text,sentiment,topic,priority`).
- `POST /api/models/pipeline-activate` `{version, note}`: verifies `models/<version>/` has all three joblib files and `metrics.json`, registers `ModelVersion` rows from it as ACTIVE, marks previous ACTIVE rows `SUPERSEDED`, hot-loads the model, records a `TrainingRun` (deployed). Also used for rollback (re-activating an older version marks the newer one `SUPERSEDED`).
- `POST /api/models/pipeline-report` `{version, status, old_f1, new_f1, training_samples, note}`: records rejected/failed runs. Jenkins build URL goes into the existing `stage` text.

### 4.5 Retrain button -> Jenkins (optional, last increment)
If `JENKINS_URL`, `JENKINS_USER`, `JENKINS_TOKEN`, `JENKINS_JOB` are set in the shell that starts the backend, `POST /api/models/retrain` triggers `buildWithParameters` (`FORCE=true`) and the in-process trainer returns 409 "managed by Jenkins" (removes the version race). Unset: today's behaviour is unchanged. UI wording changes to "Retraining requested in Jenkins".

### 4.6 Jenkins jobs
Both are Pipeline jobs ("Pipeline script from SCM", branch `main`).

**`Jenkinsfile` (CI, replaces the Docker version):** checkout; create `.venv`, install requirements, `pytest --junitxml` published with `junit`; `npm ci && npm run build` in `frontend`; archive `frontend/dist`; optional stage `RUN_E2E` (default off) runs the existing Selenium test against locally started servers using the installed Chrome.

**`Jenkinsfile.retrain`:** parameters `FORCE` (bool), `MIN_NEW_LABELS` (20), `APP_URL` (`http://127.0.0.1:8000`), `APP_MODELS_DIR` (the models folder the running app reads, e.g. `D:\PS4\backend\models`). Options: `disableConcurrentBuilds()`, timestamps. Triggers: nightly `cron`, manual "Build with Parameters", app button.

| # | Stage | Behaviour |
|---|-------|-----------|
| 1-2 | Checkout, venv | `.venv` in workspace, `pip install -r backend/requirements.txt` |
| 3 | Pipeline info | skip (`NOT_BUILT`) if `newly_labelled < MIN_NEW_LABELS` and not `FORCE` |
| 4 | Export | `artifacts/reviewed.csv` |
| 5 | Train | `artifacts/<next_version>/` |
| 6 | Gate | reject -> `NOT_BUILT` + report REJECTED; later stages skipped by `when` |
| 7 | Tests | backend pytest + test that loads candidate and predicts |
| 8 | Package | `model-<vN>.zip` + `metrics.json`, `archiveArtifacts`, `fingerprint: true` |
| 9 | Deploy | copy `<vN>` into `APP_MODELS_DIR`, `POST pipeline-activate` |
| 10 | Smoke test | `/api/health` shows `<vN>` and `/api/feedback/analyze` returns 3 predictions |
| 11 | Report | build description shows old F1 -> new F1 and version |

`post { failure }`: re-activate the previous version (still on disk) via `pipeline-activate`, report FAILED. `post { always }`: archive `metrics.json` and `gate.txt` (not `reviewed.csv`, it contains user text).

## 5. Data flow (one retrain)
reviewed rows (app DB) -> export CSV -> `train.py` (+ base CSV) -> candidate + metrics -> gate vs live aggregate F1 -> pytest -> zip artifact in Jenkins -> copied into app models folder -> activate API hot-loads it -> smoke test -> report. The app stays up; feedback data is untouched.

## 6. Error handling

| Situation | Result |
|-----------|--------|
| Too few new labels | `NOT_BUILT`, reason shown, nothing changed |
| Candidate not better | `NOT_BUILT`, run recorded as REJECTED in app history |
| App unreachable | FAILED at stage 3/4, nothing changed |
| Copy/activate fails or smoke test fails | previous version re-activated, build FAILED, run recorded FAILED |
| Report call fails after good deploy | build UNSTABLE (model is live; only history missing) |
| Concurrent builds | prevented by `disableConcurrentBuilds()` |
| Secrets | Jenkins API token only in the user's shell/Jenkins credentials; never in the repo |

## 7. Testing
- Unit (pytest, no network): train CLI on a small fixture, gate exit codes, export only reviewed rows, next-version logic, activate (valid, missing files, rollback).
- `backend/tests/conftest.py` sets `SMART_FEEDBACK_DB`/`SMART_FEEDBACK_MODELS` to a temp dir before the app imports (today's tests write into `backend/smart_feedback.db`; CI would pollute the workspace).
- Pipeline smoke stage is the integration test; manual acceptance = the demo in section 1 including one forced failure to show rollback.

## 8. Files
New: `backend/ml/{__init__,train,gate,pipeline_tools}.py`, `backend/tests/{conftest,test_train_gate,test_pipeline_api}.py`, `Jenkinsfile.retrain`, `docs/DEVOPS.md`.
Changed: `backend/app/main.py` (4 endpoints, optional Jenkins trigger), `Jenkinsfile` (Docker removed), `frontend/src/pages/ModelsPage.tsx` (wording), `README.md`.
Untouched: Dockerfiles, `docker-compose.yml`, `e2e/` (kept for reference/optional use).

## 9. Syllabus coverage
| Practical | Evidence |
|-----------|----------|
| 1 Git/GitHub | repo + branch/PR workflow |
| 2 Jenkins setup | two jobs, credentials, plugin checklist in `docs/DEVOPS.md` |
| 3 CI/CD with Jenkins | `Jenkinsfile`, `Jenkinsfile.retrain` |
| 4, 5 Docker | Dockerfiles/compose exist in repo but are not run (explicitly out of scope on this PC) |
| 6, 7 Selenium | optional `RUN_E2E` stage with local Chrome |
| 8 ML in CI/CD | train, gate, test, package, activate |
| 9 Deploy models through Jenkins | artifact + activation + rollback (without Docker) |
| 10 Mini project | whole system |

## 10. Prerequisites (verify first)
Jenkins can run `python`, `git`, `node`/`npm` from a pipeline; the Jenkins service account can read the repo checkout and write to `APP_MODELS_DIR`; the app is running when `feedback-retrain` executes. The user runs all git commands, shell environment variables, and Jenkins UI setup manually.

## 11. Out of scope
Docker/Docker Hub in the pipeline, Kubernetes, cloud hosting, MLflow, monitoring stacks, auth on the new endpoints (local prototype), model algorithm changes.
Known limitation for the report: candidate and live F1 use different validation splits as data grows, so the gate is a practical heuristic.

## 12. Build order
1. Verify Jenkins tools (throw-away job, no Docker).
2. `conftest.py` test isolation + new CI `Jenkinsfile`.
3. `ml/train.py` + `ml/gate.py` + tests.
4. Backend endpoints + tests.
5. `pipeline_tools.py` + `Jenkinsfile.retrain`; first forced end-to-end run.
6. Rollback demonstration.
7. App button -> Jenkins (optional).
8. `docs/DEVOPS.md`, README, screenshots.
