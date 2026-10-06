# Auto-Retraining DevOps Pipeline — Design

Date: 2026-10-06
Project: Smart Feedback System with Auto-Retraining Pipeline (DevOps mini project, practical 10, topic 9)

## 1. Goal and success criteria

Turn the existing working app (React + FastAPI + SQLite + scikit-learn) into a DevOps-driven system where **Jenkins retrains the model, packages it into a Docker image, publishes the image to Docker Hub and redeploys the app**, with an automatic quality gate and rollback.

Success looks like (this is the demo):

1. Admin labels >= 20 feedback records in the app (or clicks "Start retraining").
2. A Jenkins job runs by itself: exports labelled data, trains a candidate, compares F1 with the live model.
3. If better, a backend image containing the new model is built, pushed to Docker Hub as `vN` and `latest`, and the running stack is updated.
4. The `/models` page shows the new version as active; Docker Hub shows the new tag.
5. If the candidate is worse, nothing is deployed and the build is marked "not built" with the reason.
6. If the deployed container fails its health check, the previous tag is redeployed automatically.

Every syllabus practical (1-9) is visibly exercised (see section 9).

## 2. Context (what exists today)

- `backend/app/ml_service.py`: `MLService.train()` trains 3 TF-IDF + LogisticRegression classifiers (sentiment, topic, priority) and returns per-target metrics.
- `backend/app/main.py`: `run_training()` merges `data/training_data.csv` with admin-reviewed rows, trains, promotes only if aggregate F1 is strictly higher, saves `models/vN/*.joblib`, writes `ModelVersion`/`TrainingRun` rows. Triggered by `POST /api/models/retrain` (runs in-process).
- `Jenkinsfile`: pytest -> frontend build -> `docker compose up` -> Selenium smoke test. Its `post` step runs `down --volumes`.
- `docker-compose.yml`: backend, frontend, optional selenium/e2e; runtime data in volume `feedback_runtime`.
- Jenkins and Docker Desktop run on the same Windows machine as the app.

Problems this design must solve:

- Reviewed feedback lives in SQLite inside a Docker volume, so Jenkins cannot read it directly.
- `down --volumes` in the CI pipeline would delete live data if CI and the real deployment share a compose project.
- Models are generated at runtime and excluded from Docker images (`.dockerignore`), so no image carries a model.
- Two retraining paths (in-process and Jenkins) would race on version numbers.

## 3. Architecture

```
 Admin labels feedback            Jenkins (same machine)                     Docker Hub
 ┌────────────────────┐   POST    ┌───────────────────────────────┐  push   ┌──────────────┐
 │ Prod stack         │ trigger   │ Job: feedback-retrain          │ ──────► │ backend:vN   │
 │ (compose project   │ ────────► │ 1 Checkout                     │         │ backend:latest│
 │  feedback-prod)    │           │ 2 Pipeline info  (GET)         │         │ frontend:..  │
 │  frontend :8080    │ ◄──────── │ 3 Export labels  (GET csv)     │         └──────┬───────┘
 │  backend  :8001    │  export   │ 4 Train candidate (ml/train.py)│                │ pull
 │  volume: runtime   │           │ 5 Quality gate (F1 compare)    │                ▼
 └────────▲───────────┘           │ 6 pytest + model check         │         ┌──────────────┐
          │ compose up -d         │ 7 Build image + bundle model   │         │ docker       │
          └───────────────────────│ 8 Push 9 Deploy 10 Smoke test  │ ──────► │ compose pull │
                                  │ 11 Report back / rollback      │         │ + up -d      │
 Job: feedback-ci (existing)      └───────────────────────────────┘         └──────────────┘
 project feedback-ci, ports 18080/18001, own throw-away volume
```

Two Jenkins jobs, two compose projects, so CI can never touch production data.

## 4. Components

### 4.1 Standalone training CLI — `backend/ml/train.py`
- Inputs: `--reviewed-csv <path>` (optional), `--out-dir <dir>`, `--version vN`.
- Reuses `MLService.train()` and `base_dataset()` unchanged (single source of training logic).
- Outputs: `<out-dir>/<version>/{sentiment,topic,priority}.joblib` and `<out-dir>/<version>/metrics.json` (per-target metrics, aggregate F1, training sample count).
- Pure Python, no network, no DB: easy to unit-test and cross-platform.

### 4.2 Quality gate — `backend/ml/gate.py`
- Inputs: candidate `metrics.json`, current aggregate F1.
- Exit code `0` = promote (candidate aggregate F1 strictly greater, same rule as today), `10` = reject, `1` = error. Prints a one-line reason that Jenkins shows in the build description.

### 4.3 Pipeline helper — `backend/ml/pipeline_tools.py`
Small cross-platform CLI over Python `urllib` (no curl/jq dependency on Windows): `info`, `export`, `wait-healthy`, `report`. Keeps `Jenkinsfile.retrain` free of `sh`/`bat` duplication.

### 4.4 New backend endpoints (additive only)
- `GET /api/models/pipeline-info` -> `{active_version, aggregate_f1, next_version, newly_labelled, threshold, jenkins_managed}`. `next_version` uses the same "max over all versions + 1" rule as today.
- `GET /api/feedback/export` -> CSV of reviewed rows, columns `text,sentiment,topic,priority`.
- `POST /api/models/pipeline-report` -> records a `TrainingRun` row from Jenkins: `{version, status, old_f1, new_f1, training_samples, note}`. The build URL goes into the existing `stage` text, so **no schema change** and no DB migration.

### 4.5 Bundled model sync on startup
- Image build copies the chosen model to `/app/bundled_model/<version>/` (+ `metrics.json`). A `backend/bundled_model/.gitkeep` keeps plain builds working; gitignored otherwise.
- At startup, if a bundled model exists and differs from the active DB version, the **image's bundled model wins**: copy into the runtime models dir, insert `ModelVersion` rows from `metrics.json` as ACTIVE, set other ACTIVE rows to `SUPERSEDED`, load it.
- This one rule covers upgrade and rollback (rolling back to an older image reactivates its model; the newer one is marked `SUPERSEDED`). With no bundled model, startup behaves exactly as today (trains v1 from the CSV).

### 4.6 Retrain button becomes a Jenkins trigger
- When `JENKINS_URL`, `JENKINS_USER`, `JENKINS_TOKEN`, `JENKINS_JOB` are set, `POST /api/models/retrain` calls Jenkins `buildWithParameters` (API-token auth) with `FORCE=true`, and the in-process trainer is **disabled** (HTTP 409 "managed by Jenkins" if called another way). This removes the version-number race.
- When those variables are not set, today's in-process behaviour is unchanged, so the app still works standalone for local development.
- UI change is minimal: button text and notice say "Retraining requested in Jenkins"; history table already shows runs reported by the pipeline.

### 4.7 Jenkins pipelines
**`Jenkinsfile` (CI, existing, hardened):** `COMPOSE_PROJECT_NAME=feedback-ci`, host ports from `FRONTEND_PORT`/`BACKEND_PORT` (default 8080/8001, CI uses 18080/18001), `down --volumes` limited to this project. On `main`, after green tests, pushes `frontend:latest` and `backend:baseline` (no bundled model) to Docker Hub.

**`Jenkinsfile.retrain` (new), parameters** `FORCE` (bool), `MIN_NEW_LABELS` (default 20), `APP_URL` (default `http://localhost:8001`), `DOCKERHUB_USER`:

| # | Stage | Behaviour |
|---|-------|-----------|
| 1 | Checkout | `checkout scm` |
| 2 | Pipeline info | `pipeline_tools info`; skip run (result `NOT_BUILT`) if `newly_labelled < MIN_NEW_LABELS` and not `FORCE` |
| 3 | Export labels | save `artifacts/reviewed.csv` |
| 4 | Train candidate | `ml/train.py` -> `artifacts/<next_version>/` |
| 5 | Quality gate | `ml/gate.py`; reject -> `NOT_BUILT`, remaining stages skipped via `when` |
| 6 | Tests | backend pytest + a test that loads the candidate and predicts |
| 7 | Build image | copy candidate into `backend/bundled_model/`, `docker build -t <user>/smart-feedback-backend:<vN> -t ...:latest backend` |
| 8 | Push | `withCredentials` Docker Hub token (credential id `dockerhub-creds`), `docker login --password-stdin`, push both tags |
| 9 | Deploy | `docker compose -p feedback-prod -f docker-compose.prod.yml pull && up -d` with `BACKEND_TAG=<vN>` |
| 10 | Smoke test | `wait-healthy`: `/api/health` reports the new version within 90 s and `/api/feedback/analyze` returns 3 predictions |
| 11 | Report | `pipeline-report` records run (COMPLETED / REJECTED / FAILED) so the UI history is correct |

`post { failure }`: redeploy `BACKEND_TAG=<previous>` (previous = `baseline` when active was `v1`), then report FAILED. `post { always }`: archive `metrics.json` and `gate.txt`; `reviewed.csv` is not archived (contains user text).

Triggers: nightly `cron`, "Build with Parameters" (manual), and the app button via API.

### 4.8 Production compose — `docker-compose.prod.yml`
`image: ${DOCKERHUB_USER}/smart-feedback-backend:${BACKEND_TAG:-baseline}`, frontend from Docker Hub, ports 8080/8001, its own named volume. Run with `-p feedback-prod`. The existing `docker-compose.yml` stays the dev/CI file (build from source).

## 5. Data flow (one retrain)

labelled rows (SQLite, prod volume) -> `GET /api/feedback/export` -> `reviewed.csv` -> `train.py` (+ base CSV) -> candidate + `metrics.json` -> gate (vs `aggregate_f1` from `pipeline-info`) -> image `vN` (model baked in) -> Docker Hub -> `compose pull/up` -> startup sync registers `vN` ACTIVE -> smoke test -> `pipeline-report` -> UI shows `vN`. Feedback data survives because only the backend container is replaced; the volume is kept.

## 6. Error handling

| Situation | Result |
|-----------|--------|
| Not enough new labels | `NOT_BUILT`, reason in build description, nothing deployed |
| Candidate not better | `NOT_BUILT`, reported as REJECTED in the app history |
| App unreachable at export | build FAILED at stage 2/3, nothing changed |
| Docker Hub push fails | FAILED before deploy; prod untouched |
| New container unhealthy / wrong version | automatic redeploy of previous tag, build FAILED, run reported FAILED |
| Report call fails after good deploy | build UNSTABLE (deployment is fine; history only is missing) |
| Concurrent builds | Jenkins `disableConcurrentBuilds()` on the retrain job |
| Secrets | Docker Hub token and Jenkins API token live only in Jenkins credentials / the user's shell; never in the repo |

## 7. Testing

- Unit (pytest, no Docker): `train.py` on a small fixture CSV produces files + metrics; `gate.py` exit codes for better/equal/worse; export endpoint returns only reviewed rows; `pipeline-info` next-version logic; bundled-model sync for upgrade, rollback and no-bundle cases.
- `backend/tests/conftest.py` points `SMART_FEEDBACK_DB` / `SMART_FEEDBACK_MODELS` at a temp dir **before** the app imports, so tests stop writing to `backend/smart_feedback.db` (an existing weakness that would also pollute CI workspaces).
- Integration: existing Selenium e2e keeps running in the CI job; the retrain job's smoke stage is the integration test of the deployed model.
- Manual acceptance: the demo in section 1, including one forced failure (bad tag) to show rollback.

## 8. Files

New: `backend/ml/{__init__,train,gate,pipeline_tools}.py`, `backend/bundled_model/.gitkeep`, `backend/tests/{conftest,test_train_gate,test_pipeline_api,test_model_sync}.py`, `docker-compose.prod.yml`, `Jenkinsfile.retrain`, `docs/DEVOPS.md` (diagram, setup steps, screenshots checklist).
Changed: `backend/app/main.py` (3 endpoints, startup sync, Jenkins trigger mode), `backend/Dockerfile` (copy `ml/` and `bundled_model/`), `backend/.dockerignore`, `.gitignore`, `docker-compose.yml` (project-safe ports), `Jenkinsfile` (isolation + push), `frontend/src/pages/ModelsPage.tsx` (button wording), `README.md`.

## 9. Syllabus coverage

| Practical | Evidence in this project |
|-----------|--------------------------|
| 1 Git/GitHub | repo with feature branches and pull requests for the work above |
| 2 Jenkins setup | two jobs, credentials, plugins checklist in `docs/DEVOPS.md` |
| 3 CI/CD with Jenkins | `Jenkinsfile` (CI) and `Jenkinsfile.retrain` (CD) |
| 4 Docker commands | build, tag, push, pull, compose, volumes in the pipeline |
| 5 Containerised app | backend, frontend, e2e images |
| 6 Selenium install | Selenium container in the CI job |
| 7 JS + Selenium test | e2e smoke test (Python Selenium today; optional JS variant is out of scope) |
| 8 ML model in CI/CD | train, gate, test, package in `Jenkinsfile.retrain` |
| 9 Deploy models via Docker through Jenkins | Docker Hub push + compose deploy + rollback |
| 10 Mini project (topic 9) | the whole system |

## 10. Assumptions and prerequisites (to verify first)

- Jenkins runs on this Windows machine and its service account can run `docker` (Docker Desktop must be running). First plan task: a throw-away Jenkins job that runs `docker version` and `python --version`.
- Python 3.10+ and Git are on Jenkins' PATH; Jenkins has Pipeline and Credentials Binding plugins (defaults).
- The user has a Docker Hub account and creates an access token (stored in Jenkins as `dockerhub-creds`) and a Jenkins API token (for the optional button trigger). The user sets any environment variables and runs all git commands and compose/Docker commands manually; the implementation only edits repository files.
- Repository is on GitHub (Jenkins uses "Pipeline script from SCM").

## 11. Out of scope (YAGNI)

Kubernetes, cloud hosting, a model registry service (MLflow), Prometheus/Grafana, authentication on the new endpoints (local prototype, same as the rest of the API), a JavaScript Selenium variant, releasing backend code changes without retraining, and any change to the model algorithm.

Known limitation to state in the report: candidate and live F1 are measured on different validation splits as the dataset grows, so the gate is a practical heuristic, not a statistical guarantee.

## 12. Build order (increments, each independently demonstrable)

1. Verify Jenkins can run docker/python (throw-away job).
2. Isolate CI from prod (`COMPOSE_PROJECT_NAME`, ports, test `conftest.py`).
3. `ml/train.py` + `ml/gate.py` + tests.
4. Backend endpoints + bundled-model sync + tests.
5. Dockerfile bundling + `docker-compose.prod.yml`; manual local deploy check.
6. `pipeline_tools.py` + `Jenkinsfile.retrain`; first end-to-end run with `FORCE`.
7. Docker Hub push and rollback demonstration.
8. App button -> Jenkins trigger (optional, last).
9. `docs/DEVOPS.md`, README, screenshots.
