# Smart Feedback System with Auto-Retraining Pipeline

A local, end-to-end feedback analysis prototype built for a college project. It uses trained TF-IDF + Logistic Regression classifiers, SQLite persistence, human review labels, and a real local model-retraining workflow.

## What it does

1. A user submits feedback with a category and rating.
2. FastAPI runs the active local sentiment, topic, and priority classifiers. The star rating is blended into the sentiment and priority predictions (see [Model](#model)).
3. The server stores the feedback and its generated predictions in SQLite.
4. An administrator reviews the feedback and saves corrected labels.
5. A retraining run combines the original labelled CSV with reviewed database records.
6. The candidate is evaluated on a validation split. It becomes the active local model only if its aggregate F1 is strictly higher than the current version; otherwise the current model is retained.

Dashboard numbers, charts, model metrics, feedback rows, training runs, and recommendations are retrieved from the database or active model. They are not UI placeholders.

## Stack

- Frontend: React, Vite, TypeScript, Tailwind CSS, Recharts
- Backend: Python, FastAPI, SQLAlchemy
- Storage: SQLite
- ML: scikit-learn, pandas, numpy, joblib

## Model

- **Three classifiers** (sentiment, topic, priority), each a TF-IDF (unigrams + bigrams) + Logistic Regression pipeline with balanced class weights.
- **Training data:** `backend/data/training_data.csv` holds about 500 labelled campus-feedback sentences across nine topics (teaching, laboratory, cafeteria, hostel, library, transport, facilities, administration, technology). It includes negations ("not taking efforts", "not bad"), neutral questions, short casual inputs ("wifi slow"), and safety issues labelled `critical`. Reviewed feedback from the database is added on every retrain.
- **Category:** the selected category is given to the topic classifier as a context token.
- **Rating-aware predictions:** the 1–5 star rating is a weak signal from the user. The sentiment and priority probabilities are a weighted blend of the text model (60% / 70%) and a rating prior (40% / 30%), so a 1★ rating tips an uncertain prediction towards negative / higher priority, while clearly worded text still wins over a contradicting rating. See `blend_with_rating` in `backend/app/ml_service.py`.
- **Evaluation:** a stratified 75/25 split; the aggregate F1 is the mean weighted F1 of the three classifiers (about 0.87 on the current dataset). A candidate only replaces the live model when this aggregate F1 is strictly higher.

## Project structure

```text
backend/
  app/                 FastAPI app, database models, ML service
  data/training_data.csv
  models/              generated locally at first startup; ignored by Git
  tests/
  seed.py
frontend/
  src/                 React pages, shared UI, API client
```

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- Docker Desktop running, for Docker/Compose and Selenium container workflows
- Jenkins running, only when executing the supplied CI/CD pipeline

## Run locally

Open two terminals from the repository root.

### 1. Backend

```powershell
cd backend
python -m pip install -r requirements.txt
python seed.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

`seed.py` is optional. On an empty database, it inserts four actual feedback records for demonstration. It does nothing when feedback already exists.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/feedback`.

The Vite development server forwards `/api` calls to the FastAPI server at `127.0.0.1:8000`.

## Run with Docker

This starts the complete local container stack. The frontend is available at `http://127.0.0.1:3000`; the backend health endpoint is available at `http://127.0.0.1:8001/api/health`.

```powershell
docker compose up --build
```

Use `Ctrl+C` to stop foreground containers, then remove the stack when needed:

```powershell
docker compose down
```

The frontend container is published on port **3000** so it does not clash with Jenkins, which uses 8080.

### Selenium end-to-end test in Docker

```powershell
docker compose --profile test run --rm e2e
```

The `test` profile enables two extra services: `selenium` (headless Chrome) and `e2e` (the pytest script in `e2e/`). The test opens `/feedback`, analyses and saves a feedback item, then opens `/dashboard`. It reaches the app over the Compose network, so it works regardless of the published port. The first run downloads the Selenium image (about 1–2 GB). Stop the extra containers afterwards with `docker compose --profile test down`.

SQLite data and generated model versions are stored in the named Docker volume `feedback_runtime`. This is intentional: they survive ordinary container restarts. Use `docker compose down --volumes` only when you deliberately want a clean demo database and new initial model.

## Pages

| Route | Purpose |
| --- | --- |
| `/feedback` | Analyse and save new feedback using the active model. |
| `/dashboard` | View feedback counts, trends, distributions, and reviewed-label metrics. |
| `/review` | Apply actual labels to unreviewed feedback. |
| `/models` | Inspect model versions and run retraining. |

## Test

Run backend tests from `backend`:

```powershell
python -m pytest -q
```

The suite includes `tests/test_rating_blend.py` for the rating-aware predictions. The Jenkins agent runs the same suite with Python 3.10.

Build the frontend from `frontend`:

```powershell
npm run build
```

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/feedback/analyze` | Run the active ML models without saving. |
| `POST` | `/api/feedback` | Re-runs prediction server-side and saves feedback. |
| `PUT` | `/api/feedback/{id}/label` | Save verified human labels. |
| `GET` | `/api/analytics/overview` | Database-derived dashboard summary. |
| `POST` | `/api/models/retrain` | Start a real background retraining run. |
| `GET` | `/api/models/training-status/{id}` | Inspect the run stage and result. |

## Jenkins CI/CD pipeline

Two Jenkins pipelines (no Docker needed) live in the repository root:

- `Jenkinsfile` — CI: backend tests, frontend build, optional Selenium smoke test.
- `Jenkinsfile.retrain` — auto-retraining: export reviewed feedback, train a candidate, quality gate (F1 must beat
  the live model), test, package `model-vN.zip`, activate in the running app without a restart, smoke test, and
  automatic rollback on failure. The **Request retraining** button on `/models` can start it.

Create the two Jenkins jobs as *Pipeline script from SCM* jobs pointing at this repository: `feedback-ci` (script path `Jenkinsfile`) and `feedback-retrain` (script path `Jenkinsfile.retrain`). Run `feedback-retrain` once with **Build Now** so Jenkins learns its parameters.

To make the **Request retraining** button start Jenkins instead of training inside the app, start the backend from a PowerShell window that has these variables set (never commit the token):

```powershell
$env:JENKINS_URL="http://localhost:8080"
$env:JENKINS_USER="<jenkins-user-id>"
$env:JENKINS_TOKEN="<api-token>"
$env:JENKINS_JOB="feedback-retrain"
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Build results of `feedback-retrain`:

| Result | Meaning |
| --- | --- |
| SUCCESS | Candidate beat the live F1, passed tests and the smoke test, and is now the active version |
| NOT_BUILT | Skipped (too few new labels without `FORCE`) or rejected by the quality gate; the live model is kept |
| FAILURE | A stage failed after the gate; nothing was deployed, or the previous version was restored (rollback) |

The helper functions in both Jenkinsfiles are named `runCmd` and `pyTool` because `run` and `tool` clash with built-in Jenkins steps.

Setup steps, parameters, demo script and troubleshooting: [`docs/DEVOPS.md`](docs/DEVOPS.md).

## GitHub notes

The repository intentionally ignores the local SQLite database, generated model files, Python caches, Node dependencies, and build output. A clone starts clean: the backend trains its initial models automatically from `backend/data/training_data.csv` on first startup.

To reset your local demo data, stop the backend and remove `backend/smart_feedback.db` and the contents of `backend/models/`. They are generated again when the server starts.
