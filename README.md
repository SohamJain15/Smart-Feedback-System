# Smart Feedback System with Auto-Retraining Pipeline

A local, end-to-end feedback analysis prototype built for a college project. It uses trained TF-IDF + Logistic Regression classifiers, SQLite persistence, human review labels, and a real local model-retraining workflow.

## What it does

1. A user submits feedback with a category and rating.
2. FastAPI runs the active local sentiment, topic, and priority classifiers.
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

## GitHub notes

The repository intentionally ignores the local SQLite database, generated model files, Python caches, Node dependencies, and build output. A clone starts clean: the backend trains its initial models automatically from `backend/data/training_data.csv` on first startup.

To reset your local demo data, stop the backend and remove `backend/smart_feedback.db` and the contents of `backend/models/`. They are generated again when the server starts.
