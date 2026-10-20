# DevOps pipeline — Smart Feedback System with Auto-Retraining

Jenkins retrains the sentiment/topic/priority models from admin-reviewed feedback, tests them,
versions them as build artifacts, activates the new model in the running app without a restart,
checks it, and rolls back automatically if anything fails. No Docker is needed for any of this.

Design: `docs/superpowers/specs/2026-10-06-auto-retraining-pipeline-design.md`
Plan: `docs/superpowers/plans/2026-10-06-jenkins-auto-retraining.md`

## 1. How it fits together

```
 Running app (your PC)               Jenkins (same PC)
 uvicorn :8000                       Job: feedback-retrain   (Jenkinsfile.retrain)
 backend\models\  <── copy ───────── Setup  -> Info -> Export labels -> Train candidate
 SQLite database                              -> Quality gate -> Test candidate -> Package model
   GET  /api/models/pipeline-info  <────────  -> Deploy (activate) -> Smoke test
   GET  /api/feedback/export       <────────        on failure: roll back + report FAILED
   POST /api/models/pipeline-activate <─────
   POST /api/models/pipeline-report   <─────  Job: feedback-ci   (Jenkinsfile)
                                              Checkout -> Backend tests -> Frontend build -> (Selenium)
```

| Piece | Where |
|---|---|
| Training CLI | `backend/ml/train.py` |
| Quality gate (strictly higher aggregate F1 than the live model) | `backend/ml/gate.py` |
| Helper Jenkins calls (info / export / package / deploy / wait-healthy / report) | `backend/ml/pipeline_tools.py` |
| Pipeline endpoints | `backend/app/pipeline_api.py` |
| CI pipeline | `Jenkinsfile` |
| Retrain / deploy / rollback pipeline | `Jenkinsfile.retrain` |

## 2. One-time setup

1. **Jenkins tools.** Python, Git, and Node (for the frontend build) must be on the PATH that Jenkins sees.
   If a build says "'python' is not recognized", add the folders under *Manage Jenkins -> System ->
   Global properties -> Environment variables* (name `PATH`) and restart Jenkins.
2. **Job `feedback-ci`**: New Item -> Pipeline -> *Pipeline script from SCM* -> Git
   `https://github.com/SohamJain15/Smart-Feedback-System.git`, branch `*/main`, script path `Jenkinsfile`.
3. **Job `feedback-retrain`**: same repo and branch, script path `Jenkinsfile.retrain`.
   **Run it once with *Build Now*.** Jenkins only learns the parameters from the first run, and
   *Build with Parameters* (and the app's button) work only after that. The first run will most likely
   end as `NOT_BUILT` (not enough labels) — that is correct.
4. **Folder permissions.** The Windows account running the Jenkins service must be able to write to
   `D:\PS4\backend\models` (the default `APP_MODELS_DIR`). If a deploy fails with "Permission denied",
   change the service's *Log On* account in Windows Services to your own user and restart it.

## 3. Running the app so the pipeline can reach it

```powershell
cd D:\PS4\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Optional — make the **Request retraining** button on `/models` start the Jenkins job. Create an API token in
Jenkins (your user menu -> Security -> API Token), then start the backend from the same PowerShell window:

```powershell
$env:JENKINS_URL="http://localhost:8080"
$env:JENKINS_USER="<your-jenkins-user>"
$env:JENKINS_TOKEN="<api-token>"
$env:JENKINS_JOB="feedback-retrain"
cd D:\PS4\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Never put the token in a file in the repository. With all four variables set, the button asks Jenkins
to retrain (and nothing is trained inside the app); without them, the button trains inside the app as before.

## 4. Job parameters (`feedback-retrain`)

| Parameter | Default | Meaning |
|---|---|---|
| `FORCE` | off | Retrain even if fewer than `MIN_NEW_LABELS` new reviewed labels exist |
| `MIN_NEW_LABELS` | 20 | Threshold for the nightly run (02:xx) |
| `APP_URL` | `http://127.0.0.1:8000` | Where the running backend listens |
| `APP_MODELS_DIR` | `D:\PS4\backend\models` | The models folder the running backend reads |
| `SIMULATE_FAILURE` | off | Demo: makes the smoke test fail to show the automatic rollback |

Build results: **SUCCESS** = new model live; **NOT_BUILT** = skipped (few labels) or rejected (not better);
**FAILURE** = something broke, previous model restored; **UNSTABLE** = model is fine but the history
report could not be saved.

## 5. Demo script

1. Open `http://127.0.0.1:5173/review`, label at least a few feedback items (20+ for the nightly rule).
2. Run `feedback-retrain` -> *Build with Parameters* -> tick `FORCE` -> Build (or click **Request retraining** in the app).
3. Watch the stage view. Show the build description (`v7 -> v8  F1 0.xxxx -> 0.yyyy`) and the archived
   `model-v8.zip` + `metrics.json` under *Build Artifacts*.
4. Open `http://127.0.0.1:5173/models`: the new version is active and the run is in the history.
5. **Rejection:** run again with `FORCE` and no new labels. If the new candidate is not better, the build is
   `NOT_BUILT`, `gate.txt` says why, and the app keeps its model (history shows `REJECTED`).
6. **Rollback:** run with `FORCE` + `SIMULATE_FAILURE`. The smoke test fails, the build turns red,
   the description says `ROLLED BACK to vN`, and `/models` shows the previous version active again.

## 6. Screenshot checklist (for the report)

- [ ] Jenkins stage view of a successful `feedback-retrain` build
- [ ] Build page with the description and archived `model-vN.zip`
- [ ] `/models` page before and after (version, F1, history row)
- [ ] A `NOT_BUILT` rejection with `gate.txt`
- [ ] The rollback build (red) and `/models` back on the previous version
- [ ] `feedback-ci` build with the test-result trend
- [ ] The GitHub repository with `Jenkinsfile` and `Jenkinsfile.retrain`

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ERROR: cannot reach http://127.0.0.1:8000/...` | The backend is not running, or `APP_URL` is wrong |
| `'python' is not recognized` | Add Python to Jenkins' PATH (setup step 1) |
| Deploy fails with "Permission denied" | The Jenkins service account cannot write `APP_MODELS_DIR` (setup step 4) |
| Button says `Could not reach Jenkins: Bad Request` | Usually: run `feedback-retrain` once manually so Jenkins registers its parameters |
| Button says `Could not reach Jenkins: Unauthorized` (or `Forbidden`) | Wrong `JENKINS_USER` / `JENKINS_TOKEN` |
| Build is `NOT_BUILT` right away | Fewer than `MIN_NEW_LABELS` new labels — tick `FORCE` |

## 8. Syllabus mapping

| Practical | Evidence |
|---|---|
| 1 Git / GitHub | The repository, commit history, pull requests |
| 2 Jenkins install and setup | Two Jenkins jobs, plugins and credentials (this document) |
| 3 CI / CD with Jenkins | `Jenkinsfile` (CI) and `Jenkinsfile.retrain` (CD) |
| 4, 5 Docker | `Dockerfile`s and `docker-compose.yml` are in the repository but are not run by the pipeline |
| 6, 7 Selenium | `e2e/test_feedback_flow.py`; optional `RUN_E2E` stage runs it locally with Chrome |
| 8 ML model in a CI/CD pipeline | Train, quality gate, test, package, activate in `Jenkinsfile.retrain` |
| 9 Deploy models through Jenkins | Versioned model artifact, activation without restart, smoke test, rollback |
| 10 Mini project: smart feedback with auto-retraining | The whole system |

## 9. Known limitations

- The candidate's F1 and the live model's F1 are measured on different validation splits (the dataset grows
  between versions), so the gate is a practical heuristic, not a statistical guarantee.
- The pipeline endpoints have no authentication, like the rest of this local prototype API.
- Jenkins and the app must run on the same machine (the pipeline copies the model into the app's models folder).
