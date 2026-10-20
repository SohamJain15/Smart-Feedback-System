"""Endpoints used by the Jenkins retraining pipeline."""
import csv
import io
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_session
from .ml_service import MODELS, TARGETS, MLService, ml
from .models import Feedback, ModelVersion, TrainingRun
from .schemas import ActivateRequest, ReportRequest

router = APIRouter(prefix="/api")
THRESHOLD = 20
EXPORT_HEADER = ["text", "sentiment", "topic", "priority"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score"]


JENKINS_VARS = ["JENKINS_URL", "JENKINS_USER", "JENKINS_TOKEN", "JENKINS_JOB"]


def jenkins_configured() -> bool:
    return all(os.environ.get(name, "").strip() for name in JENKINS_VARS)


def trigger_jenkins() -> dict:
    """Ask Jenkins to start the retrain job (API-token basic auth)."""
    base = os.environ["JENKINS_URL"].strip().rstrip("/")
    job = os.environ["JENKINS_JOB"].strip()
    credentials = f"{os.environ['JENKINS_USER']}:{os.environ['JENKINS_TOKEN']}".encode("utf-8")
    request = urllib.request.Request(
        f"{base}/job/{urllib.parse.quote(job)}/buildWithParameters?FORCE=true",
        data=b"",
        method="POST",
        headers={"Authorization": "Basic " + base64.b64encode(credentials).decode("ascii")},
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            pass
    except (urllib.error.URLError, OSError) as error:  # HTTPError is a URLError
        raise HTTPException(502, f"Could not reach Jenkins: {getattr(error, 'reason', error)}")
    return {"status": "REQUESTED", "job": job}


def next_version(session: Session) -> str:
    numbers = [
        int(row.version[1:])
        for row in session.scalars(select(ModelVersion)).all()
        if row.version.startswith("v") and row.version[1:].isdigit()
    ]
    return f"v{max(numbers, default=0) + 1}"


def newly_labelled(session: Session) -> int:
    last = session.scalars(
        select(TrainingRun).where(TrainingRun.deployed == True).order_by(TrainingRun.completed_at.desc())  # noqa: E712
    ).first()
    cutoff = last.completed_at if last else datetime.min
    return sum(
        1 for row in session.scalars(select(Feedback)).all() if row.reviewed and row.created_at > cutoff
    )


def rows_to_csv(rows: list[Feedback]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EXPORT_HEADER)
    for row in rows:
        writer.writerow([row.text, row.actual_sentiment, row.actual_topic, row.actual_priority])
    return buffer.getvalue()


@router.get("/models/pipeline-info")
def pipeline_info(session: Session = Depends(get_session)):
    active = session.scalars(select(ModelVersion).where(ModelVersion.status == "ACTIVE")).all()
    if not active:
        raise HTTPException(404, "No active model")
    return {
        "active_version": active[0].version,
        "aggregate_f1": sum(row.f1_score for row in active) / len(active),
        "next_version": next_version(session),
        "newly_labelled": newly_labelled(session),
        "threshold": THRESHOLD,
        "jenkins_managed": jenkins_configured(),
    }


@router.get("/feedback/export")
def export_reviewed(session: Session = Depends(get_session)):
    rows = session.scalars(select(Feedback).where(Feedback.reviewed == True).order_by(Feedback.id)).all()  # noqa: E712
    return Response(content=rows_to_csv(list(rows)), media_type="text/csv")


def _check_folder(version: str) -> None:
    """The three joblib files must exist and really load (raises HTTPException)."""
    folder = MODELS / version
    if not folder.is_dir():
        raise HTTPException(404, f"Model folder for {version} was not found")
    missing = [f"{target}.joblib" for target in TARGETS if not (folder / f"{target}.joblib").is_file()]
    if missing:
        raise HTTPException(422, f"Model folder for {version} is incomplete: missing {', '.join(missing)}")
    try:
        MLService().load(version)  # probe, before anything changes
    except Exception as error:  # corrupt joblib files can raise many exception types
        raise HTTPException(422, f"Model folder for {version} is not usable: {error}")


def _read_metrics(version: str) -> dict:
    """Parse metrics.json of a version the database does not know yet (raises HTTPException)."""
    path = MODELS / version / "metrics.json"
    if not path.is_file():
        raise HTTPException(422, f"Model folder for {version} is incomplete: missing metrics.json")
    try:
        metrics = json.loads(path.read_text())
        metrics["training_samples"] = int(metrics["training_samples"])
        metrics["aggregate_f1"] = float(metrics["aggregate_f1"])
        for target in TARGETS:
            for key in METRIC_KEYS:
                metrics["metrics"][target][key] = float(metrics["metrics"][target][key])
    except Exception as error:
        raise HTTPException(422, f"metrics.json of {version} is not usable: {error}")
    return metrics


@router.post("/models/pipeline-activate")
def pipeline_activate(data: ActivateRequest, session: Session = Depends(get_session)):
    _check_folder(data.version)
    existing = {
        row.model_type: row
        for row in session.scalars(select(ModelVersion).where(ModelVersion.version == data.version)).all()
    }
    known = all(target in existing for target in TARGETS)
    # Versions trained by the app itself have no metrics.json; their numbers are already in the DB.
    metrics = None if known else _read_metrics(data.version)
    new_f1 = (
        sum(existing[t].f1_score for t in TARGETS) / len(TARGETS) if known else metrics["aggregate_f1"]
    )
    samples = existing[TARGETS[0]].training_samples if known else metrics["training_samples"]

    active = session.scalars(select(ModelVersion).where(ModelVersion.status == "ACTIVE")).all()
    if len(active) == len(TARGETS) and all(row.version == data.version for row in active):
        ml.load(data.version)
        return {"active_version": data.version, "aggregate_f1": new_f1}
    if known and data.kind == "deploy":
        raise HTTPException(409, f"{data.version} already exists in the model history; use kind=rollback to restore it")
    old_f1 = sum(row.f1_score for row in active) / len(active) if active else None
    superseded = {row.version for row in active if row.version != data.version}
    for row in active:
        if row.version != data.version:
            row.status = "SUPERSEDED"
    if data.kind == "rollback" and superseded:
        undone = session.scalars(
            select(TrainingRun).where(TrainingRun.version.in_(superseded), TrainingRun.deployed == True)  # noqa: E712
        ).all()
        for run in undone:
            run.deployed, run.status, run.stage = False, "ROLLED_BACK", f"{run.stage} (rolled back)"
    for target in TARGETS:
        if target in existing:
            existing[target].status = "ACTIVE"
        else:
            session.add(ModelVersion(
                version=data.version, model_type=target, training_samples=samples,
                status="ACTIVE", **metrics["metrics"][target],
            ))
    if data.kind == "deploy":
        session.add(TrainingRun(
            version=data.version, status="COMPLETED", stage=data.note or "Activated by Jenkins",
            training_samples=samples, old_f1=old_f1, new_f1=new_f1,
            deployed=True, completed_at=datetime.utcnow(),
        ))
    session.commit()
    ml.load(data.version)
    return {"active_version": data.version, "aggregate_f1": new_f1}


@router.post("/models/pipeline-report")
def pipeline_report(data: ReportRequest, session: Session = Depends(get_session)):
    run = TrainingRun(
        version=data.version, status=data.status, stage=data.note or data.status.title(),
        training_samples=data.training_samples, old_f1=data.old_f1, new_f1=data.new_f1,
        deployed=False, completed_at=datetime.utcnow(),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run
