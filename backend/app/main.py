from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
from .database import engine, get_session, init_db
from .models import Feedback, ModelVersion, TrainingRun, RetrainingEvent
from .schemas import AnalyzeRequest, FeedbackCreate, LabelUpdate
from .ml_service import ml, base_dataset, TARGETS
from .pipeline_api import router as pipeline_router, next_version, newly_labelled, THRESHOLD, jenkins_configured, trigger_jenkins

def all_rows(session, statement): return session.scalars(statement).all()
def active_versions(session): return all_rows(session, select(ModelVersion).where(ModelVersion.status == "ACTIVE"))

def initialize_models():
    with Session(engine) as session:
        active = active_versions(session)
        if len(active) == 3:
            ml.load(active[0].version); return
        frame = base_dataset(); trained, metrics = ml.train(frame); version = "v1"; ml.save(version, trained)
        for target in TARGETS:
            m = metrics[target]
            session.add(ModelVersion(version=version, model_type=target, training_samples=len(frame), status="ACTIVE", **m))
        session.commit(); ml.load(version)

@asynccontextmanager
async def lifespan(app):
    init_db(); initialize_models(); yield

app = FastAPI(title="Smart Feedback System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(pipeline_router)

def metrics_for(session, field):
    rows = all_rows(session, select(Feedback).where(Feedback.reviewed == True))
    if len(rows) < 2: return None
    actual = [getattr(r, f"actual_{field}") for r in rows]
    predicted = [getattr(r, f"predicted_{field}") for r in rows]
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p, rec, f, _ = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(actual,predicted), "precision": p, "recall": rec, "f1_score": f, "samples": len(rows)}

@app.get("/api/health")
def health(): return {"status": "ok", "active_version": ml.version}

@app.get("/api/meta/options")
def options():
    """Controlled values exposed from the active trained models, for forms and review."""
    labels = ml.label_options()
    return {"categories": [value.title() for value in labels["topic"]], "labels": labels}

@app.post("/api/feedback/analyze")
def analyze(data: AnalyzeRequest): return {"model_version": ml.version, "predictions": ml.predict(data.text, data.category, data.rating)}

@app.post("/api/feedback")
def create_feedback(data: FeedbackCreate, session: Session = Depends(get_session)):
    prediction = ml.predict(data.text, data.category, data.rating)
    record = Feedback(
        **data.model_dump(),
        predicted_sentiment=prediction["sentiment"]["label"],
        predicted_topic=prediction["topic"]["label"],
        predicted_priority=prediction["priority"]["label"],
        sentiment_confidence=prediction["sentiment"]["confidence"],
        topic_confidence=prediction["topic"]["confidence"],
        priority_confidence=prediction["priority"]["confidence"],
    )
    session.add(record); session.commit(); session.refresh(record); return record

@app.get("/api/feedback")
def feedback(session: Session = Depends(get_session)):
    return all_rows(session, select(Feedback).order_by(Feedback.created_at.desc()))

@app.get("/api/feedback/unreviewed")
def unreviewed(session: Session = Depends(get_session)):
    return all_rows(session, select(Feedback).where(Feedback.reviewed == False).order_by(Feedback.created_at.desc()))

@app.put("/api/feedback/{feedback_id}/label")
def label(feedback_id: int, data: LabelUpdate, session: Session = Depends(get_session)):
    record = session.get(Feedback, feedback_id)
    if not record: raise HTTPException(404, "Feedback not found")
    allowed = ml.label_options()
    submitted = {"sentiment": data.actual_sentiment, "topic": data.actual_topic, "priority": data.actual_priority}
    invalid = [kind for kind, value in submitted.items() if value not in allowed[kind]]
    if invalid:
        raise HTTPException(422, f"Invalid label value for: {', '.join(invalid)}")
    record.actual_sentiment, record.actual_topic, record.actual_priority = data.actual_sentiment, data.actual_topic, data.actual_priority
    record.reviewed = True; session.add(record); session.commit(); session.refresh(record); return record

@app.get("/api/analytics/overview")
def overview(session: Session = Depends(get_session)):
    rows = all_rows(session, select(Feedback)); total = len(rows)
    counts = {s: sum(r.predicted_sentiment == s for r in rows) for s in ["positive", "neutral", "negative"]}
    return {"total_feedback": total, "positive_percent": round(100*counts['positive']/total,2) if total else 0, "neutral_percent": round(100*counts['neutral']/total,2) if total else 0, "negative_percent": round(100*counts['negative']/total,2) if total else 0, "high_priority": sum(r.predicted_priority in ['high','critical'] for r in rows), "reviewed": sum(r.reviewed for r in rows), "unreviewed": sum(not r.reviewed for r in rows), "metrics": {t: metrics_for(session,t) for t in TARGETS}}

@app.get("/api/analytics/sentiment")
def sentiment(session: Session = Depends(get_session)):
    rows=all_rows(session, select(Feedback)); return [{"name": x.title(), "value":sum(r.predicted_sentiment==x for r in rows)} for x in ["positive","neutral","negative"]]

@app.get("/api/analytics/topics")
def topics(session: Session = Depends(get_session)):
    rows=all_rows(session, select(Feedback)); names=sorted(set(r.predicted_topic for r in rows)); return [{"name":n.title(),"value":sum(r.predicted_topic==n for r in rows)} for n in names]

@app.get("/api/analytics/priorities")
def priorities(session: Session = Depends(get_session)):
    rows = all_rows(session, select(Feedback))
    return [{"name": level.title(), "value": sum(r.predicted_priority == level for r in rows)} for level in ["low", "medium", "high", "critical"]]

@app.get("/api/analytics/trends")
def trends(session: Session = Depends(get_session)):
    rows=all_rows(session, select(Feedback)); dates=sorted(set(r.created_at.date().isoformat() for r in rows)); return [{"date": d, **{x:sum(r.created_at.date().isoformat()==d and r.predicted_sentiment==x for r in rows) for x in ['positive','negative','neutral']}, "total":sum(r.created_at.date().isoformat()==d for r in rows)} for d in dates]

@app.get("/api/models/current")
def current(session: Session = Depends(get_session)):
    records=active_versions(session)
    if not records: raise HTTPException(404, "No active model")
    return {"version":records[0].version,"models":records,"aggregate_f1":sum(x.f1_score for x in records)/len(records)}

@app.get("/api/models/history")
def history(session: Session = Depends(get_session)): return all_rows(session, select(ModelVersion).order_by(ModelVersion.id.desc()))

@app.get("/api/models/retraining-status")
def retraining_status(session: Session = Depends(get_session)):
    current_version = active_versions(session)[0].version
    count = newly_labelled(session)
    return {"newly_labelled": count, "threshold": THRESHOLD, "recommended": count >= THRESHOLD, "active_version": current_version}

def run_training(run_id):
    with Session(engine) as session:
        run=session.get(TrainingRun,run_id)
        try:
            run.stage="Preparing dataset"; session.add(run); session.add(RetrainingEvent(training_run_id=run_id,stage=run.stage)); session.commit()
            base=base_dataset(); reviewed=all_rows(session, select(Feedback).where(Feedback.reviewed == True))
            extra=pd.DataFrame([{"text":x.text,"sentiment":x.actual_sentiment,"topic":x.actual_topic,"priority":x.actual_priority} for x in reviewed])
            frame=pd.concat([base,extra],ignore_index=True); run.training_samples=len(frame); run.stage="Training"; session.add(run); session.add(RetrainingEvent(training_run_id=run_id,stage=run.stage)); session.commit()
            trained, metrics=ml.train(frame); run.stage="Evaluating"; session.add(run); session.add(RetrainingEvent(training_run_id=run_id,stage=run.stage)); session.commit()
            old=active_versions(session); old_f1=sum(x.f1_score for x in old)/len(old); new_f1=sum(metrics[t]['f1_score'] for t in TARGETS)/3; run.old_f1=old_f1; run.new_f1=new_f1; run.stage="Comparing"; session.add(run); session.add(RetrainingEvent(training_run_id=run_id,stage=run.stage)); session.commit()
            if new_f1 > old_f1:
                run.stage="Deploying"; ml.save(run.version,trained)
                for item in old: item.status="REJECTED"; session.add(item)
                for target in TARGETS: session.add(ModelVersion(version=run.version,model_type=target,training_samples=len(frame),status="ACTIVE",**metrics[target]))
                run.deployed=True; run.status="COMPLETED"; run.stage="Activated locally"; ml.load(run.version)
            else: run.stage="Candidate rejected"; run.status="REJECTED"
            run.completed_at=datetime.utcnow(); session.add(run); session.add(RetrainingEvent(training_run_id=run_id,stage=run.stage)); session.commit()
        except Exception as exc:
            run=session.get(TrainingRun,run_id); run.status="FAILED"; run.error_message=str(exc); run.completed_at=datetime.utcnow(); session.add(run); session.commit()

@app.post("/api/models/retrain")
def retrain(background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    if jenkins_configured(): return trigger_jenkins()  # Jenkins owns retraining when configured
    running=session.scalars(select(TrainingRun).where(TrainingRun.status=="RUNNING")).first()
    if running: raise HTTPException(409,"A retraining run is already active")
    run=TrainingRun(version=next_version(session)); session.add(run); session.commit(); session.refresh(run); background_tasks.add_task(run_training,run.id); return run

@app.get("/api/models/training-runs")
def runs(session: Session = Depends(get_session)): return all_rows(session, select(TrainingRun).order_by(TrainingRun.started_at.desc()))

@app.get("/api/models/training-status/{run_id}")
def run_status(run_id:int, session:Session=Depends(get_session)):
    run=session.get(TrainingRun,run_id)
    if not run: raise HTTPException(404,"Training run not found")
    events=all_rows(session, select(RetrainingEvent).where(RetrainingEvent.training_run_id==run_id).order_by(RetrainingEvent.created_at))
    return {"run":run,"events":events}
