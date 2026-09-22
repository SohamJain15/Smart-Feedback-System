from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, DateTime
from .database import Base

class Feedback(Base):
    __tablename__="feedback"
    id: Mapped[int] = mapped_column(primary_key=True); text: Mapped[str]; category: Mapped[str]; rating: Mapped[int]
    predicted_sentiment: Mapped[str]; predicted_topic: Mapped[str]; predicted_priority: Mapped[str]
    sentiment_confidence: Mapped[float]; topic_confidence: Mapped[float]; priority_confidence: Mapped[float]
    actual_sentiment: Mapped[Optional[str]] = mapped_column(nullable=True); actual_topic: Mapped[Optional[str]] = mapped_column(nullable=True); actual_priority: Mapped[Optional[str]] = mapped_column(nullable=True)
    reviewed: Mapped[bool] = mapped_column(default=False); created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class ModelVersion(Base):
    __tablename__="model_version"
    id: Mapped[int] = mapped_column(primary_key=True); version: Mapped[str] = mapped_column(index=True); model_type: Mapped[str]
    accuracy: Mapped[float]; precision: Mapped[float]; recall: Mapped[float]; f1_score: Mapped[float]; training_samples: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow); status: Mapped[str] = mapped_column(default="ACTIVE")

class TrainingRun(Base):
    __tablename__="training_run"
    id: Mapped[int] = mapped_column(primary_key=True); version: Mapped[str]; started_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True); status: Mapped[str] = mapped_column(default="RUNNING"); stage: Mapped[str] = mapped_column(default="Preparing dataset")
    training_samples: Mapped[int] = mapped_column(default=0); old_f1: Mapped[Optional[float]] = mapped_column(nullable=True); new_f1: Mapped[Optional[float]] = mapped_column(nullable=True); deployed: Mapped[bool] = mapped_column(default=False); error_message: Mapped[Optional[str]] = mapped_column(nullable=True)

class RetrainingEvent(Base):
    __tablename__="retraining_event"
    id: Mapped[int] = mapped_column(primary_key=True); training_run_id: Mapped[int]; stage: Mapped[str]; created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
