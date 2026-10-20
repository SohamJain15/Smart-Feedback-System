from typing import Literal, Optional
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)
    category: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=1, le=5)

class FeedbackCreate(AnalyzeRequest):
    """The server derives model output. Clients never submit claimed predictions."""

class LabelUpdate(BaseModel):
    actual_sentiment: str = Field(min_length=1, max_length=50)
    actual_topic: str = Field(min_length=1, max_length=100)
    actual_priority: str = Field(min_length=1, max_length=50)


VERSION_PATTERN = r"^v\d+$"

class ActivateRequest(BaseModel):
    version: str = Field(pattern=VERSION_PATTERN)
    kind: Literal["deploy", "rollback"] = "deploy"
    note: str = Field(default="", max_length=300)

class ReportRequest(BaseModel):
    version: str = Field(pattern=VERSION_PATTERN)
    status: Literal["REJECTED", "FAILED"]
    old_f1: Optional[float] = None
    new_f1: Optional[float] = None
    training_samples: int = 0
    note: str = Field(default="", max_length=300)
