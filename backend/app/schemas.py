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
