"""Optional: insert real, reviewable sample feedback into SQLite exactly once."""
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.database import engine, init_db
from app.main import initialize_models
from app.models import Feedback
from app.ml_service import ml

SAMPLES = [
    ("Laboratory", 2, "The computers in the laboratory are extremely slow and crash during practical sessions.", "negative", "laboratory", "high"),
    ("Library", 4, "The library staff helped me locate the books for my research project.", "positive", "library", "low"),
    ("Hostel", 1, "Water has not reached our hostel block since last night.", "negative", "hostel", "critical"),
    ("Transport", 3, "Please clarify whether the evening shuttle runs on Saturday.", "neutral", "transport", "medium"),
]

if __name__ == "__main__":
    init_db(); initialize_models()
    with Session(engine) as session:
        if session.scalars(select(Feedback)).first():
            print("Database already contains feedback; no seed records added.")
        else:
            for category, rating, text, sentiment, topic, priority in SAMPLES:
                p=ml.predict(text, category)
                session.add(Feedback(text=text, category=category, rating=rating,
                    predicted_sentiment=p['sentiment']['label'], predicted_topic=p['topic']['label'], predicted_priority=p['priority']['label'],
                    sentiment_confidence=p['sentiment']['confidence'], topic_confidence=p['topic']['confidence'], priority_confidence=p['priority']['confidence'],
                    actual_sentiment=sentiment, actual_topic=topic, actual_priority=priority, reviewed=True))
            session.commit(); print("Inserted 4 real seed feedback records.")
