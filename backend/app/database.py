from pathlib import Path
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = f"sqlite:///{ROOT / 'smart_feedback.db'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
class Base(DeclarativeBase): pass

def init_db() -> None:
    Base.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
