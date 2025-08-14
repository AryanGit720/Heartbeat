from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Use /tmp by default so containers always have write access
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/tmp/hsc"))
DB_PATH = Path(os.getenv("DB_PATH", str(RUNTIME_DIR / "app.db")))
DB_DIR = DB_PATH.parent
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Analysis(Base):
    __tablename__ = "analysis"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    filename = Column(String(512))
    stored_path = Column(String(1024))
    primary_prediction = Column(String(64))
    confidence = Column(Float)
    probs_json = Column(Text)
    highlight_segments_json = Column(Text)
    segment_len = Column(Float)
    segment_hop = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)

def add_analysis(session_id: str, filename: str, stored_path: str, result: dict) -> int:
    with SessionLocal() as db:
        row = Analysis(
            session_id=session_id,
            filename=filename,
            stored_path=str(Path(stored_path).resolve()),
            primary_prediction=result["record"]["primary_prediction"],
            confidence=float(result["record"]["confidence"]),
            probs_json=json.dumps(result["record"]["probs"]),
            highlight_segments_json=json.dumps(result.get("highlight_segments", [])),
            segment_len=float(result["segment_seconds"]["length"]),
            segment_hop=float(result["segment_seconds"]["hop"]),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id

def get_history(session_id: str, limit: int = 200) -> List[Analysis]:
    with SessionLocal() as db:
        return (
            db.query(Analysis)
            .filter(Analysis.session_id == session_id)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
            .all()
        )

def get_analysis(analysis_id: int) -> Optional[Analysis]:
    with SessionLocal() as db:
        return db.get(Analysis, analysis_id)

def delete_analysis(analysis_id: int, session_id: Optional[str] = None, delete_file: bool = True) -> bool:
    with SessionLocal() as db:
        row = db.get(Analysis, analysis_id)
        if not row:
            return False
        if session_id and row.session_id != session_id:
            return False
        if delete_file and row.stored_path:
            try:
                Path(row.stored_path).unlink(missing_ok=True)
            except Exception:
                pass
        db.delete(row)
        db.commit()
        return True