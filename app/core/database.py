import os
from sqlalchemy import create_engine, Column, Integer, DateTime, String, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "DB_USER": os.getenv("DB_USER"),
    "DB_PASSWORD": os.getenv("DB_PASSWORD"),
    "DB_HOST": os.getenv("DB_HOST"),
    "DB_PORT": os.getenv("DB_PORT"),
    "DB_NAME": os.getenv("DB_NAME"),
    "LOCAL_URL": os.getenv("LOCAL_URL")
}

if all(db_config[k] for k in ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")):
    DATABASE_URL = f"postgresql://{db_config['DB_USER']}:{db_config['DB_PASSWORD']}@{db_config['DB_HOST']}:{db_config['DB_PORT']}/{db_config['DB_NAME']}"
elif db_config["LOCAL_URL"]:
    DATABASE_URL = db_config["LOCAL_URL"]
else:
    raise RuntimeError("Faltan variables de conexión a la BD (DB_* o LOCAL_URL).")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False)
    training_date = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    metadata_config = Column(JSONB, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    inferences = relationship("InferenceRecord", back_populates="model")


class InferenceRecord(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_id = Column(Integer, ForeignKey("ml_models.id"), nullable=False)
    inference_time_ms = Column(Float, nullable=True)
    input_data = Column(JSONB, nullable=False)
    prediction_result = Column(JSONB, nullable=False)

    model = relationship("MLModel", back_populates="inferences")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
