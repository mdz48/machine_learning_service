import os
from sqlalchemy import create_engine, Column, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
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

DATABASE_URL = f"postgresql://{db_config['DB_USER']}:{db_config['DB_PASSWORD']}@{db_config['DB_HOST']}:{db_config['DB_PORT']}/{db_config['DB_NAME']}" or os.getenv("LOCAL_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class InferenceRecord(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(JSONB, nullable=False)
    prediction_result = Column(JSONB, nullable=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
