from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.features.risk_prediction.domain.schemas import (
    PatientMedicalData,
    ModelInfoResponse,
    PredictionResponse,
)
from app.features.risk_prediction.services.predictor_service import predictor_service

router = APIRouter(tags=["Risk Prediction"])


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(db: Session = Depends(get_db)):
    model = predictor_service.get_model_info(db)
    if not model:
        raise HTTPException(status_code=404, detail="No hay ningun modelo activo registrado.")
    return {
        "id": model.id,
        "version": model.version,
        "training_date": model.training_date,
        "algorithm": model.algorithm,
        "metadata_config": model.metadata_config,
        "is_active": model.is_active,
        "created_at": model.created_at
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_risk(data: PatientMedicalData, db: Session = Depends(get_db)):
    try:
        return predictor_service.predict(data, db)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    return predictor_service.get_history(db)
