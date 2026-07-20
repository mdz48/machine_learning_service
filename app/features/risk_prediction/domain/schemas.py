from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime


class PatientMedicalData(BaseModel):
    age_years: int
    bmi_initial: float
    gestational_week: int
    gestational_trimester: int
    height_cm: float
    initial_weight: float
    weight_kg: float
    weight_gain: float
    systolic: float
    diastolic: float
    mean_arterial_pressure: float
    diabetes: int
    chronic_hypertension: int
    previous_preeclampsia: int
    family_history_hypertension: int
    family_history_heart_disease: int
    chronic_kidney_disease: int
    multiple_pregnancy: int
    active_smoking: int
    previous_pregnancies: int
    previous_deliveries: int
    previous_miscarriages: int
    previous_cesareans: int
    nulliparous: int
    education_level: str
    residence: str
    marital_status: str


class ModelInfoResponse(BaseModel):
    id: int
    version: str
    training_date: str
    algorithm: str
    metadata_config: Dict[str, Any]
    is_active: bool
    created_at: datetime


class PredictionResponse(BaseModel):
    risk_cluster: int
    diagnosis: str
    interpretation: str
    model_version: str
    inference_time_ms: float
    afinidad: Dict[str, float]
    caso_limitrofe: bool
    factores_determinantes: List[Dict[str, Any]]
    pacientes_similares: List[Dict[str, Any]]
    explicacion: str
    recomendaciones: Optional[Dict[str, Any]] = None
