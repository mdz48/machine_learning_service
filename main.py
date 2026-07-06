from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from database import engine, Base, get_db, InferenceRecord, MLModel, SessionLocal
import explainability
import time

app = FastAPI(title="ML Service", root_path="/ml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
Base.metadata.create_all(bind=engine)

# Load models at startup
try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    pca = joblib.load("models/pca.pkl")
    knn = joblib.load("models/knn_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    with open("models/model_metadata.json", encoding="utf-8") as _f:
        MODEL_METADATA = json.load(_f)
    # Centroides clínicos por clúster (bmi/systolic/diastolic) para detección de atípicos.
    # Se cargan del artefacto para no desincronizarse al reentrenar.
    CENTROIDS = {int(k): v for k, v in MODEL_METADATA["centroids"].items()}
except Exception as e:
    print(f"Error loading models: {e}")
    CENTROIDS = {}

try:
    _startup_db = SessionLocal()
    ACTIVE_MODEL = _startup_db.query(MLModel).filter(MLModel.is_active == True).first()
    if not ACTIVE_MODEL:
        raise RuntimeError("No hay ningun modelo activo en la base de datos. Corre seed_model.py primero.")
    ACTIVE_MODEL_ID = ACTIVE_MODEL.id
    print(f"Modelo activo cargado: id={ACTIVE_MODEL_ID}, version={ACTIVE_MODEL.version}")
except Exception as e:
    print(f"Error consultando modelo activo: {e}")
    ACTIVE_MODEL = None
    ACTIVE_MODEL_ID = None
finally:
    _startup_db.close()

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

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Service is running"}

@app.get("/model/info")
def get_model_info(db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.is_active == True).first()
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

@app.post("/predict")
def predict_risk(data: PatientMedicalData, db: Session = Depends(get_db)):
    start_time = time.perf_counter()
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.model_dump()])
        
        # Ensure exact column order
        df = df[feature_columns]
        
        # 1. Preprocess
        X_pre = preprocessor.transform(df)
        
        # 2. PCA
        X_pca = pca.transform(X_pre)
        
        # 3. Predict
        cluster = knn.predict(X_pca)[0]
        
        # Interpret cluster based on Data Mining Profiling
        diagnosis_map = {
            0: "Primigestas Sanas (Riesgo Bajo-Medio)",
            1: "Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)",
            2: "Multíparas Sanas (Riesgo Bajo)",
            3: "Riesgo Metabólico / Obesidad (Riesgo Alto)"
        }
        
        diagnosis_text = diagnosis_map.get(cluster, f"Cluster {cluster}")
        
        # Interpretaciones estáticas (Opción 1)
        interpretations = {
            0: "Paciente con características clínicas normales. Se sugiere control prenatal estándar.",
            1: "Paciente con alto riesgo hipertensivo/preeclampsia. Monitoreo estricto de presión arterial recomendado.",
            2: "Paciente multípara con características normales. Mantener protocolo prenatal habitual.",
            3: "Paciente con marcadores de riesgo metabólico. Se sugiere evaluación nutricional y control de peso."
        }
        
        # Detección de atípicos: centroides cargados del artefacto (models/model_metadata.json)
        c_stats = CENTROIDS.get(int(cluster), {})
        anomalies = []
        if c_stats:
            if data.systolic > c_stats["systolic"] * 1.15:
                anomalies.append(f"Nota: Presión sistólica ({data.systolic}) elevada respecto al promedio de este grupo.")
            if data.bmi_initial > c_stats["bmi_initial"] * 1.20:
                anomalies.append(f"Nota: IMC inicial ({data.bmi_initial}) elevado respecto al promedio de este grupo.")
                
        base_interp = interpretations.get(cluster, "Sin interpretación disponible.")
        if anomalies:
            base_interp += " " + " ".join(anomalies)
        
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        model_version = ACTIVE_MODEL.version if ACTIVE_MODEL else "unknown"
        
        # Capa de explicabilidad (XAI): afinidad, factores, similares, narrativa
        xai = explainability.explain(knn, X_pca, data.model_dump(), int(cluster))

        result = {
            "risk_cluster": int(cluster),
            "diagnosis": diagnosis_text,
            "interpretation": base_interp,
            "model_version": model_version,
            "inference_time_ms": inference_time_ms,
            **xai
        }
        
        # Save inference to database
        db_record = InferenceRecord(
            model_id=ACTIVE_MODEL_ID,
            inference_time_ms=inference_time_ms,
            input_data=data.model_dump(),
            prediction_result=result
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    inferences = db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).all()
    return inferences
