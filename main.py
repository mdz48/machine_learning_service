from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Preeclampsia Risk Prediction ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    pca = joblib.load("models/pca_5.pkl")
    knn = joblib.load("models/knn_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

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

@app.post("/predict")
def predict_risk(data: PatientMedicalData):
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
            0: "Jóvenes Primigestas Sanas (Riesgo Bajo)",
            1: "Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)",
            2: "Jóvenes Multíparas Sanas (Riesgo Bajo-Medio)",
            3: "Riesgo Metabólico / Obesidad (Riesgo Alto)"
        }
        
        diagnosis_text = diagnosis_map.get(cluster, f"Cluster {cluster}")
        
        return {
            "risk_cluster": int(cluster),
            "diagnosis": diagnosis_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
