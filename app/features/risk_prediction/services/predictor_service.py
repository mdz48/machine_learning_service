import joblib
import json
import time
import pandas as pd
from typing import Dict, Any, Tuple
from sqlalchemy.orm import Session

from app.core.config import (
    PREPROCESSOR_PATH,
    PCA_PATH,
    KNN_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_METADATA_PATH,
)
from app.core.database import InferenceRecord, MLModel, SessionLocal
from app.features.risk_prediction.domain.schemas import PatientMedicalData
from app.features.risk_prediction.services.explainability_service import explain
from app.features.risk_prediction.services.recommendation_service import get_recommendations


class PredictorService:
    def __init__(self):
        self.preprocessor = None
        self.pca = None
        self.knn = None
        self.feature_columns = None
        self.model_metadata = None
        self.centroids = {}
        self.active_model_id = None
        self.active_model_version = "unknown"
        self._load_artifacts()
        self._load_active_model()

    def _load_artifacts(self):
        try:
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.pca = joblib.load(PCA_PATH)
            self.knn = joblib.load(KNN_PATH)
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            with open(MODEL_METADATA_PATH, encoding="utf-8") as _f:
                self.model_metadata = json.load(_f)
            self.centroids = {int(k): v for k, v in self.model_metadata["centroids"].items()}
        except Exception as e:
            print(f"Error loading predictor models/artifacts: {e}")

    def _load_active_model(self):
        db = SessionLocal()
        try:
            active_model = db.query(MLModel).filter(MLModel.is_active == True).first()
            if active_model:
                self.active_model_id = active_model.id
                self.active_model_version = active_model.version
                print(f"Modelo activo cargado: id={self.active_model_id}, version={self.active_model_version}")
        except Exception as e:
            print(f"Error consultando modelo activo: {e}")
        finally:
            db.close()

    def get_model_info(self, db: Session) -> MLModel:
        model = db.query(MLModel).filter(MLModel.is_active == True).first()
        return model

    def predict(self, data: PatientMedicalData, db: Session) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        df = pd.DataFrame([data.model_dump()])
        df = df[self.feature_columns]
        
        X_pre = self.preprocessor.transform(df)
        X_pca = self.pca.transform(X_pre)
        cluster = self.knn.predict(X_pca)[0]
        
        diagnosis_map = {
            0: "Primigestas Sanas (Riesgo Bajo-Medio)",
            1: "Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)",
            2: "Multíparas Sanas (Riesgo Bajo)",
            3: "Riesgo Metabólico / Obesidad (Riesgo Alto)"
        }
        diagnosis_text = diagnosis_map.get(cluster, f"Cluster {cluster}")
        
        interpretations = {
            0: "Paciente con características clínicas normales. Se sugiere control prenatal estándar.",
            1: "Paciente con alto riesgo hipertensivo/preeclampsia. Monitoreo estricto de presión arterial recomendado.",
            2: "Paciente multípara con características normales. Mantener protocolo prenatal habitual.",
            3: "Paciente con marcadores de riesgo metabólico. Se sugiere evaluación nutricional y control de peso."
        }
        
        c_stats = self.centroids.get(int(cluster), {})
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
        
        xai = explain(self.knn, X_pca, data.model_dump(), int(cluster))

        result = {
            "risk_cluster": int(cluster),
            "diagnosis": diagnosis_text,
            "interpretation": base_interp,
            "model_version": self.active_model_version,
            "inference_time_ms": inference_time_ms,
            **xai
        }

        recs = get_recommendations(int(cluster), data.gestational_week)
        if recs:
            result["recomendaciones"] = recs
        
        # Save inference
        if self.active_model_id:
            db_record = InferenceRecord(
                model_id=self.active_model_id,
                inference_time_ms=inference_time_ms,
                input_data=data.model_dump(),
                prediction_result=result
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
        
        return result

    def get_history(self, db: Session):
        return db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).all()


# Singleton instance
predictor_service = PredictorService()
