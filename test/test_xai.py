"""Pruebas de la capa de explicabilidad (no requiere BD, solo los artefactos del modelo)."""
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import explainability

_pre = joblib.load("models/preprocessor.pkl")
_pca = joblib.load("models/pca.pkl")
_knn = joblib.load("models/knn_model.pkl")
_fc = joblib.load("models/feature_columns.pkl")

PACIENTE_HIPERTENSA = {
    "age_years": 37, "bmi_initial": 35.8, "gestational_week": 30, "gestational_trimester": 3,
    "height_cm": 160, "initial_weight": 80, "weight_kg": 92, "weight_gain": 12,
    "systolic": 152, "diastolic": 98, "mean_arterial_pressure": 116,
    "diabetes": 1, "chronic_hypertension": 1, "previous_preeclampsia": 1,
    "family_history_hypertension": 1, "family_history_heart_disease": 0,
    "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
    "previous_pregnancies": 2, "previous_deliveries": 2, "previous_miscarriages": 0,
    "previous_cesareans": 1, "nulliparous": 0,
    "education_level": "superior", "residence": "urbana", "marital_status": "married",
}


def _explain(patient):
    X = _pca.transform(_pre.transform(pd.DataFrame([patient])[_fc]))
    c = int(_knn.predict(X)[0])
    return c, explainability.explain(_knn, X, patient, c)


def test_estructura_completa():
    c, xai = _explain(PACIENTE_HIPERTENSA)
    assert set(xai) == {"afinidad", "caso_limitrofe", "factores_determinantes",
                        "pacientes_similares", "explicacion"}
    assert explainability.CLUSTER_MAP[c] == "Alto Riesgo Hipertensivo"


def test_afinidad_suma_100():
    _, xai = _explain(PACIENTE_HIPERTENSA)
    assert round(sum(xai["afinidad"].values()), 1) == 100.0


def test_factores_son_clinicamente_correctos():
    # Los factores deben apuntar a hipertension/presion, no a variables atipicas
    _, xai = _explain(PACIENTE_HIPERTENSA)
    top = {f["variable"] for f in xai["factores_determinantes"][:3]}
    assert top & {"chronic_hypertension", "systolic", "diastolic",
                  "mean_arterial_pressure", "previous_preeclampsia"}


def test_vecinas_devuelve_tres():
    _, xai = _explain(PACIENTE_HIPERTENSA)
    assert len(xai["pacientes_similares"]) == 3
    assert all("perfil" in v for v in xai["pacientes_similares"])
