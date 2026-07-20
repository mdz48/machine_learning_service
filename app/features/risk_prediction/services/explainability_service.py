"""Capa de explicabilidad (XAI) para el clasificador de perfiles de riesgo prenatal."""
import json
import numpy as np
import pandas as pd
from app.core.config import MODELS_DIR, os

# Carga de artefactos
with open(os.path.join(MODELS_DIR, "cluster_stats.json"), encoding="utf-8") as _f:
    _STATS = json.load(_f)

CLUSTER_MAP = {int(k): v for k, v in _STATS["cluster_map"].items()}
_NUMERIC = _STATS["numeric_features"]
_GLOBAL = _STATS["global"]
_CLUSTERS = {int(k): v for k, v in _STATS["clusters"].items()}

_REF = pd.read_csv(os.path.join(MODELS_DIR, "train_reference.csv"))

AFFINITY_THRESHOLD = 70.0
_DISTINCTIVENESS_MIN = 0.3

_LABELS = {
    "age_years": "Edad materna",
    "bmi_initial": "Índice de masa corporal (IMC)",
    "gestational_week": "Semana gestacional",
    "gestational_trimester": "Trimestre gestacional",
    "height_cm": "Estatura",
    "initial_weight": "Peso inicial",
    "weight_kg": "Peso actual",
    "weight_gain": "Ganancia de peso",
    "systolic": "Presión sistólica",
    "diastolic": "Presión diastólica",
    "mean_arterial_pressure": "Presión arterial media",
    "diabetes": "Diabetes",
    "chronic_hypertension": "Hipertensión crónica",
    "previous_preeclampsia": "Preeclampsia previa",
    "family_history_hypertension": "Antecedente familiar de hipertensión",
    "family_history_heart_disease": "Antecedente familiar de cardiopatía",
    "chronic_kidney_disease": "Enfermedad renal crónica",
    "multiple_pregnancy": "Embarazo múltiple",
    "active_smoking": "Tabaquismo activo",
    "previous_pregnancies": "Embarazos previos",
    "previous_deliveries": "Partos previos",
    "previous_miscarriages": "Abortos previos",
    "previous_cesareans": "Cesáreas previas",
    "nulliparous": "Nuliparidad",
}

_SIMILAR_COLS = ["age_years", "systolic", "diastolic", "bmi_initial", "perfil"]


def _label(var: str) -> str:
    return _LABELS.get(var, var)


def affinity(knn, X_pca):
    proba = knn.predict_proba(X_pca)[0]
    result = {CLUSTER_MAP[int(c)]: round(float(p) * 100, 1)
              for c, p in zip(knn.classes_, proba)}
    es_limitrofe = max(result.values()) < AFFINITY_THRESHOLD
    return result, es_limitrofe


def top_factors(patient: dict, cluster: int, k: int = 5):
    factors = []
    for v in _NUMERIC:
        if v not in patient:
            continue
        gm = _GLOBAL[v]["mean"]
        gs = _GLOBAL[v]["std"] or 1.0
        cm = _CLUSTERS[cluster][v]["mean"]

        distintividad = (cm - gm) / gs
        paciente_z = (patient[v] - gm) / gs
        if abs(distintividad) < _DISTINCTIVENESS_MIN:
            continue
        if np.sign(distintividad) != np.sign(paciente_z):
            continue

        factors.append({
            "variable": v,
            "etiqueta": _label(v),
            "valor_paciente": round(float(patient[v]), 2),
            "promedio_perfil": round(float(cm), 2),
            "score": round(float(abs(distintividad) * abs(paciente_z)), 2),
        })

    factors.sort(key=lambda d: d["score"], reverse=True)
    return factors[:k]


def _clean_num(value):
    return None if pd.isna(value) else round(float(value), 1)


def similar_patients(knn, X_pca, n: int = 3):
    n = min(n, knn.n_neighbors)
    _, idx = knn.kneighbors(X_pca, n_neighbors=n)
    out = []
    for i in idx[0]:
        row = _REF.iloc[int(i)]
        out.append({
            c: (str(row[c]) if c == "perfil" else _clean_num(row[c]))
            for c in _SIMILAR_COLS
        })
    return out


def narrative(cluster: int, factors: list) -> str:
    perfil = CLUSTER_MAP[cluster]
    if not factors:
        return f"La paciente fue asignada al perfil «{perfil}»."
    causas = ", ".join(f["etiqueta"].lower() for f in factors[:3])
    return (
        f"La paciente fue asignada al perfil «{perfil}» debido principalmente a: "
        f"{causas}. Estos rasgos coinciden con el patrón clínico característico "
        f"de este grupo."
    )


def explain(knn, X_pca, patient: dict, cluster: int) -> dict:
    afin, limitrofe = affinity(knn, X_pca)
    factors = top_factors(patient, cluster)
    return {
        "afinidad": afin,
        "caso_limitrofe": bool(limitrofe),
        "factores_determinantes": factors,
        "pacientes_similares": similar_patients(knn, X_pca),
        "explicacion": narrative(cluster, factors),
    }
