"""Capa de explicabilidad (XAI) para el clasificador de perfiles de riesgo prenatal.

No reentrena ni modifica el modelo: se apoya en los artefactos exportados por el
notebook (`cluster_stats.json`, `train_reference.csv`) y en el KNN ya entrenado.

Cuatro explicaciones complementarias:
  - afinidad:  % de las k pacientes mas similares en cada perfil (predict_proba).
  - factores:  variables que DEFINEN el perfil y que la paciente exhibe.
  - similares: pacientes historicas mas parecidas (vecinas del KNN).
  - narrativa: explicacion en lenguaje natural para el ginecologo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).parent / "models"

# --- Carga de artefactos (una sola vez, al importar) ---
with open(_DIR / "cluster_stats.json", encoding="utf-8") as _f:
    _STATS = json.load(_f)

CLUSTER_MAP = {int(k): v for k, v in _STATS["cluster_map"].items()}
_NUMERIC = _STATS["numeric_features"]
_GLOBAL = _STATS["global"]
_CLUSTERS = {int(k): v for k, v in _STATS["clusters"].items()}

_REF = pd.read_csv(_DIR / "train_reference.csv")

# Umbral: si la afinidad maxima cae por debajo, se marca como caso limitrofe.
AFFINITY_THRESHOLD = 70.0

# Solo las variables numericas se usan para "factores"; requieren distintividad minima.
_DISTINCTIVENESS_MIN = 0.3

# Etiquetas legibles para el ginecologo.
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

# Columnas que se muestran de cada paciente similar.
_SIMILAR_COLS = ["age_years", "systolic", "diastolic", "bmi_initial", "perfil"]


def _label(var: str) -> str:
    return _LABELS.get(var, var)


def affinity(knn, X_pca):
    """% de pertenencia a cada perfil segun las k vecinas. Devuelve (dict, es_limitrofe)."""
    proba = knn.predict_proba(X_pca)[0]
    result = {CLUSTER_MAP[int(c)]: round(float(p) * 100, 1)
              for c, p in zip(knn.classes_, proba)}
    es_limitrofe = max(result.values()) < AFFINITY_THRESHOLD
    return result, es_limitrofe


def top_factors(patient: dict, cluster: int, k: int = 5):
    """Variables que definen el perfil asignado Y que la paciente exhibe.

    score = distintividad_del_perfil * cuanto_lo_tiene_la_paciente
    (ambas en z-score respecto a la poblacion). Se descartan las variables donde
    la paciente va en direccion contraria al perfil o el perfil no es distintivo.
    """
    factors = []
    for v in _NUMERIC:
        if v not in patient:
            continue
        gm = _GLOBAL[v]["mean"]
        gs = _GLOBAL[v]["std"] or 1.0
        cm = _CLUSTERS[cluster][v]["mean"]

        distintividad = (cm - gm) / gs          # cuanto define este rasgo al perfil (con signo)
        paciente_z = (patient[v] - gm) / gs      # la paciente sobre ese eje
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


def similar_patients(knn, X_pca, n: int = 3):
    """Pacientes historicas mas parecidas (vecinas del KNN, en su espacio PCA)."""
    n = min(n, knn.n_neighbors)
    _, idx = knn.kneighbors(X_pca, n_neighbors=n)
    out = []
    for i in idx[0]:
        row = _REF.iloc[int(i)]
        out.append({
            c: (round(float(row[c]), 1) if c != "perfil" else str(row[c]))
            for c in _SIMILAR_COLS
        })
    return out


def narrative(cluster: int, factors: list) -> str:
    """Explicacion en lenguaje natural, armada con los factores determinantes."""
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
    """Ensambla las cuatro explicaciones en un solo bloque para el JSON de /predict."""
    afin, limitrofe = affinity(knn, X_pca)
    factors = top_factors(patient, cluster)
    return {
        "afinidad": afin,
        "caso_limitrofe": bool(limitrofe),
        "factores_determinantes": factors,
        "pacientes_similares": similar_patients(knn, X_pca),
        "explicacion": narrative(cluster, factors),
    }
