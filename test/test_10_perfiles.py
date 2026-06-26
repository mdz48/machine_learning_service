import joblib
import pandas as pd
import json

feature_columns = joblib.load("models/feature_columns.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
pca = joblib.load("models/pca_5.pkl")
knn = joblib.load("models/knn_model.pkl")

diagnosis_map = {
    0: "Primigestas Sanas (Riesgo Bajo-Medio)",
    1: "Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Critico)",
    2: "Multiparas Sanas (Riesgo Bajo)",
    3: "Riesgo Metabolico / Obesidad (Riesgo Alto)"
}

# Categorias validas del modelo:
# education_level: primaria | secundaria | superior
# residence: rural | urbana
# marital_status: married | single

perfiles = [
    {
        "desc": "#1 — Primigesta joven y sana (esperado: Cluster 0)",
        "age_years": 23, "bmi_initial": 21.5, "gestational_week": 14, "gestational_trimester": 2,
        "height_cm": 163.0, "initial_weight": 57.0, "weight_kg": 60.0, "weight_gain": 3.0,
        "systolic": 110.0, "diastolic": 70.0, "mean_arterial_pressure": 83.0,
        "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 0, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 0, "previous_deliveries": 0, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 1,
        "education_level": "superior", "residence": "urbana", "marital_status": "single"
    },
    {
        "desc": "#2 — Multipara joven y sana, 2 partos previos (esperado: Cluster 2)",
        "age_years": 28, "bmi_initial": 22.8, "gestational_week": 20, "gestational_trimester": 2,
        "height_cm": 160.0, "initial_weight": 58.4, "weight_kg": 63.0, "weight_gain": 4.6,
        "systolic": 112.0, "diastolic": 72.0, "mean_arterial_pressure": 85.0,
        "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 0, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 2, "previous_deliveries": 2, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 0,
        "education_level": "secundaria", "residence": "urbana", "marital_status": "married"
    },
    {
        "desc": "#3 — HTA + preeclampsia previa (esperado: Cluster 1 CRITICO)",
        "age_years": 36, "bmi_initial": 27.2, "gestational_week": 28, "gestational_trimester": 3,
        "height_cm": 158.0, "initial_weight": 67.9, "weight_kg": 78.0, "weight_gain": 10.1,
        "systolic": 148.0, "diastolic": 96.0, "mean_arterial_pressure": 113.0,
        "diabetes": 0, "chronic_hypertension": 1, "previous_preeclampsia": 1,
        "family_history_hypertension": 1, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 1, "previous_deliveries": 1, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 0,
        "education_level": "superior", "residence": "urbana", "marital_status": "married"
    },
    {
        "desc": "#4 — Obesidad + diabetes pregestacional (esperado: Cluster 3)",
        "age_years": 32, "bmi_initial": 34.5, "gestational_week": 16, "gestational_trimester": 2,
        "height_cm": 155.0, "initial_weight": 82.9, "weight_kg": 89.0, "weight_gain": 6.1,
        "systolic": 125.0, "diastolic": 82.0, "mean_arterial_pressure": 96.0,
        "diabetes": 1, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 1, "family_history_heart_disease": 1,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 1, "previous_deliveries": 1, "previous_miscarriages": 0,
        "previous_cesareans": 1, "nulliparous": 0,
        "education_level": "primaria", "residence": "rural", "marital_status": "married"
    },
    {
        "desc": "#5 — Primigesta de 38 anos sin comorbilidades (esperado: Cluster 0)",
        "age_years": 38, "bmi_initial": 24.1, "gestational_week": 10, "gestational_trimester": 1,
        "height_cm": 162.0, "initial_weight": 63.3, "weight_kg": 64.5, "weight_gain": 1.2,
        "systolic": 118.0, "diastolic": 76.0, "mean_arterial_pressure": 90.0,
        "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 1, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 0, "previous_deliveries": 0, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 1,
        "education_level": "superior", "residence": "urbana", "marital_status": "married"
    },
    {
        "desc": "#6 — HTA + primigesta de 40 anos + presion critica (esperado: Cluster 1 CRITICO)",
        "age_years": 40, "bmi_initial": 29.8, "gestational_week": 32, "gestational_trimester": 3,
        "height_cm": 157.0, "initial_weight": 73.5, "weight_kg": 86.0, "weight_gain": 12.5,
        "systolic": 155.0, "diastolic": 100.0, "mean_arterial_pressure": 118.0,
        "diabetes": 0, "chronic_hypertension": 1, "previous_preeclampsia": 0,
        "family_history_hypertension": 1, "family_history_heart_disease": 1,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 0, "previous_deliveries": 0, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 1,
        "education_level": "secundaria", "residence": "rural", "marital_status": "single"
    },
    {
        "desc": "#7 — Multipara sana, 3 partos sin complicaciones (esperado: Cluster 2)",
        "age_years": 31, "bmi_initial": 23.5, "gestational_week": 24, "gestational_trimester": 2,
        "height_cm": 161.0, "initial_weight": 60.9, "weight_kg": 67.0, "weight_gain": 6.1,
        "systolic": 108.0, "diastolic": 68.0, "mean_arterial_pressure": 81.0,
        "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 0, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 3, "previous_deliveries": 3, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 0,
        "education_level": "superior", "residence": "urbana", "marital_status": "married"
    },
    {
        "desc": "#8 — Obesidad morbida + diabetes severa (esperado: Cluster 3)",
        "age_years": 34, "bmi_initial": 38.2, "gestational_week": 22, "gestational_trimester": 2,
        "height_cm": 152.0, "initial_weight": 88.3, "weight_kg": 96.0, "weight_gain": 7.7,
        "systolic": 130.0, "diastolic": 85.0, "mean_arterial_pressure": 100.0,
        "diabetes": 1, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 1, "family_history_heart_disease": 1,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 2, "previous_deliveries": 2, "previous_miscarriages": 0,
        "previous_cesareans": 2, "nulliparous": 0,
        "education_level": "primaria", "residence": "rural", "marital_status": "married"
    },
    {
        "desc": "#9 — Primigesta de 20 anos, todo normal (esperado: Cluster 0)",
        "age_years": 20, "bmi_initial": 20.1, "gestational_week": 8, "gestational_trimester": 1,
        "height_cm": 165.0, "initial_weight": 54.7, "weight_kg": 55.5, "weight_gain": 0.8,
        "systolic": 105.0, "diastolic": 65.0, "mean_arterial_pressure": 78.0,
        "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
        "family_history_hypertension": 0, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 0, "previous_deliveries": 0, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 1,
        "education_level": "secundaria", "residence": "urbana", "marital_status": "single"
    },
    {
        "desc": "#10 — Multipara con HTA controlada (caso borderline, puede ser 1 o 2)",
        "age_years": 33, "bmi_initial": 26.5, "gestational_week": 18, "gestational_trimester": 2,
        "height_cm": 159.0, "initial_weight": 67.0, "weight_kg": 71.5, "weight_gain": 4.5,
        "systolic": 138.0, "diastolic": 88.0, "mean_arterial_pressure": 105.0,
        "diabetes": 0, "chronic_hypertension": 1, "previous_preeclampsia": 0,
        "family_history_hypertension": 1, "family_history_heart_disease": 0,
        "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
        "previous_pregnancies": 1, "previous_deliveries": 1, "previous_miscarriages": 0,
        "previous_cesareans": 0, "nulliparous": 0,
        "education_level": "superior", "residence": "urbana", "marital_status": "married"
    },
]

print("=" * 72)
print("   TEST DE 10 PERFILES CLINICOS — CLASIFICADOR KNN")
print("=" * 72)

coherente = 0
for p in perfiles:
    datos = {k: v for k, v in p.items() if k != "desc"}
    df = pd.DataFrame([datos])
    df = df[feature_columns]
    X_pre = preprocessor.transform(df)
    X_pca = pca.transform(X_pre)
    cluster = int(knn.predict(X_pca)[0])
    diagnostico = diagnosis_map.get(cluster, f"Cluster {cluster}")
    print(f"\nPaciente {p['desc']}")
    print(f"  Cluster predicho : {cluster}")
    print(f"  Diagnostico      : {diagnostico}")

print("\n" + "=" * 72)
