"""Pruebas de la API. La sesión de BD viene del fixture `client` (test/conftest.py),
que envuelve cada test en una transacción con rollback: no ensucia la BD real."""

valid_payload_1 = {
    "age_years": 23, "bmi_initial": 21.5, "gestational_week": 14, "gestational_trimester": 2,
    "height_cm": 163.0, "initial_weight": 57.0, "weight_kg": 60.0, "weight_gain": 3.0,
    "systolic": 110.0, "diastolic": 70.0, "mean_arterial_pressure": 83.0,
    "diabetes": 0, "chronic_hypertension": 0, "previous_preeclampsia": 0,
    "family_history_hypertension": 0, "family_history_heart_disease": 0,
    "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
    "previous_pregnancies": 0, "previous_deliveries": 0, "previous_miscarriages": 0,
    "previous_cesareans": 0, "nulliparous": 1,
    "education_level": "superior", "residence": "urbana", "marital_status": "single"
}

valid_payload_2 = {
    "age_years": 36, "bmi_initial": 27.2, "gestational_week": 28, "gestational_trimester": 3,
    "height_cm": 158.0, "initial_weight": 67.9, "weight_kg": 78.0, "weight_gain": 10.1,
    "systolic": 148.0, "diastolic": 96.0, "mean_arterial_pressure": 113.0,
    "diabetes": 0, "chronic_hypertension": 1, "previous_preeclampsia": 1,
    "family_history_hypertension": 1, "family_history_heart_disease": 0,
    "chronic_kidney_disease": 0, "multiple_pregnancy": 0, "active_smoking": 0,
    "previous_pregnancies": 1, "previous_deliveries": 1, "previous_miscarriages": 0,
    "previous_cesareans": 0, "nulliparous": 0,
    "education_level": "superior", "residence": "urbana", "marital_status": "married"
}

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "ML Service is running"}

def test_predict_success_1(client):
    response = client.post("/predict", json=valid_payload_1)
    assert response.status_code == 200
    data = response.json()
    assert "risk_cluster" in data
    assert "diagnosis" in data

def test_predict_success_2(client):
    response = client.post("/predict", json=valid_payload_2)
    assert response.status_code == 200
    data = response.json()
    assert "risk_cluster" in data
    assert "diagnosis" in data

def test_predict_missing_fields(client):
    # Payload sin la mayoría de los campos (sólo uno) -> Falla adrede
    invalid_payload = {"age_years": 25}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422 # Unprocessable Entity (ValidationError)

def test_predict_wrong_data_type(client):
    # Payload con tipo incorrecto -> Falla adrede
    invalid_payload = valid_payload_1.copy()
    invalid_payload["age_years"] = "no_es_un_numero"
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_get_history(client):
    # Aseguramos de que haya algo insertando una inferencia
    client.post("/predict", json=valid_payload_1)
    
    # Verificamos que el historial se obtenga correctamente
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
