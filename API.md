# Documentación de la API: Microservicio de Machine Learning

Este microservicio se encarga de realizar predicciones sobre el nivel de riesgo de salud prenatal de pacientes basado en sus datos clínicos y biométricos. 

## Sobre el Modelo de Machine Learning

El motor predictivo está construido utilizando **scikit-learn** y se compone de un pipeline de tres etapas principales:

1. **Preprocesamiento (`preprocessor.pkl`)**: 
   - **Datos numéricos**: Se imputan valores nulos usando la mediana y se escalan mediante `StandardScaler`.
   - **Datos categóricos**: Se codifican utilizando `OneHotEncoder`.
2. **Reducción de Dimensionalidad (`pca_5.pkl`)**: Utiliza Análisis de Componentes Principales (PCA) para reducir la alta dimensionalidad de los features médicos a 5 componentes principales que capturan la mayor varianza de los datos.
3. **Clasificación Predictiva (`knn_model.pkl`)**: Utiliza el algoritmo *K-Nearest Neighbors* (KNN, `k=15`, distancia euclidiana) para asignar a la paciente a uno de los 4 clústeres de riesgo (inicialmente descubiertos mediante agrupamiento K-Means, `K=4`).

> **Versión del modelo:** `v2.0.0` — PCA 5 componentes, KNN `k=15`. Ver `models/model_metadata.json` para los hiperparámetros y métricas exactas.

### Perfiles de Riesgo (Clústeres)
- **Cluster 0**: Primigestas Sanas (Riesgo Bajo-Medio)
- **Cluster 1**: Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)
- **Cluster 2**: Multíparas Sanas (Riesgo Bajo)
- **Cluster 3**: Riesgo Metabólico / Obesidad (Riesgo Alto)

---

## Endpoints

### 1. Health Check
Verifica que el servicio y la API estén funcionando correctamente.

- **Método**: `GET`
- **Ruta**: `/`
- **Respuesta Exitosa (200 OK)**:
```json
{
  "status": "ok",
  "message": "ML Service is running"
}
```

---

### 2. Predicción de Riesgo
Recibe los datos médicos de una paciente, ejecuta el pipeline de Machine Learning y devuelve el perfil de riesgo asignado. La inferencia se guarda automáticamente en la base de datos PostgreSQL.

- **Método**: `POST`
- **Ruta**: `/predict`
- **Headers**: `Content-Type: application/json`

#### Body (JSON)
*Nota importante sobre las variables categóricas: Deben coincidir exactamente (en minúsculas/inglés) con las listadas a continuación.*

```json
{
  "age_years": 28,
  "bmi_initial": 24.5,
  "gestational_week": 20,
  "gestational_trimester": 2,
  "height_cm": 165,
  "initial_weight": 65,
  "weight_kg": 70,
  "weight_gain": 5,
  "systolic": 110,
  "diastolic": 70,
  "mean_arterial_pressure": 83.33,
  "diabetes": 0,                      // 1 = Sí, 0 = No
  "chronic_hypertension": 0,          
  "previous_preeclampsia": 0,
  "family_history_hypertension": 0,
  "family_history_heart_disease": 0,
  "chronic_kidney_disease": 0,
  "multiple_pregnancy": 0,
  "active_smoking": 0,
  "previous_pregnancies": 0,
  "previous_deliveries": 0,
  "previous_miscarriages": 0,
  "previous_cesareans": 0,
  "nulliparous": 1,                   // 1 = Sí (Primeriza), 0 = No
  "education_level": "superior",      // "primaria", "secundaria", "superior"
  "residence": "urbana",              // "rural", "urbana"
  "marital_status": "married"         // "married", "single"
}
```

#### Respuesta Exitosa (200 OK)
Además del clúster asignado, la respuesta incluye una **capa de explicabilidad (XAI)** pensada para el ginecólogo: por qué la paciente cayó en ese perfil.

```json
{
  "risk_cluster": 1,
  "diagnosis": "Alto Riesgo Hipertensivo / Preeclampsia (Riesgo Crítico)",
  "interpretation": "Paciente con alto riesgo hipertensivo/preeclampsia. Monitoreo estricto de presión arterial recomendado.",
  "model_version": "v2.0.0",
  "inference_time_ms": 34.99,
  "afinidad": {
    "Primigestas Sanas": 0.0,
    "Alto Riesgo Hipertensivo": 80.0,
    "Multiparas Sanas": 0.0,
    "Riesgo Metabolico": 20.0
  },
  "caso_limitrofe": false,
  "factores_determinantes": [
    { "variable": "chronic_hypertension", "etiqueta": "Hipertensión crónica", "valor_paciente": 1.0, "promedio_perfil": 0.55, "score": 4.18 },
    { "variable": "previous_preeclampsia", "etiqueta": "Preeclampsia previa", "valor_paciente": 1.0, "promedio_perfil": 0.23, "score": 3.88 },
    { "variable": "mean_arterial_pressure", "etiqueta": "Presión arterial media", "valor_paciente": 116.0, "promedio_perfil": 111.3, "score": 3.18 }
  ],
  "pacientes_similares": [
    { "age_years": 46.0, "systolic": 167.0, "diastolic": 95.0, "bmi_initial": 33.3, "perfil": "Alto Riesgo Hipertensivo" },
    { "age_years": 39.0, "systolic": 147.0, "diastolic": 88.0, "bmi_initial": 36.2, "perfil": "Alto Riesgo Hipertensivo" },
    { "age_years": 34.0, "systolic": 141.0, "diastolic": 96.0, "bmi_initial": 30.0, "perfil": "Riesgo Metabolico" }
  ],
  "explicacion": "La paciente fue asignada al perfil «Alto Riesgo Hipertensivo» debido principalmente a: hipertensión crónica, preeclampsia previa, presión arterial media. Estos rasgos coinciden con el patrón clínico característico de este grupo.",
  "recomendaciones": {
    "fuente": "SOMANZ – Prevención de preeclampsia (Parte 3A)",
    "descargo": "Recomendaciones generales de guía clínica para este perfil de riesgo; no constituyen una prescripción. La decisión final corresponde al médico tratante.",
    "aplica_a_perfil": "Alto Riesgo Hipertensivo / Preeclampsia",
    "items": [
      { "intervencion": "Aspirina", "recomendacion": "El inicio de aspirina se recomienda antes de la semana 16; en este momento la ventana de inicio ya pasó.", "grade": "1B", "aplicable_ahora": false, "nota": "Evaluar de forma individualizada la continuación si la aspirina ya fue iniciada previamente." },
      { "intervencion": "Calcio oral", "recomendacion": "En mujeres con baja ingesta dietética de calcio (< 1 g/día), se recomienda suplementación de calcio.", "grade": "1C", "aplicable_ahora": true, "nota": "Evaluar la ingesta dietética de calcio antes de recomendar la suplementación (punto de práctica)." }
    ],
    "no_recomendados": [
      { "intervencion": "Omega-3 (LCPUFA)", "grade": "2B", "nota": "No recomendado hasta contar con más datos." },
      { "intervencion": "Suplementación con ajo", "grade": "2D", "nota": "No recomendado hasta contar con más datos." }
    ]
  }
}
```

> El campo `recomendaciones` **solo aparece para el perfil de Alto Riesgo Hipertensivo (`risk_cluster: 1`)**. Para los demás perfiles se omite.

#### Descripción de los campos

| Campo | Tipo | Descripción |
|---|---|---|
| `risk_cluster` | int (0-3) | Clúster de riesgo asignado. |
| `diagnosis` | str | Nombre del perfil clínico. |
| `interpretation` | str | Recomendación estática del perfil (+ notas de atípicos si los hay). |
| `model_version` | str | Versión del modelo activo (`v2.0.0`). |
| `inference_time_ms` | float | Tiempo de inferencia en milisegundos. |
| `afinidad` | objeto `{perfil: %}` | **Similitud** con cada perfil: fracción de las 15 pacientes más parecidas que caen en cada grupo (suma 100). **No es una probabilidad clínica calibrada**, es cercanía a los perfiles. |
| `caso_limitrofe` | bool | `true` si la afinidad máxima < 70% → caso ambiguo, se sugiere revisión más cuidadosa. |
| `factores_determinantes` | lista (top-5) | Variables que **definen** el perfil y que la paciente exhibe. Cada una: `variable`, `etiqueta` (legible), `valor_paciente`, `promedio_perfil`, `score` (mayor = más determinante). |
| `pacientes_similares` | lista (3) | Pacientes históricas más parecidas (vecinas del KNN): `age_years`, `systolic`, `diastolic`, `bmi_initial`, `perfil`. *Nota: son datos sintéticos; en un despliegue con datos reales estas pacientes deberían anonimizarse.* |
| `explicacion` | str | Explicación en lenguaje natural, armada con los factores determinantes. |
| `recomendaciones` | objeto \| ausente | **Solo para `risk_cluster: 1`.** Recomendaciones de guía clínica (SOMANZ) para el perfil hipertensivo. Ver detalle abajo. |

#### Campo `recomendaciones` (solo perfil hipertensivo)

Recomendaciones de la guía **SOMANZ** (prevención de preeclampsia). **No son una prescripción**: son guía general para el perfil, para consideración del médico tratante (ver `descargo`). Cada ítem incluye su calificación **GRADE** (`1B`, `1C`, `2B`, `2D`...).

| Sub-campo | Descripción |
|---|---|
| `fuente` | Referencia de la guía. |
| `descargo` | Aviso de que no es prescripción; decide el médico tratante. |
| `aplica_a_perfil` | Perfil de riesgo al que aplican. |
| `items` | Lista de intervenciones recomendadas: `intervencion`, `recomendacion`, `grade`, `aplicable_ahora` (bool), `nota`. |
| `no_recomendados` | Intervenciones no recomendadas por evidencia insuficiente (omega-3, ajo). |

**Aspirina sensible a la edad gestacional** (`aplicable_ahora` cambia según `gestational_week`):
- `< 16` semanas → iniciar aspirina 150 mg/día (`aplicable_ahora: true`, GRADE 1B).
- `16–33` semanas → ventana de inicio pasada (`aplicable_ahora: false`); evaluar continuación si ya se inició.
- `>= 34` semanas → considerar cese entre la semana 34 y el parto (GRADE 2B).

---

### 3. Historial de Inferencias
Consulta el registro histórico de todas las predicciones realizadas por el modelo.

- **Método**: `GET`
- **Ruta**: `/history`
- **Respuesta Exitosa (200 OK)**:
```json
[
  {
    "id": 1,
    "timestamp": "2026-06-25T21:30:00.000000",
    "input_data": {
       // ... JSON payload de entrada
    },
    "prediction_result": {
      // Mismo objeto que devuelve /predict (incluye afinidad, factores_determinantes,
      // pacientes_similares, caso_limitrofe y explicacion).
      "risk_cluster": 0,
      "diagnosis": "Primigestas Sanas (Riesgo Bajo-Medio)"
    }
  }
]
```

---

### 4. Extracción de Síntomas y Zonas del Cuerpo (NLP)
Recibe texto libre en español (lo que la paciente/médico escribe en la bitácora) y devuelve los
**síntomas** y las **zonas del cuerpo** detectados. Tolera errores de escritura de la paciente
(las zonas se detectan con coincidencia difusa por distancia de edición: `caeza` → cabeza).

- **Método**: `POST`
- **Ruta**: `/nlp/extract-symptoms`
- **Headers**: `Content-Type: application/json`

#### Body (JSON)
```json
{ "text": "me duele la caeza y tengo los pies hinchados" }
```

#### Respuesta Exitosa (200 OK)
Los campos `symptoms[].zones` y `body_zones` son **aditivos**: un consumidor que solo lea
`symptoms[]` sigue funcionando.

```json
{
  "symptoms": [
    {
      "code": "CEFALEA",
      "label": "Cefalea",
      "raw_text": "me duele la caeza",
      "negated": false,
      "score": 0.83,
      "alarm": true,
      "zones": [
        { "code": "CABEZA", "label": "Cabeza", "raw_text": "caeza", "negated": false, "score": 1.0 }
      ]
    }
  ],
  "body_zones": [
    { "code": "CABEZA", "label": "Cabeza", "raw_text": "caeza", "negated": false, "score": 1.0 }
  ],
  "model_version": "symptemist-onnx-int8"
}
```

#### Descripción de los campos

| Campo | Tipo | Descripción |
|---|---|---|
| `symptoms` | lista | Síntomas detectados. Cada uno mantiene sus campos originales + `zones`. |
| `symptoms[].negated` | bool | `true` si la paciente lo **niega** ("no me duele..."). No tratar como presente. |
| `symptoms[].alarm` | bool | `true` = signo de alarma obstétrica. |
| `symptoms[].zones` | lista | Zonas del cuerpo vinculadas a **ese** síntoma (misma frase). Puede ir vacía. |
| `body_zones` | lista | **Todas** las zonas detectadas, incluidas las que no se ligaron a ningún síntoma. |
| *objeto zona* | | `{ code, label, raw_text, negated, score }`. `score = 1.0` (coincidencia por gazetteer). |
| `model_version` | str | Versión del modelo NER de síntomas. |

> Las zonas son **opcionales**: puede haber síntomas sin zonas y zonas sin síntoma. La extracción de
> zonas no usa un modelo nuevo (gazetteer difuso con spaCy); no afecta el presupuesto de memoria.

#### Respuestas de error
- **503 Service Unavailable**: el runtime NLP no está disponible (los artefactos ONNX aún no se
  exportaron con `scripts/export_nlp_models.py`). El backend debe **degradar sin bloquear** el flujo
  del paciente (permitir captura manual de síntomas).
- **500 Internal Server Error**: error procesando el texto.

---
> **Documentación Interactiva Automática**
> Al estar construida sobre FastAPI, puedes probar todos los endpoints y ver los esquemas dinámicamente accediendo a la ruta `/docs` (Swagger UI) o `/redoc` cuando el servidor esté corriendo (por defecto en `http://localhost:8000/docs`).

# El notebook dentro de la carpeta /cluster NO funcionara con este .venv,  solo fue traido a este repositorio para la entrega de minería de datos. Se recomienda usar Jupyter Notebook o Google Colab para ejecutarlo, si se desea volver a entrenar o analizar los datos.
