"""Recomendaciones clínicas para el perfil de Alto Riesgo Hipertensivo / Preeclampsia (cluster 1).

Basadas en la guía **SOMANZ** (prevención de preeclampsia, Parte 3A). Contenido estático,
codificado tal cual la tabla provista (con su calificación GRADE). NO es una prescripción:
son recomendaciones generales de guía para el perfil, para consideración del médico tratante.

Las recomendaciones de aspirina dependen de la edad gestacional (ventana de inicio < sem 16;
cese entre sem 34 y el parto), por eso `get_recommendations` recibe `gestational_week`.
"""

# Cluster al que aplican estas recomendaciones (Alto Riesgo Hipertensivo / Preeclampsia).
_HYPERTENSIVE_CLUSTER = 1

SOMANZ_SOURCE = "SOMANZ – Prevención de preeclampsia (Parte 3A)"

DISCLAIMER = (
    "Recomendaciones generales de guía clínica para este perfil de riesgo; no constituyen "
    "una prescripción. La decisión final corresponde al médico tratante."
)

# Umbrales de la ventana gestacional para la aspirina (semanas).
_ASPIRIN_START_MAX_WEEK = 16   # inicio óptimo: antes de la semana 16
_ASPIRIN_CESSATION_WEEK = 34   # cese: entre la semana 34 y el parto


def _aspirin_item(gestational_week: int) -> dict:
    """Ítem de aspirina ajustado a la ventana gestacional (SOMANZ 3A.1.1–3A.1.4)."""
    if gestational_week < _ASPIRIN_START_MAX_WEEK:
        return {
            "intervencion": "Aspirina",
            "recomendacion": "Iniciar aspirina 150 mg/día (idealmente nocturna) en mujeres con alto riesgo de preeclampsia, antes de la semana 16 de gestación.",
            "grade": "1B",
            "aplicable_ahora": True,
            "nota": "Ventana de inicio óptima (antes de la semana 16).",
        }
    if gestational_week < _ASPIRIN_CESSATION_WEEK:
        return {
            "intervencion": "Aspirina",
            "recomendacion": "El inicio de aspirina se recomienda antes de la semana 16; en este momento la ventana de inicio ya pasó.",
            "grade": "1B",
            "aplicable_ahora": False,
            "nota": "Evaluar de forma individualizada la continuación si la aspirina ya fue iniciada previamente.",
        }
    return {
        "intervencion": "Aspirina",
        "recomendacion": "Considerar el cese de aspirina entre la semana 34 y el parto.",
        "grade": "2B",
        "aplicable_ahora": False,
        "nota": "El momento exacto de cese se basa en juicio clínico individualizado y decisión compartida con la paciente.",
    }


def _calcium_item() -> dict:
    """Ítem de calcio (SOMANZ 3A.2.1–3A.2.2); no depende de la edad gestacional."""
    return {
        "intervencion": "Calcio oral",
        "recomendacion": "En mujeres con baja ingesta dietética de calcio (< 1 g/día), se recomienda suplementación de calcio.",
        "grade": "1C",
        "aplicable_ahora": True,
        "nota": "Evaluar la ingesta dietética de calcio antes de recomendar la suplementación (punto de práctica).",
    }


# Intervenciones no recomendadas por evidencia insuficiente (SOMANZ 3A.3, 3A.4).
_NOT_RECOMMENDED = [
    {"intervencion": "Omega-3 (LCPUFA)", "grade": "2B", "nota": "No recomendado hasta contar con más datos."},
    {"intervencion": "Suplementación con ajo", "grade": "2D", "nota": "No recomendado hasta contar con más datos."},
]


def get_recommendations(cluster: int, gestational_week: int):
    """Recomendaciones SOMANZ para el perfil hipertensivo, o None si el cluster no aplica.

    Solo el cluster 1 (Alto Riesgo Hipertensivo / Preeclampsia) tiene recomendaciones; para
    los demás perfiles devuelve None (el campo se omite en la respuesta).
    """
    if cluster != _HYPERTENSIVE_CLUSTER:
        return None

    return {
        "fuente": SOMANZ_SOURCE,
        "descargo": DISCLAIMER,
        "aplica_a_perfil": "Alto Riesgo Hipertensivo / Preeclampsia",
        "items": [
            _aspirin_item(gestational_week),
            _calcium_item(),
        ],
        "no_recomendados": _NOT_RECOMMENDED,
    }
