"""Recomendaciones clínicas para el perfil de Alto Riesgo Hipertensivo / Preeclampsia (cluster 1)."""

_HYPERTENSIVE_CLUSTER = 1

SOMANZ_SOURCE = "SOMANZ – Prevención de preeclampsia (Parte 3A)"

DISCLAIMER = (
    "Recomendaciones generales de guía clínica para este perfil de riesgo; no constituyen "
    "una prescripción. La decisión final corresponde al médico tratante."
)

_ASPIRIN_START_MAX_WEEK = 16
_ASPIRIN_CESSATION_WEEK = 34


def _aspirin_item(gestational_week: int) -> dict:
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
    return {
        "intervencion": "Calcio oral",
        "recomendacion": "En mujeres con baja ingesta dietética de calcio (< 1 g/día), se recomienda suplementación de calcio.",
        "grade": "1C",
        "aplicable_ahora": True,
        "nota": "Evaluar la ingesta dietética de calcio antes de recomendar la suplementación (punto de práctica).",
    }


_NOT_RECOMMENDED = [
    {"intervencion": "Omega-3 (LCPUFA)", "grade": "2B", "nota": "No recomendado hasta contar con más datos."},
    {"intervencion": "Suplementación con ajo", "grade": "2D", "nota": "No recomendado hasta contar con más datos."},
]


def get_recommendations(cluster: int, gestational_week: int):
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
