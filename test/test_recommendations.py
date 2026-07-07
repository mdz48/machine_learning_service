"""Pruebas de las recomendaciones clínicas SOMANZ (no requiere BD ni modelo)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import recommendations


def _aspirin(recs):
    return next(i for i in recs["items"] if i["intervencion"] == "Aspirina")


def test_solo_cluster_1_tiene_recomendaciones():
    assert recommendations.get_recommendations(1, 12) is not None
    for c in (0, 2, 3):
        assert recommendations.get_recommendations(c, 12) is None


def test_aspirina_ventana_inicio_antes_de_16():
    recs = recommendations.get_recommendations(1, 12)
    asp = _aspirin(recs)
    assert asp["aplicable_ahora"] is True
    assert asp["grade"] == "1B"


def test_aspirina_ventana_pasada_entre_16_y_34():
    recs = recommendations.get_recommendations(1, 20)
    asp = _aspirin(recs)
    assert asp["aplicable_ahora"] is False
    assert "antes de la semana 16" in asp["recomendacion"]


def test_aspirina_cese_desde_semana_34():
    recs = recommendations.get_recommendations(1, 36)
    asp = _aspirin(recs)
    assert asp["aplicable_ahora"] is False
    assert "cese" in asp["recomendacion"].lower()
    assert asp["grade"] == "2B"


def test_incluye_calcio_y_fuente():
    recs = recommendations.get_recommendations(1, 12)
    assert any(i["intervencion"] == "Calcio oral" for i in recs["items"])
    assert "SOMANZ" in recs["fuente"]
    assert recs["descargo"]  # descargo no vacío


def test_serializa_json_estricto():
    recs = recommendations.get_recommendations(1, 30)
    json.dumps(recs, allow_nan=False)
