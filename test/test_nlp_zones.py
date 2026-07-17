"""Pruebas de la extracción de zonas del cuerpo (no requiere ONNX ni es_core_news_sm)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy

import nlp_catalog
import nlp as nlp_mod


def _blank_doc(text):
    """Doc de spaCy sin modelo: solo tokenizer + sentencizer (offline, sin descargas)."""
    nlp = spacy.blank("es")
    nlp.add_pipe("sentencizer")
    return nlp(text)


def _blank_nlp():
    nlp = spacy.blank("es")
    nlp.add_pipe("sentencizer")
    return nlp


def test_catalogo_zonas_no_vacio_y_codigos_unicos():
    cat = nlp_catalog.BODY_ZONE_CATALOG
    assert len(cat) >= 10
    codes = [z.code for z in cat]
    assert len(codes) == len(set(codes))  # sin códigos duplicados


def test_cada_zona_tiene_anclas_y_label():
    for z in nlp_catalog.BODY_ZONE_CATALOG:
        assert z.label.strip()
        assert len(z.anchors) >= 1
        assert all(a.strip() for a in z.anchors)


def test_index_por_codigo_coincide():
    idx = nlp_catalog.BODY_ZONE_BY_CODE
    assert set(idx.keys()) == {z.code for z in nlp_catalog.BODY_ZONE_CATALOG}
    assert idx["CABEZA"].label == "Cabeza"


# --- Task 2: negación a nivel de módulo ---

def test_is_negated_detecta_negacion():
    text = "no me duele la cabeza"
    doc = _blank_doc(text)
    start = text.index("cabeza")
    assert nlp_mod._is_negated(doc, start) is True


def test_is_negated_sin_negacion():
    text = "me duele la cabeza"
    doc = _blank_doc(text)
    start = text.index("cabeza")
    assert nlp_mod._is_negated(doc, start) is False


def test_is_negated_terminador_corta_scope():
    text = "no tengo fiebre pero me duele la cabeza"
    doc = _blank_doc(text)
    start = text.index("cabeza")
    # "pero" reinicia el scope: la cabeza NO queda negada
    assert nlp_mod._is_negated(doc, start) is False


def test_is_negated_no_marca_sintoma_expresado_con_negacion():
    """'no puedo respirar' ES el sintoma (disnea), no una negacion de disnea.

    El NER corta el span en 'respirar', dejando 'no puedo' fuera; NegEx no debe
    interpretar ese 'no' como que la paciente niega la disnea (signo de alarma).
    """
    text = "no puedo respirar bien"
    doc = _blank_doc(text)
    start = text.index("respirar")
    assert nlp_mod._is_negated(doc, start) is False


def test_is_negated_sigue_detectando_negacion_real():
    """Regresion: una negacion de verdad ('no tengo X') debe seguir marcandose."""
    for text, palabra in [("no tengo fiebre", "fiebre"),
                          ("no tengo dolor de cabeza", "dolor"),
                          ("sin sangrado", "sangrado")]:
        doc = _blank_doc(text)
        assert nlp_mod._is_negated(doc, text.index(palabra)) is True, text


# --- Task 3: detección de zonas (Matcher difuso) ---

def test_find_zone_spans_detecta_zona_simple():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    doc = nlp("me duele la cabeza")
    zones = nlp_mod.find_zone_spans(doc, matcher)
    codes = {z["code"] for z in zones}
    assert "CABEZA" in codes
    z = next(z for z in zones if z["code"] == "CABEZA")
    assert z["raw_text"].lower() == "cabeza"
    assert z["negated"] is False
    assert z["score"] == 1.0


def test_find_zone_spans_marca_negacion():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    doc = nlp("no me duele la cabeza")
    z = next(z for z in nlp_mod.find_zone_spans(doc, matcher) if z["code"] == "CABEZA")
    assert z["negated"] is True


def test_find_zone_spans_prefiere_frase_mas_larga():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    doc = nlp("me arde la boca del estomago")
    codes = {z["code"] for z in nlp_mod.find_zone_spans(doc, matcher)}
    # "boca del estomago" (EPIGASTRIO) gana sobre "estomago" (ABDOMEN)
    assert "EPIGASTRIO" in codes
    assert "ABDOMEN" not in codes


def test_find_zone_spans_sin_zonas():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    doc = nlp("me siento muy cansada y con nauseas")
    assert nlp_mod.find_zone_spans(doc, matcher) == []


def test_find_zone_spans_tolera_typos():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    casos = {
        "me duele la caeza": "CABEZA",       # cabeza, dist 1
        "me arde la vagia": "ZONA_GENITAL",  # vagina, dist 1
        "me duele el estomgo": "ABDOMEN",    # estomago, dist 1
    }
    for text, code in casos.items():
        doc = nlp(text)
        codes = {z["code"] for z in nlp_mod.find_zone_spans(doc, matcher)}
        assert code in codes, f"'{text}' debería detectar {code}, obtuvo {codes}"


def test_find_zone_spans_palabra_corta_no_genera_falso_positivo():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    # "casa" está a 1 edición de "cara", pero "cara" es corta (< 5) → match exacto, sin falso positivo
    doc = nlp("me gusta mucho mi casa nueva")
    assert nlp_mod.find_zone_spans(doc, matcher) == []


# --- Task 4: vinculación zona↔síntoma + body_zones ---

def _sym(code, start, **extra):
    d = {"code": code, "label": code.title(), "raw_text": code, "_start": start,
         "negated": False, "score": 0.9, "alarm": False}
    d.update(extra)
    return d


def test_link_zones_adjunta_a_sintoma_misma_frase():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    text = "me duele la cabeza"
    doc = nlp(text)
    zones = nlp_mod.find_zone_spans(doc, matcher)
    symptoms = [_sym("CEFALEA", text.index("duele"))]

    body_zones = nlp_mod.link_zones(symptoms, zones, doc)

    assert [z["code"] for z in symptoms[0]["zones"]] == ["CABEZA"]
    assert symptoms[0]["zones"][0].keys() == {"code", "label", "raw_text", "negated", "score"}
    assert [z["code"] for z in body_zones] == ["CABEZA"]


def test_link_zones_suelta_va_solo_en_body_zones():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    text = "me duele la cabeza. tengo las manos raras"
    doc = nlp(text)
    zones = nlp_mod.find_zone_spans(doc, matcher)
    symptoms = [_sym("CEFALEA", text.index("duele"))]  # síntoma solo en la 1a frase

    body_zones = nlp_mod.link_zones(symptoms, zones, doc)

    assert [z["code"] for z in symptoms[0]["zones"]] == ["CABEZA"]  # manos NO se adjunta
    assert {z["code"] for z in body_zones} == {"CABEZA", "MANOS"}   # pero sí está en la lista


def test_link_zones_nearest_symptom_en_frase_con_dos():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    text = "me duele la cabeza y siento nauseas en el pecho"
    doc = nlp(text)
    zones = nlp_mod.find_zone_spans(doc, matcher)
    s_cef = _sym("CEFALEA", text.index("duele"))
    s_nau = _sym("NAUSEA_VOMITO", text.index("nauseas"))
    symptoms = [s_cef, s_nau]

    nlp_mod.link_zones(symptoms, zones, doc)

    assert [z["code"] for z in s_cef["zones"]] == ["CABEZA"]  # cabeza → cefalea (más cercano)
    assert [z["code"] for z in s_nau["zones"]] == ["PECHO"]   # pecho → nauseas (más cercano)


def test_link_zones_body_zones_dedup_por_codigo():
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    text = "me duele la cabeza. otra vez la cabeza"
    doc = nlp(text)
    zones = nlp_mod.find_zone_spans(doc, matcher)
    body_zones = nlp_mod.link_zones([], zones, doc)
    assert [z["code"] for z in body_zones] == ["CABEZA"]  # una sola entrada


def test_link_zones_contencion_evita_mal_vinculacion():
    """Frase corrida con dos dolores y spans LARGOS de síntoma (incluyen su zona).

    La vinculación debe hacerse por contención/borde, no por inicio del span, para no
    pegar la 'cabeza' al dolor de estómago.
    """
    nlp = _blank_nlp()
    matcher = nlp_mod.build_zone_matcher(nlp)
    text = "me duele la cabeza y me duele el estomago"
    doc = nlp(text)
    zones = nlp_mod.find_zone_spans(doc, matcher)
    # cada síntoma cubre TODA su cláusula (incluye la palabra de zona)
    s_cef = _sym("CEFALEA", 0, _end=text.index(" y"))
    s_abd = _sym("DOLOR_ABDOMINAL", text.index("y ") + 2, _end=len(text))

    nlp_mod.link_zones([s_cef, s_abd], zones, doc)

    assert [z["code"] for z in s_cef["zones"]] == ["CABEZA"]        # cabeza → cefalea (la contiene)
    assert [z["code"] for z in s_abd["zones"]] == ["ABDOMEN"]       # estomago → dolor abdominal


# --- Task 5: extract() integra zonas ---

class _FakePipeline(nlp_mod.SymptomExtractionPipeline):
    """Evita cargar ONNX: stubbea init, NER y normalización; usa spaCy blank real."""
    def __init__(self, ner_spans):
        self._fake_ner = ner_spans
        self._nlp = _blank_nlp()
        self._zone_matcher = nlp_mod.build_zone_matcher(self._nlp)

    def _ner_spans(self, text):
        return self._fake_ner

    def _normalize(self, raw_text):
        # mapa mínimo raw_text -> (code, sim) para el test
        m = {"me duele la cabeza": ("CEFALEA", 0.9)}
        return m.get(raw_text)


def test_extract_devuelve_symptoms_y_body_zones():
    text = "me duele la cabeza"
    ner = [{"start": 0, "end": len(text), "raw_text": text, "score": 0.88}]
    pipe = _FakePipeline(ner)

    out = pipe.extract(text)

    assert set(out.keys()) == {"symptoms", "body_zones"}
    assert out["symptoms"][0]["code"] == "CEFALEA"
    assert [z["code"] for z in out["symptoms"][0]["zones"]] == ["CABEZA"]
    assert [z["code"] for z in out["body_zones"]] == ["CABEZA"]
    # el campo interno _start no se filtra al exterior
    assert "_start" not in out["symptoms"][0]


def test_extract_texto_vacio():
    pipe = _FakePipeline([])
    assert pipe.extract("   ") == {"symptoms": [], "body_zones": []}


# --- Task 6: contrato Pydantic de la API ---

def test_response_model_acepta_zonas_y_body_zones():
    import main as main_mod
    payload = {
        "symptoms": [{
            "code": "CEFALEA", "label": "Cefalea", "raw_text": "me duele la cabeza",
            "negated": False, "score": 0.83, "alarm": True,
            "zones": [{"code": "CABEZA", "label": "Cabeza", "raw_text": "cabeza",
                       "negated": False, "score": 1.0}],
        }],
        "body_zones": [{"code": "CABEZA", "label": "Cabeza", "raw_text": "cabeza",
                        "negated": False, "score": 1.0}],
        "model_version": "symptemist-onnx-int8",
    }
    resp = main_mod.SymptomExtractionResponse(**payload)
    assert resp.symptoms[0].zones[0].code == "CABEZA"
    assert resp.body_zones[0].label == "Cabeza"


def test_symptom_sin_zonas_default_lista_vacia():
    import main as main_mod
    s = main_mod.ExtractedSymptom(
        code="MAREO", label="Mareo", raw_text="mareo", negated=False, score=0.7, alarm=False)
    assert s.zones == []


# --- Fix problema 1: descartar spans que son puro stopword (el NER etiqueta "me" -> Edema) ---

def test_is_content_span_descarta_stopwords():
    for w in ("me", "la", "el", "de", "y"):
        assert nlp_mod._is_content_span(w) is False, w


def test_is_content_span_acepta_contenido():
    assert nlp_mod._is_content_span("cabeza") is True
    assert nlp_mod._is_content_span("dolor de cabeza") is True
    assert nlp_mod._is_content_span("me duele") is True   # tiene al menos una palabra de contenido


def test_is_content_span_texto_vacio_o_puntuacion():
    assert nlp_mod._is_content_span("") is False
    assert nlp_mod._is_content_span("...") is False


# --- Stopwords configurables (listas editables en nlp_catalog.py) ---

def test_stopwords_extra_agrega_palabra():
    # "chekeo" no es stopword de spaCy; al agregarla, un span que sea solo esa palabra se descarta
    assert nlp_mod._is_content_span("chekeo") is True
    nlp_catalog.EXTRA_STOPWORDS.add("chekeo")
    try:
        assert nlp_mod._is_content_span("chekeo") is False
    finally:
        nlp_catalog.EXTRA_STOPWORDS.discard("chekeo")
    assert nlp_mod._is_content_span("chekeo") is True  # revertido


def test_stopwords_keep_protege_palabra():
    # "algo" ES stopword de spaCy -> normalmente un span "algo" se descarta
    assert nlp_mod._is_content_span("algo") is False
    nlp_catalog.KEEP_WORDS.add("algo")
    try:
        assert nlp_mod._is_content_span("algo") is True   # protegida, ya no cuenta como vacía
    finally:
        nlp_catalog.KEEP_WORDS.discard("algo")
    assert nlp_mod._is_content_span("algo") is False  # revertido
