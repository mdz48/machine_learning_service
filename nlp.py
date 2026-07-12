"""Pipeline NLP para extraccion de sintomas de texto libre (RF-29/30/31).

Tres etapas, todas ejecutadas sobre CPU con modelos ONNX cuantizados (int8) para
caber en el presupuesto de RAM del servidor compartido (~600-800 MB, sin PyTorch
en memoria):

  1. NER clinico  -> transformer fine-tuned en corpus de sintomas en espanol,
                     exportado a ONNX. Detecta los spans que mencionan sintomas.
  2. Negacion     -> spaCy (segmentacion + tokens) con reglas estilo NegEx en
                     espanol. Distingue "tuve dolor de cabeza" de "no tuve dolor".
  3. Normalizacion-> embeddings (MiniLM multilingue en ONNX). Cada span se mapea
                     al concepto del catalogo con mayor similitud coseno. No es un
                     diccionario: tolera sinonimos y variantes no listadas.

El runtime solo necesita onnxruntime + transformers(tokenizer) + spacy + numpy.
Los artefactos .onnx se generan una sola vez con scripts/export_nlp_models.py.

Carga perezosa: los modelos se cargan en el primer uso y se reutilizan como
singleton. Si los artefactos no existen, get_pipeline() lanza RuntimeError y el
endpoint responde 503 (la escritura de la bitacora en el backend nunca se bloquea).
"""

import json
import os
import re
import threading
from typing import List, Optional

import numpy as np

from nlp_catalog import (CATALOG, CATALOG_BY_CODE, BODY_ZONE_CATALOG,
                         BODY_ZONE_BY_CODE, EXTRA_STOPWORDS, KEEP_WORDS)

_MODELS_DIR = os.getenv("NLP_MODELS_DIR", "models/nlp")
_NER_DIR = os.path.join(_MODELS_DIR, "ner")
_EMBED_DIR = os.path.join(_MODELS_DIR, "embed")
_SPACY_MODEL = os.getenv("NLP_SPACY_MODEL", "es_core_news_sm")
_SIM_THRESHOLD = float(os.getenv("NLP_SIM_THRESHOLD", "0.55"))
_NER_SCORE_MIN = float(os.getenv("NLP_NER_SCORE_MIN", "0.50"))
_NUM_THREADS = int(os.getenv("NLP_NUM_THREADS", "2"))

# Cues de negacion pre-puestas (NegEx en espanol). El scope se corta en signos de
# puntuacion y en conjunciones adversativas.
_NEG_CUES = {
    "no", "sin", "ni", "ningun", "ninguna", "ningunos", "ningunas",
    "nunca", "jamas", "tampoco", "niega", "negativo", "negativa", "ausencia", "descarta",
}
_NEG_TERMINATORS = {"pero", "aunque", "sino", "salvo", "excepto"}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


_ES_STOP_BASE = None


def _stopwords() -> set:
    """Conjunto efectivo = (stopwords de spaCy es | EXTRA_STOPWORDS) - KEEP_WORDS.

    La base de spaCy se cachea (import perezoso, para no cargar spaCy al importar el modulo);
    EXTRA/KEEP se aplican en cada llamada para que editar esas listas en nlp_catalog.py surta
    efecto sin reiniciar.
    """
    global _ES_STOP_BASE
    if _ES_STOP_BASE is None:
        from spacy.lang.es.stop_words import STOP_WORDS
        _ES_STOP_BASE = {w.lower() for w in STOP_WORDS}
    return (_ES_STOP_BASE | {w.lower() for w in EXTRA_STOPWORDS}) - {w.lower() for w in KEEP_WORDS}


def _is_content_span(text: str) -> bool:
    """False si el span del NER es puro stopword/puntuacion.

    El NER a veces etiqueta pronombres sueltos como 'me' (que la normalizacion mapea
    debilmente a 'Edema'). Un sintoma real nunca es solo palabras vacias, asi que esos
    spans se descartan. Las stopwords son configurables (ver nlp_catalog.py).
    """
    stops = _stopwords()
    toks = re.findall(r"\w+", text.lower())
    return bool(toks) and not all(t in stops for t in toks)


def _is_negated(doc, span_start: int) -> bool:
    """True si hay una cue de negacion no cancelada antes del span, en su misma frase."""
    for sent in doc.sents:
        if sent.start_char <= span_start < sent.end_char:
            negated = False
            for tok in sent:
                if tok.idx >= span_start:
                    break  # ya pasamos el inicio del span
                low = tok.text.lower()
                if low in _NEG_TERMINATORS or tok.is_punct:
                    negated = False  # reinicia scope
                elif low in _NEG_CUES:
                    negated = True
            return negated
    return False


_ZONE_SCORE = 1.0  # coincidencia por gazetteer: no hay probabilidad


def _fuzzy_attr(token_text: str):
    """Predicado de atributo LOWER con tolerancia por longitud.

    Palabras cortas (< 5) van exactas para no generar falsos positivos ("cara"/"casa").
    5-7 letras toleran 1 edicion; >= 8 toleran 2. Cubre typos por distancia de edicion
    (incluye tildes faltantes: "estomago"/"estomago" son distancia 1).
    """
    n = len(token_text)
    if n >= 8:
        return {"FUZZY2": token_text}
    if n >= 5:
        return {"FUZZY1": token_text}
    return token_text  # exacto


def _zone_token_patterns(anchor: str) -> List[dict]:
    """Convierte una frase-ancla en un patron de tokens para el Matcher."""
    return [{"LOWER": _fuzzy_attr(tok)} for tok in anchor.lower().split()]


def build_zone_matcher(nlp):
    """Matcher difuso: un codigo de zona por conjunto de patrones-ancla."""
    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    for zone in BODY_ZONE_CATALOG:
        patterns = [_zone_token_patterns(a) for a in zone.anchors]
        matcher.add(zone.code, patterns)
    return matcher


def find_zone_spans(doc, matcher) -> List[dict]:
    """Detecta zonas en el doc; resuelve solapes (gana el span mas largo)."""
    from spacy.util import filter_spans

    by_span = {}  # (start_tok, end_tok) -> code, para recuperar tras filter_spans
    spans = []
    for match_id, start_tok, end_tok in matcher(doc):
        code = doc.vocab.strings[match_id]
        by_span[(start_tok, end_tok)] = code
        spans.append(doc[start_tok:end_tok])

    out = []
    for span in filter_spans(spans):
        code = by_span.get((span.start, span.end))
        if code is None:
            continue
        zone = BODY_ZONE_BY_CODE[code]
        out.append({
            "code": code,
            "label": zone.label,
            "raw_text": span.text,
            "start": span.start_char,
            "end": span.end_char,
            "negated": _is_negated(doc, span.start_char),
            "score": _ZONE_SCORE,
        })
    return out


def _public_zone(z: dict) -> dict:
    return {k: z[k] for k in ("code", "label", "raw_text", "negated", "score")}


def _sentence_bounds(doc, char_pos: int):
    for sent in doc.sents:
        if sent.start_char <= char_pos < sent.end_char:
            return sent.start_char, sent.end_char
    return None


def _span_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Distancia entre dos intervalos [a0,a1) y [b0,b1). 0 si se solapan o uno contiene al otro."""
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return 0  # solapamiento / contencion


def link_zones(symptoms: List[dict], zone_spans: List[dict], doc) -> List[dict]:
    """Adjunta cada zona al sintoma de su misma frase por contencion/cercania y arma body_zones.

    La cercania se mide borde-a-borde (no por inicio del span): si el span del sintoma
    contiene a la zona, la distancia es 0 y gana. Asi "me duele la cabeza y me duele el
    estomago" (una frase corrida, spans largos) no pega la 'cabeza' al dolor equivocado.

    Muta cada sintoma anadiendole la clave "zones". Devuelve body_zones deduplicada por code.
    """
    for s in symptoms:
        s.setdefault("zones", [])

    for z in zone_spans:
        bounds = _sentence_bounds(doc, z["start"])
        if bounds is None:
            continue
        s0, s1 = bounds
        # sintomas cuyo inicio cae en la misma frase que la zona
        candidatos = [s for s in symptoms if s0 <= s.get("_start", -1) < s1]
        if candidatos:
            nearest = min(
                candidatos,
                key=lambda s: _span_gap(s["_start"], s.get("_end", s["_start"]),
                                        z["start"], z["end"]),
            )
            if z["code"] not in {zz["code"] for zz in nearest["zones"]}:
                nearest["zones"].append(_public_zone(z))

    body_zones, seen = [], set()
    for z in zone_spans:
        if z["code"] in seen:
            continue
        seen.add(z["code"])
        body_zones.append(_public_zone(z))
    return body_zones


class SymptomExtractionPipeline:
    def __init__(self) -> None:
        import onnxruntime as ort  # import perezoso: solo si se usa NLP
        from transformers import AutoTokenizer

        so = ort.SessionOptions()
        so.intra_op_num_threads = _NUM_THREADS
        so.inter_op_num_threads = 1

        # --- Etapa 1: NER ---
        ner_onnx = self._find_onnx(_NER_DIR)
        self._ner_sess = ort.InferenceSession(ner_onnx, sess_options=so, providers=["CPUExecutionProvider"])
        self._ner_tok = AutoTokenizer.from_pretrained(_NER_DIR)
        self._ner_inputs = {i.name for i in self._ner_sess.get_inputs()}
        with open(os.path.join(_NER_DIR, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self._id2label = {int(k): v for k, v in cfg["id2label"].items()}

        # --- Etapa 3: embeddings ---
        embed_onnx = self._find_onnx(_EMBED_DIR)
        self._emb_sess = ort.InferenceSession(embed_onnx, sess_options=so, providers=["CPUExecutionProvider"])
        self._emb_tok = AutoTokenizer.from_pretrained(_EMBED_DIR)
        self._emb_inputs = {i.name for i in self._emb_sess.get_inputs()}

        # Anclas del catalogo precomputadas una sola vez. Guardamos el codigo de
        # concepto por cada frase ancla y tomamos el maximo coseno (mejor recall).
        anchor_texts, self._anchor_codes = [], []
        for concept in CATALOG:
            for phrase in concept.anchors:
                anchor_texts.append(phrase)
                self._anchor_codes.append(concept.code)
        self._anchor_emb = self._embed(anchor_texts)  # [n_anchors, dim], normalizado

        # --- Etapa 2: negacion ---
        import spacy
        self._nlp = spacy.load(_SPACY_MODEL, disable=["ner", "lemmatizer"])

        # --- Zonas del cuerpo (gazetteer difuso, sin modelo nuevo) ---
        self._zone_matcher = build_zone_matcher(self._nlp)

    @staticmethod
    def _find_onnx(dir_path: str) -> str:
        if not os.path.isdir(dir_path):
            raise RuntimeError(
                f"No existe el directorio de modelo NLP '{dir_path}'. "
                "Corre scripts/export_nlp_models.py una vez para generarlo."
            )
        # Prioriza la version cuantizada int8 si esta presente.
        for name in ("model_quantized.onnx", "model.onnx"):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                return p
        raise RuntimeError(f"No se encontro ningun .onnx en '{dir_path}'.")

    def _feed(self, enc, allowed: set) -> dict:
        feed = {}
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            if name not in allowed:
                continue
            if name in enc:
                feed[name] = enc[name].astype(np.int64)
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        return feed

    def _embed(self, texts: List[str]) -> np.ndarray:
        enc = self._emb_tok(texts, padding=True, truncation=True, max_length=64, return_tensors="np")
        out = self._emb_sess.run(None, self._feed(enc, self._emb_inputs))[0]  # [n, seq, dim]
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        summed = np.sum(out * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vec = summed / counts  # mean pooling
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.clip(norm, 1e-9, None)

    def _ner_spans(self, text: str) -> List[dict]:
        """Decodifica logits ONNX -> spans de caracteres (agregacion 'simple' BIO)."""
        enc = self._ner_tok(text, return_offsets_mapping=True, truncation=True,
                            max_length=256, return_tensors="np")
        offsets = enc.pop("offset_mapping")[0]
        logits = self._ner_sess.run(None, self._feed(enc, self._ner_inputs))[0][0]  # [seq, labels]
        probs = _softmax(logits, axis=-1)
        label_ids = probs.argmax(axis=-1)
        scores = probs.max(axis=-1)

        spans, cur = [], None
        for idx, (start, end) in enumerate(offsets):
            if start == end:  # token especial ([CLS]/[SEP]/pad)
                continue
            label = self._id2label.get(int(label_ids[idx]), "O")
            if label == "O":
                if cur:
                    spans.append(cur); cur = None
                continue
            group = label.split("-", 1)[-1]
            is_begin = label.startswith("B-")
            if cur and not is_begin and cur["group"] == group:
                cur["end"] = int(end); cur["scores"].append(float(scores[idx]))
            else:
                if cur:
                    spans.append(cur)
                cur = {"start": int(start), "end": int(end), "group": group,
                       "scores": [float(scores[idx])]}
        if cur:
            spans.append(cur)

        out = []
        for s in spans:
            avg = float(np.mean(s["scores"]))
            if avg < _NER_SCORE_MIN:
                continue
            raw = text[s["start"]:s["end"]]
            if not _is_content_span(raw):  # descarta 'me', 'la', etc. mal etiquetados
                continue
            out.append({"start": s["start"], "end": s["end"], "raw_text": raw, "score": avg})
        return out

    def _normalize(self, raw_text: str) -> Optional[tuple]:
        """Mapea un span al concepto del catalogo por similitud coseno."""
        vec = self._embed([raw_text])[0]
        sims = self._anchor_emb @ vec  # coseno (todo normalizado)
        best = int(sims.argmax())
        if float(sims[best]) < _SIM_THRESHOLD:
            return None
        return self._anchor_codes[best], float(sims[best])

    def extract(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"symptoms": [], "body_zones": []}
        doc = self._nlp(text)

        results, seen = [], set()
        for span in self._ner_spans(text):
            mapped = self._normalize(span["raw_text"])
            if not mapped:
                continue
            code, sim = mapped
            if code in seen:  # dedup por concepto dentro de una misma entrada
                continue
            seen.add(code)
            concept = CATALOG_BY_CODE[code]
            results.append({
                "code": code,
                "label": concept.label,
                "raw_text": span["raw_text"],
                "negated": _is_negated(doc, span["start"]),
                "score": round(min(span["score"], sim), 4),
                "alarm": concept.alarm,
                "_start": span["start"],  # interno: para vincular zonas
                "_end": span["end"],      # interno: para vincular por contencion
            })

        zone_spans = find_zone_spans(doc, self._zone_matcher)
        body_zones = link_zones(results, zone_spans, doc)

        for s in results:
            s.pop("_start", None)  # no exponer los offsets internos
            s.pop("_end", None)
            s.setdefault("zones", [])

        return {"symptoms": results, "body_zones": body_zones}


_pipeline: Optional[SymptomExtractionPipeline] = None
_lock = threading.Lock()


def get_pipeline() -> SymptomExtractionPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = SymptomExtractionPipeline()
    return _pipeline
