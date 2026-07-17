"""Extraccion de sintomas con LLM local (Ollama) - alternativa experimental al pipeline NER.

Version v3 (2026-07-17): prompt rico + reglas anti-alucinacion + ANCLAJE DIFUSO + validacion Pydantic.
NO requiere entrenamiento: todo es contexto (prompt + few-shot + JSON schema).

Resultados medidos (45 bitacoras, ver evaluation/ y docs/HANDOFF-nlp-llm.md):
    Alarmas detectadas 24/24 | Alucinaciones 4 | Exactas 39/45 (87%).
    (vs pipeline NER curado: alarmas 15/24, alucinaciones 15, exactas 23/45.)
    OJO: el prompt se afino sobre esas 45 bitacoras -> falta validar en un set held-out NUEVO.

Config por variables de entorno:
    OLLAMA_URL   (default http://localhost:11434 ; en Docker: http://ollama:11434)
    OLLAMA_MODEL (default qwen2.5:3b)
    LLM_TIMEOUT  (segundos, default 60 ; latencia real en VPS 4-cores: mediana ~12s)
"""
import json
import os
import re
import urllib.request
from typing import Optional

from pydantic import BaseModel, ValidationError

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

ALLOWED = [
    "CEFALEA", "VISION_BORROSA", "EDEMA", "DOLOR_EPIGASTRICO", "SANGRADO",
    "DISMINUCION_MOVIMIENTO_FETAL", "CONTRACCIONES", "DIFICULTAD_RESPIRATORIA", "MAREO",
    "DEBILIDAD", "FIEBRE", "NAUSEA_VOMITO", "DOLOR_ABDOMINAL", "DISURIA", "OTRO",
]
_ALLOWED_SET = set(ALLOWED)

_SCHEMA = {
    "type": "object",
    "properties": {"symptoms": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": ALLOWED},
            "raw_text": {"type": "string"},
            "negated": {"type": "boolean"},
            "intensity": {"type": ["string", "null"]},
            "duration": {"type": ["string", "null"]},
            "body_zone": {"type": ["string", "null"]},
        },
        "required": ["code", "raw_text", "negated", "intensity", "duration", "body_zone"],
    }}},
    "required": ["symptoms"],
}

# Prompt v3: descripciones ricas de codigo (clasificacion) + reglas anti-alucinacion / anti-hipotetico.
_SYSTEM = """Eres un extractor clinico obstetrico. Tu UNICA fuente es el mensaje de la paciente.
Extrae los SINTOMAS que la paciente dice tener AHORA.

CODIGOS (elige el correcto segun el ejemplo):
- CEFALEA: dolor de cabeza, jaqueca, me duele la cabeza (o "caeza")
- VISION_BORROSA: veo borroso, lucecitas, manchas en la vista
- EDEMA: hinchazon, hinchada, pies/manos/cara/tobillos hinchados
- DOLOR_EPIGASTRICO: dolor en la BOCA DEL ESTOMAGO, debajo de las costillas
- SANGRADO: sangrado, manchado, hemorragia vaginal
- DISMINUCION_MOVIMIENTO_FETAL: el bebe no se mueve, se mueve menos, no siento al bebe
- CONTRACCIONES: contracciones, el vientre se endurece/esta duro
- DIFICULTAD_RESPIRATORIA: falta de aire, me ahogo, no puedo respirar, me cuesta respirar
- MAREO: mareo, mareada, vertigo
- DEBILIDAD: debilidad, cansancio, fatiga, sin fuerzas, agotada
- FIEBRE: fiebre, calentura, escalofrios
- NAUSEA_VOMITO: nauseas, ganas de vomitar, vomito
- DOLOR_ABDOMINAL: dolor de vientre, colico, dolor de barriga (NO la boca del estomago, NO al orinar)
- DISURIA: ardor o dolor AL ORINAR
- OTRO: cualquier sintoma que NO este arriba (comezon, tos, diarrea, insomnio, palpitaciones, dolor de espalda, romper la fuente)

REGLAS:
1. raw_text = copia LITERAL de palabras del mensaje de la paciente. NUNCA copies de estas instrucciones/ejemplos.
2. Extrae TODOS los sintomas, incluso si comparten una negacion ("no tengo A ni B" -> A y B).
3. Si el sintoma NO esta en la lista, usa OTRO. No lo fuerces al mas parecido.
4. negated=true SOLO si la paciente NIEGA tenerlo ("no tengo X", "ya no", "sin X"). Se reporta igual.
   "no puedo respirar" y "el bebe no se mueve" NO son negaciones: SON el sintoma (negated=false).
5. Ignora sintomas hipoteticos, de terceros, o del medico. Si la paciente dice que algo esta BIEN/NORMAL, NO es sintoma.
6. Si no hay ningun sintoma real, devuelve {"symptoms":[]}.

--- EJEMPLOS (NO son la entrada) ---
IN: "no tengo sangrado ni dolor de cabeza"
OUT: {"symptoms":[{"code":"SANGRADO","raw_text":"no tengo sangrado","negated":true,"intensity":null,"duration":null,"body_zone":null},{"code":"CEFALEA","raw_text":"dolor de cabeza","negated":true,"intensity":null,"duration":null,"body_zone":"cabeza"}]}
IN: "me duele la boca del estomago"
OUT: {"symptoms":[{"code":"DOLOR_EPIGASTRICO","raw_text":"me duele la boca del estomago","negated":false,"intensity":null,"duration":null,"body_zone":null}]}
IN: "me arde al orinar"
OUT: {"symptoms":[{"code":"DISURIA","raw_text":"me arde al orinar","negated":false,"intensity":null,"duration":null,"body_zone":null}]}
IN: "tengo diarrea desde ayer"
OUT: {"symptoms":[{"code":"OTRO","raw_text":"tengo diarrea","negated":false,"intensity":null,"duration":"desde ayer","body_zone":null}]}
IN: "la doctora me dijo que si sangro vaya al hospital"
OUT: {"symptoms":[]}
IN: "todo bien, el bebe se mueve bien"
OUT: {"symptoms":[]}
--- FIN EJEMPLOS ---"""


class ExtractedSymptomLLM(BaseModel):
    code: str
    raw_text: str
    negated: bool
    intensity: Optional[str] = None
    duration: Optional[str] = None
    body_zone: Optional[str] = None


def _content_words(s: str):
    return [w for w in re.findall(r"\w+", s.lower()) if len(w) >= 4]


def _grounded(raw: str, text: str) -> bool:
    """Anclaje DIFUSO: raw_text debe estar (o casi) en el texto de la paciente.

    Mata las alucinaciones por 'fuga de prompt' (el 3B copia sus instrucciones como
    sintomas), pero tolera typos del propio modelo en raw_text (substring exacto fallaria).
    """
    r = " ".join((raw or "").lower().split())
    tl = " ".join(text.lower().split())
    if not r:
        return False
    if r in tl:
        return True
    cw = _content_words(r)
    if not cw:
        return False
    return sum(1 for w in cw if w in tl) / len(cw) >= 0.6


def _call_ollama(text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "format": _SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text},
        ],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())["message"]["content"]


def extract(text: str, use_fallback: bool = False) -> dict:
    """Extrae sintomas con el LLM. Devuelve {"symptoms":[...], "source": "llm"|"fallback"|"empty"}.

    Cada sintoma: code, raw_text, negated, intensity, duration, body_zone.
    Se descartan (anclaje) los que no esten en el texto de la paciente (anti-alucinacion).
    Con use_fallback=True, si el LLM falla usa el pipeline curado (nlp.py) como respaldo.
    """
    text = (text or "").strip()
    if not text:
        return {"symptoms": [], "source": "empty"}
    try:
        obj = json.loads(_call_ollama(text))
        syms = []
        for s in obj.get("symptoms", []):
            try:
                m = ExtractedSymptomLLM(**s)
            except ValidationError:
                continue
            if m.code in _ALLOWED_SET and _grounded(m.raw_text, text):
                syms.append(m.model_dump())
        return {"symptoms": syms, "source": "llm"}
    except Exception:
        if use_fallback:
            import nlp
            return {"symptoms": nlp.get_pipeline().extract(text)["symptoms"], "source": "fallback"}
        raise
