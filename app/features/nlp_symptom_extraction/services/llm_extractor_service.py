"""Extraccion de sintomas con LLM local (Ollama) via Qwen 2.5."""
import json
import os
import re
import urllib.request
from typing import Optional
from pydantic import BaseModel, ValidationError

from app.core.config import OLLAMA_MODEL

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

ALLOWED = [
    "CEFALEA", "VISION_BORROSA", "EDEMA", "DOLOR_EPIGASTRICO", "SANGRADO",
    "DISMINUCION_MOVIMIENTO_FETAL", "CONTRACCIONES", "DIFICULTAD_RESPIRATORIA", "MAREO",
    "DEBILIDAD", "FIEBRE", "NAUSEA_VOMITO", "DOLOR_ABDOMINAL", "DISURIA", "OTRO",
]
_ALLOWED_SET = set(ALLOWED)

_ALARM_SET = {
    "CEFALEA", "VISION_BORROSA", "EDEMA", "DOLOR_EPIGASTRICO", "SANGRADO",
    "DISMINUCION_MOVIMIENTO_FETAL", "CONTRACCIONES", "DIFICULTAD_RESPIRATORIA",
}

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

_SYSTEM = """Eres un extractor clinico obstetrico. Tu UNICA fuente es el mensaje de la paciente.
Extrae los SINTOMAS que la paciente dice tener AHORA.

CODIGOS (elige el correcto segun el ejemplo):
- CEFALEA: dolor de cabeza, jaqueca, me duele la cabeza (o "caeza")
- VISION_BORROSA: veo borroso, lucecitas, manchas en la vista
- EDEMA: hinchazon, hinchada, pies/manos/cara/tobillos hinchados, gordos, abultados, como globos
- DOLOR_EPIGASTRICO: dolor en la BOCA DEL ESTOMAGO, debajo de las costillas
- SANGRADO: sangrado, manchado, hemorragia vaginal
- DISMINUCION_MOVIMIENTO_FETAL: el bebe no se mueve, se mueve menos, no siento al bebe, no patea
- CONTRACCIONES: contracciones, el vientre/estomago/panza se endurece/esta duro/como piedra
- DIFICULTAD_RESPIRATORIA: falta de aire, me ahogo, no puedo respirar, me cuesta respirar
- MAREO: mareo, mareada, vertigo
- DEBILIDAD: debilidad, cansancio, fatiga, sin fuerzas, agotada (NOT escalofrios: escalofrios -> FIEBRE)
- FIEBRE: fiebre, calentura, escalofrios
- NAUSEA_VOMITO: nauseas, ganas de vomitar, vomito, vomite
- DOLOR_ABDOMINAL: dolor de vientre, colico, dolor de barriga, pinchazos, punzadas en el vientre (NO la boca del estomago, NO al orinar, NO dolor de cuerpo entero)
- DISURIA: ardor o dolor AL ORINAR
- OTRO: cualquier sintoma que NO este en la lista de arriba: comezon/picazon, tos, diarrea, insomnio/no
  dormir, palpitaciones, dolor de espalda, romper la fuente/liquido amniotico, fotofobia/sensibilidad
  a la luz, dolor de garganta, caida de cabello, ronchas/erupciones. Si tienes duda entre OTRO y
  cualquier codigo, usa OTRO — nunca fuerces un sintoma al codigo mas parecido.

REGLAS:
1. raw_text = copia LITERAL Y EXACTA de palabras del mensaje de la paciente. NUNCA pluralices ni cambies palabras. NUNCA copies de estas instrucciones/ejemplos.
2. Extrae TODOS los sintomas mencionados, aunque compartan una negacion:
   - "no tengo A ni B" -> A negado Y B negado (dos entradas separadas).
   - "sin A ni B", "ya no tengo A ni tampoco B" -> igual, dos entradas negadas.
   - NO omitas el segundo (o tercer) sintoma de una negacion compuesta.
3. Si el sintoma NO esta en la lista, usa OTRO. No lo fuerces al mas parecido.
4. negated=true SOLO si la paciente NIEGA tenerlo ("no tengo X", "ya no", "sin X"). Se reporta igual.
   PRINCIPIO: cuando la AUSENCIA de algo ES el problema, usa negated=false. Ejemplos:
   - "no puedo respirar" -> DIFICULTAD_RESPIRATORIA negated:false
   - "no veo bien" -> VISION_BORROSA negated:false
   - "el bebe no se mueve" / "no patea" / "no da senales" -> DISMINUCION_MOVIMIENTO_FETAL negated:false
   - "no tolero" / "no puedo caminar" -> sintoma presente (negated:false)
5. Ignora sintomas hipoteticos, de terceros, o del medico. Si la paciente dice que algo esta BIEN,
   NORMAL, SIN NOVEDAD o TODO TRANQUILO -> NO es sintoma. "sin novedad" = normal, no extraer nada.
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
IN: "no tengo contracciones ni tampoco sangrado"
OUT: {"symptoms":[{"code":"CONTRACCIONES","raw_text":"no tengo contracciones","negated":true,"intensity":null,"duration":null,"body_zone":null},{"code":"SANGRADO","raw_text":"tampoco sangrado","negated":true,"intensity":null,"duration":null,"body_zone":null}]}
IN: "tengo tos y dolor de garganta"
OUT: {"symptoms":[{"code":"OTRO","raw_text":"tos","negated":false,"intensity":null,"duration":null,"body_zone":null},{"code":"OTRO","raw_text":"dolor de garganta","negated":false,"intensity":null,"duration":null,"body_zone":null}]}
IN: "me molesta mucho la luz del sol"
OUT: {"symptoms":[{"code":"OTRO","raw_text":"me molesta mucho la luz del sol","negated":false,"intensity":null,"duration":null,"body_zone":null}]}
IN: "el bebe se mueve normal hoy"
OUT: {"symptoms":[]}
IN: "el bebe se mueve bien, todo sin novedad"
OUT: {"symptoms":[]}
IN: "no veo bien, todo se me nubla"
OUT: {"symptoms":[{"code":"VISION_BORROSA","raw_text":"no veo bien","negated":false,"intensity":null,"duration":null,"body_zone":null}]}
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
            if m.code not in _ALLOWED_SET:
                continue
            if m.code in _ALARM_SET or m.code == "OTRO" or _grounded(m.raw_text, text):
                syms.append(m.model_dump())
        return {"symptoms": syms, "source": "llm"}

    except Exception:
        if use_fallback:
            from app.features.nlp_symptom_extraction.services.onnx_extractor_service import get_pipeline
            return {"symptoms": get_pipeline().extract(text)["symptoms"], "source": "fallback"}
        raise
