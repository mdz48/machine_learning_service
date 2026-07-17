# v3: prompt RICO (clasifica) + reglas anti-alucinacion + anclaje DIFUSO (recupera typos del modelo).
import json, re, sys, time, urllib.request
from eval_data import DATA

URL, MODEL = "http://localhost:11434/api/chat", "qwen2.5:3b"
ALLOWED = ["CEFALEA","VISION_BORROSA","EDEMA","DOLOR_EPIGASTRICO","SANGRADO",
           "DISMINUCION_MOVIMIENTO_FETAL","CONTRACCIONES","DIFICULTAD_RESPIRATORIA","MAREO",
           "DEBILIDAD","FIEBRE","NAUSEA_VOMITO","DOLOR_ABDOMINAL","DISURIA","OTRO"]
SCHEMA = {"type":"object","properties":{"symptoms":{"type":"array","items":{"type":"object",
  "properties":{"code":{"type":"string","enum":ALLOWED},"raw_text":{"type":"string"},
    "negated":{"type":"boolean"},"intensity":{"type":["string","null"]},
    "duration":{"type":["string","null"]},"body_zone":{"type":["string","null"]}},
  "required":["code","raw_text","negated","intensity","duration","body_zone"]}}},"required":["symptoms"]}

SYS = """Eres un extractor clinico obstetrico. Tu UNICA fuente es el mensaje de la paciente.
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

def _content(s): return [w for w in re.findall(r"\w+", s.lower()) if len(w) >= 4]
def grounded(raw, text):
    r = " ".join((raw or "").lower().split()); tl = " ".join(text.lower().split())
    if not r: return False
    if r in tl: return True
    cw = _content(r)
    if not cw: return False
    return sum(1 for w in cw if w in tl) / len(cw) >= 0.6

def call(texto):
    p={"model":MODEL,"format":SCHEMA,"stream":False,"options":{"temperature":0},
       "messages":[{"role":"system","content":SYS},{"role":"user","content":texto}]}
    req=urllib.request.Request(URL,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req,timeout=300) as r: resp=json.loads(r.read())
    return json.loads(resp["message"]["content"]).get("symptoms",[]), time.perf_counter()-t0

OUTF="/tmp/llm3.jsonl"; open(OUTF,"w").close()
call("calentando")
for i,c in enumerate(DATA):
    try: syms,dt=call(c["text"])
    except Exception: syms,dt=[],-1
    kept=[s for s in syms if grounded(s.get("raw_text",""), c["text"])]
    dropped=[s for s in syms if not grounded(s.get("raw_text",""), c["text"])]
    with open(OUTF,"a",encoding="utf-8") as f:
        f.write(json.dumps({"text":c["text"],"symptoms":kept,"dropped":dropped,"lat":round(dt,1)},ensure_ascii=False)+"\n"); f.flush()
open("/tmp/llm3.done","w").write("ok")
