import json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from eval_data_heldout import DATA, ALARM

# Carga de resultados JSONL de LLM
llm_file = os.path.join(D, "llm3_heldout.jsonl")
if not os.path.exists(llm_file):
    print(f"Error: No se encuentra el archivo de resultados {llm_file}")
    sys.exit(1)

llm = {}
lats = []
with open(llm_file, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        llm[r["text"]] = r["symptoms"]
        if "lat" in r and r["lat"] > 0:
            lats.append(r["lat"])

CATALOG = ALARM | {"MAREO","DEBILIDAD","FIEBRE","NAUSEA_VOMITO","DOLOR_ABDOMINAL","DISURIA"}

def norm(syms):
    d, otro = {}, False
    for s in syms:
        c = s.get("code")
        if c == "OTRO": otro = True
        elif c in CATALOG: d[c] = bool(s.get("negated"))
    return d, otro

def evaluate(name, results):
    M = dict(alarm_tot=0, alarm_hit=0, flip=0, flip_alarm=0, missed=0, missed_alarm=0,
             halluc=0, realneg_tot=0, realneg_ok=0, otro_tot=0, otro_ok=0, exact=0)
    danger = []
    for c in DATA:
        gold = {code: neg for code, neg in c["gold"]}
        got, has_otro = norm(results.get(c["text"], []))
        
        # exact match sobre catalogo (codigo+negacion)
        # Nota: en eval_score.py original se evalúa got == gold
        if got == gold:
            M["exact"] += 1
            
        # por sintoma gold
        for code, neg in gold.items():
            is_alarm = code in ALARM
            if not neg:  # sintoma PRESENTE
                if is_alarm: M["alarm_tot"] += 1
                if code in got and got[code] == False:
                    if is_alarm: M["alarm_hit"] += 1
                elif code in got and got[code] == True:  # marcado negado = flip peligroso
                    M["flip"] += 1
                    if is_alarm: M["flip_alarm"] += 1
                    danger.append((name, c["text"], f"{code} PRESENTE marcado NEGADO"))
                else:  # ausente = perdido
                    M["missed"] += 1
                    if is_alarm:
                        M["missed_alarm"] += 1
                        danger.append((name, c["text"], f"{code} (ALARMA) NO detectado"))
            else:  # negacion real
                M["realneg_tot"] += 1
                if code in got and got[code] == True: M["realneg_ok"] += 1
                
        # alucinaciones: codigos de catalogo que el sistema puso y no estan en gold
        for code in got:
            if code not in gold:
                M["halluc"] += 1
                danger.append((name, c["text"], f"{code} INVENTADO (no presente)"))
                
        # fuera de catalogo
        if c["otro"]:
            M["otro_tot"] += 1
            ok = has_otro  # LLM acierta si detectó OTRO
            if ok: M["otro_ok"] += 1
            
    return M, danger

ml, dl = evaluate("LLM_V3", llm)

print("="*84)
print("EVALUACION LLM v3   sobre", len(DATA), "bitacoras de test HELD-OUT")
print("="*84)
print("\n-- SEGURIDAD (lo critico: va directo al medico, sin revision) --")
print(f"  Signos de ALARMA presentes detectados (recall): {ml['alarm_hit']}/{ml['alarm_tot']}")
print(f"  >> ALARMAS perdidas (no detectadas)  [PELIGRO]: {ml['missed_alarm']}")
print(f"  >> ALARMAS presentes marcadas NEGADAS [PELIGRO]: {ml['flip_alarm']}")
print(f"  Alucinaciones (sintomas de catalogo inventados): {ml['halluc']}")

print("\n-- CALIDAD GENERAL --")
print(f"  Sintomas presentes perdidos (total)            : {ml['missed']}")
print(f"  Flips de negacion (presente->negado, total)    : {ml['flip']}")
print(f"  Negaciones reales bien marcadas                : {ml['realneg_ok']}/{ml['realneg_tot']}")
print(f"  Fuera-de-catalogo bien manejado                : {ml['otro_ok']}/{ml['otro_tot']}")
print(f"  Bitacoras EXACTAS (codigo+negacion)            : {ml['exact']}/{len(DATA)} ({ml['exact']/len(DATA)*100:.1f}%)")

if lats:
    import statistics
    print(f"\nLatencia LLM (VPS): mediana {statistics.median(lats):.1f}s | max {max(lats):.1f}s | min {min(lats):.1f}s")

print("\n" + "="*84)
print("CASOS PELIGROSOS (alarmas perdidas/invertidas y alucinaciones)")
print("="*84)
peligros = [d for d in dl if "ALARMA" in d[2] or "INVENTADO" in d[2]]
if not peligros:
    print("  (ninguno)")
else:
    for _, txt, desc in peligros:
        print(f"  - {txt[:52]!r:56} -> {desc}")
