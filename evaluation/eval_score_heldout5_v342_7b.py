import json, os, sys, statistics
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from eval_data_heldout5 import DATA, ALARM

llm_file = os.path.join(D, "llm7b_heldout5_v342.jsonl")
if not os.path.exists(llm_file):
    print(f"Error: No se encuentra {llm_file}"); sys.exit(1)

llm = {}; lats = []
with open(llm_file, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        llm[r["text"]] = r["symptoms"]
        if "lat" in r and r["lat"] > 0: lats.append(r["lat"])

CATALOG = ALARM | {"MAREO","DEBILIDAD","FIEBRE","NAUSEA_VOMITO","DOLOR_ABDOMINAL","DISURIA"}

def norm(syms):
    d, otro = {}, False
    for s in syms:
        c = s.get("code")
        if c == "OTRO": otro = True
        elif c in CATALOG: d[c] = bool(s.get("negated"))
    return d, otro

def evaluate(results):
    M = dict(alarm_tot=0, alarm_hit=0, flip_alarm=0, missed_alarm=0,
             flip=0, missed=0, halluc=0, realneg_tot=0, realneg_ok=0,
             otro_tot=0, otro_ok=0, exact=0)
    danger = []
    for c in DATA:
        gold = {code: neg for code, neg in c["gold"]}
        got, has_otro = norm(results.get(c["text"], []))
        if got == gold: M["exact"] += 1
        for code, neg in gold.items():
            is_alarm = code in ALARM
            if not neg:
                if is_alarm: M["alarm_tot"] += 1
                if code in got and not got[code]:
                    if is_alarm: M["alarm_hit"] += 1
                elif code in got and got[code]:
                    M["flip"] += 1
                    if is_alarm:
                        M["flip_alarm"] += 1
                        danger.append((c["text"], f"{code} PRESENTE marcado NEGADO"))
                else:
                    M["missed"] += 1
                    if is_alarm:
                        M["missed_alarm"] += 1
                        danger.append((c["text"], f"{code} (ALARMA) NO detectado"))
            else:
                M["realneg_tot"] += 1
                if code in got and got[code]: M["realneg_ok"] += 1
        for code in got:
            if code not in gold:
                M["halluc"] += 1
                danger.append((c["text"], f"{code} INVENTADO"))
        if c["otro"]:
            M["otro_tot"] += 1
            if has_otro: M["otro_ok"] += 1
    return M, danger

M, danger = evaluate(llm)
n = len(DATA)
print("="*80)
print(f"EVALUACION LLM 7B v3.4.2 + anclaje asimetrico  |  {n} bitacoras HELD-OUT 5")
print("="*80)
print("\n-- SEGURIDAD --")
print(f"  Alarmas presentes detectadas (recall)  : {M['alarm_hit']}/{M['alarm_tot']}")
print(f"  >> Alarmas perdidas           [PELIGRO]: {M['missed_alarm']}")
print(f"  >> Alarmas invertidas/negadas [PELIGRO]: {M['flip_alarm']}")
print(f"  Alucinaciones (catalogo inventado)      : {M['halluc']}")
print("\n-- CALIDAD GENERAL --")
print(f"  Sintomas presentes perdidos (total)     : {M['missed']}")
print(f"  Negaciones reales correctas             : {M['realneg_ok']}/{M['realneg_tot']}")
print(f"  Fuera-de-catalogo (OTRO) detectado      : {M['otro_ok']}/{M['otro_tot']}")
print(f"  Bitacoras EXACTAS                       : {M['exact']}/{n} ({M['exact']/n*100:.1f}%)")
if lats:
    print(f"\nLatencia: mediana {statistics.median(lats):.1f}s | max {max(lats):.1f}s | min {min(lats):.1f}s")
print("\n" + "="*80)
print("CASOS PELIGROSOS")
print("="*80)
if not danger: print("  (ninguno)")
else:
    for txt, desc in danger:
        print(f"  - {txt[:55]!r:59} -> {desc}")
print("\n" + "="*80)
print("TODOS LOS FALLOS")
print("="*80)
if not danger: print("  (ninguno)")
for txt, desc in danger:
    print(f"  - {txt[:55]!r:59} -> {desc}")




