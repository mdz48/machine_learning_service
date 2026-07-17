import json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from eval_data import DATA, ALARM

def load(f): return json.load(open(os.path.join(D, f), encoding="utf-8"))
ner = {r["text"]: r["symptoms"] for r in load("ner_results.json")}
llm = {r["text"]: r["symptoms"] for r in load("llm_results.json")}
lats = [r.get("lat", 0) for r in load("llm_results.json") if r.get("lat", 0) > 0]

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
        if got == gold and (not c["otro"] or has_otro or True):  # otro se mide aparte
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
            ok = has_otro if name == "LLM" else (len(got) == 0)  # NER no dice OTRO: acierta si no inventa
            if ok: M["otro_ok"] += 1
    return M, danger

mn, dn = evaluate("NER", ner)
ml, dl = evaluate("LLM", llm)

def row(lbl, kn, kl, tot=None):
    a = f"{mn[kn]}/{mn[tot]}" if tot else str(mn[kn])
    b = f"{ml[kl]}/{ml[tot]}" if tot else str(ml[kl])
    print(f"  {lbl:52} NER: {a:8}  LLM: {b:8}")

print("="*84)
print("COMPARACION  (NER local  vs  LLM qwen2.5:3b VPS)   sobre", len(DATA), "bitacoras")
print("="*84)
print("\n-- SEGURIDAD (lo critico: va directo al medico, sin revision) --")
row("Signos de ALARMA presentes detectados (recall)", "alarm_hit", "alarm_hit", "alarm_tot")
row("  >> ALARMAS perdidas (no detectadas)  [PELIGRO]", "missed_alarm", "missed_alarm")
row("  >> ALARMAS presentes marcadas NEGADAS [PELIGRO]", "flip_alarm", "flip_alarm")
row("Alucinaciones (sintomas de catalogo inventados)", "halluc", "halluc")
print("\n-- CALIDAD GENERAL --")
row("Sintomas presentes perdidos (total)", "missed", "missed")
row("Flips de negacion (presente->negado, total)", "flip", "flip")
row("Negaciones reales bien marcadas", "realneg_ok", "realneg_ok", "realneg_tot")
row("Fuera-de-catalogo bien manejado", "otro_ok", "otro_ok", "otro_tot")
row("Bitacoras EXACTAS (codigo+negacion)", "exact", "exact")
print(f"  {'de un total de bitacoras':52} {len(DATA)}")
if lats:
    import statistics
    print(f"\nLatencia LLM (VPS): mediana {statistics.median(lats):.1f}s | max {max(lats):.1f}s")

print("\n" + "="*84)
print("CASOS PELIGROSOS (alarmas perdidas/invertidas y alucinaciones)")
print("="*84)
for name, tag in [("NER", dn), ("LLM", dl)]:
    print(f"\n[{name}]")
    peligros = [d for d in tag if "ALARMA" in d[2] or "INVENTADO" in d[2]]
    if not peligros: print("  (ninguno)")
    for _, txt, desc in peligros:
        print(f"  - {txt[:52]!r:56} -> {desc}")
