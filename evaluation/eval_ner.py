import json, os, sys
sys.path.insert(0, r"C:\Universidad\machine_learning_service")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp
from eval_data import DATA

p = nlp.get_pipeline()
out = []
for c in DATA:
    r = p.extract(c["text"])
    syms = [{"code": s["code"], "negated": s["negated"]} for s in r["symptoms"]]
    out.append({"text": c["text"], "symptoms": syms})
print(json.dumps(out, ensure_ascii=False))
