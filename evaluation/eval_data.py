# Dataset de evaluacion: bitacoras + gold standard (etiquetado a mano, proxy sin clinico).
# gold: lista de [CODIGO, negated]. otro=True: hay un sintoma que NO esta en el catalogo de 14.
ALARM = {"CEFALEA","VISION_BORROSA","EDEMA","DOLOR_EPIGASTRICO","SANGRADO",
         "DISMINUCION_MOVIMIENTO_FETAL","CONTRACCIONES","DIFICULTAD_RESPIRATORIA"}

DATA = [
    # --- sintomas simples en catalogo ---
    {"text":"me duele mucho la cabeza y veo lucecitas","gold":[["CEFALEA",False],["VISION_BORROSA",False]],"otro":False},
    {"text":"tengo los pies y las manos muy hinchadas","gold":[["EDEMA",False]],"otro":False},
    {"text":"me arde mucho al orinar desde ayer","gold":[["DISURIA",False]],"otro":False},
    {"text":"estoy sangrando por la vagina","gold":[["SANGRADO",False]],"otro":False},
    {"text":"el bebe casi no se mueve hoy","gold":[["DISMINUCION_MOVIMIENTO_FETAL",False]],"otro":False},
    {"text":"tengo contracciones cada diez minutos","gold":[["CONTRACCIONES",False]],"otro":False},
    {"text":"no puedo respirar bien, me agito","gold":[["DIFICULTAD_RESPIRATORIA",False]],"otro":False},
    {"text":"me duele la boca del estomago","gold":[["DOLOR_EPIGASTRICO",False]],"otro":False},
    {"text":"tengo mucha fiebre y escalofrios","gold":[["FIEBRE",False]],"otro":False},
    {"text":"me siento muy mareada cuando me paro","gold":[["MAREO",False]],"otro":False},
    {"text":"estoy muy cansada y sin fuerzas","gold":[["DEBILIDAD",False]],"otro":False},
    {"text":"tengo nauseas y ganas de vomitar","gold":[["NAUSEA_VOMITO",False]],"otro":False},
    {"text":"me duele el vientre como colico","gold":[["DOLOR_ABDOMINAL",False]],"otro":False},
    # --- sin sintomas ---
    {"text":"todo bien hoy, sin molestias","gold":[],"otro":False},
    {"text":"hoy me senti tranquila, sin novedades","gold":[],"otro":False},
    {"text":"no me duele nada, solo vengo a control","gold":[],"otro":False},
    # --- negacion real ---
    {"text":"no tengo sangrado ni dolor de cabeza","gold":[["SANGRADO",True],["CEFALEA",True]],"otro":False},
    {"text":"ya no tengo nauseas, se me quitaron","gold":[["NAUSEA_VOMITO",True]],"otro":False},
    {"text":"no he tenido fiebre en estos dias","gold":[["FIEBRE",True]],"otro":False},
    {"text":"no siento dolor de cabeza pero si mareos","gold":[["CEFALEA",True],["MAREO",False]],"otro":False},
    {"text":"me duele la cabeza, no tengo fiebre","gold":[["CEFALEA",False],["FIEBRE",True]],"otro":False},
    # --- negacion que ES el sintoma (trampa NegEx) ---
    {"text":"el bebe no se mueve y no puedo respirar","gold":[["DISMINUCION_MOVIMIENTO_FETAL",False],["DIFICULTAD_RESPIRATORIA",False]],"otro":False},
    {"text":"me siento sin fuerzas y con nauseas","gold":[["DEBILIDAD",False],["NAUSEA_VOMITO",False]],"otro":False},
    # --- typos ---
    {"text":"me duele la caeza y estoy mareda","gold":[["CEFALEA",False],["MAREO",False]],"otro":False},
    {"text":"tengo bision borrosa","gold":[["VISION_BORROSA",False]],"otro":False},
    {"text":"sangrre un poco ayer","gold":[["SANGRADO",False]],"otro":False},
    {"text":"me arde cuando ago pipi","gold":[["DISURIA",False]],"otro":False},
    # --- fuera de catalogo (LLM=OTRO, NER ideal=nada) ---
    {"text":"tengo mucha comezon en la piel","gold":[],"otro":True},
    {"text":"tengo tos y flemas","gold":[],"otro":True},
    {"text":"no puedo dormir en las noches","gold":[],"otro":True},
    {"text":"tengo diarrea desde ayer","gold":[],"otro":True},
    {"text":"siento palpitaciones en el pecho","gold":[],"otro":True},
    {"text":"se me rompio la fuente","gold":[],"otro":True},
    {"text":"tengo dolor de espalda baja","gold":[],"otro":True},
    # --- multi-sintoma / desordenado ---
    {"text":"hoy amaneci con dolor de cabeza fuerte, los pies hinchados y ademas me arde al orinar","gold":[["CEFALEA",False],["EDEMA",False],["DISURIA",False]],"otro":False},
    {"text":"desde hace tres dias tengo el vientre duro con contracciones y algo de sangrado","gold":[["CONTRACCIONES",False],["SANGRADO",False]],"otro":False},
    {"text":"tengo hinchados los tobillos y me cuesta respirar al caminar","gold":[["EDEMA",False],["DIFICULTAD_RESPIRATORIA",False]],"otro":False},
    {"text":"me duele al orinar y tengo un poco de fiebre","gold":[["DISURIA",False],["FIEBRE",False]],"otro":False},
    {"text":"estoy sangrando bastante y me siento debil y mareada","gold":[["SANGRADO",False],["DEBILIDAD",False],["MAREO",False]],"otro":False},
    # --- intensidad/duracion ---
    {"text":"me duele la cabeza desde hace 4 dias, cada vez mas fuerte","gold":[["CEFALEA",False]],"otro":False},
    {"text":"tuve un dolor de cabeza leve en la manana","gold":[["CEFALEA",False]],"otro":False},
    {"text":"vi como lucecitas y manchas en la vista esta tarde","gold":[["VISION_BORROSA",False]],"otro":False},
    # --- trampas de alucinacion / falso positivo ---
    {"text":"fui al control y me revisaron la cabeza y el vientre, todo normal","gold":[],"otro":False},
    {"text":"la doctora me dijo que si tengo sangrado vaya al hospital","gold":[],"otro":False},
    {"text":"sin sangrado, el bebe se mueve bien","gold":[["SANGRADO",True]],"otro":False},
]
