# Dataset held-out 5: 30 casos nuevos (extremos y trampas) para 7B.
ALARM = {"CEFALEA", "VISION_BORROSA", "EDEMA", "DOLOR_EPIGASTRICO", "SANGRADO",
         "DISMINUCION_MOVIMIENTO_FETAL", "CONTRACCIONES", "DIFICULTAD_RESPIRATORIA"}

DATA = [
    # 1-5: Falsas alarmas ortográficas y semánticas
    {"text": "me duele la cabeza de tanto pensar en el parto pero hoy amaneci mejor", "gold": [["CEFALEA", False]], "otro": False},
    {"text": "siento q me ahogo por el calor, pero de aire ando bien", "gold": [], "otro": False}, # "ahogo" pero niega falta de aire
    {"text": "tuve un leve manchado ayer, es sangre marroncita", "gold": [["SANGRADO", False]], "otro": False},
    {"text": "tengo el estomago duro, la panza parece una piedra", "gold": [["CONTRACCIONES", False]], "otro": False},
    {"text": "estoy viendo como lucecitas o chispas volando", "gold": [["VISION_BORROSA", False]], "otro": False},

    # 6-10: Combinación de Ausencia y Presencia
    {"text": "el bebe se esta moviendo muchisimo, aunque sigo con asco y vomito", "gold": [["NAUSEA_VOMITO", False]], "otro": False},
    {"text": "no he sentido que patee en todo el dia, y aparte tengo punzadas en el vientre", "gold": [["DISMINUCION_MOVIMIENTO_FETAL", False], ["DOLOR_ABDOMINAL", False]], "otro": False},
    {"text": "cero contracciones y cero dolor de barriga", "gold": [["CONTRACCIONES", True], ["DOLOR_ABDOMINAL", True]], "otro": False},
    {"text": "cero sangrado gracias a dios pero los tobillos siguen hinchados", "gold": [["SANGRADO", True], ["EDEMA", False]], "otro": False},
    {"text": "sin novedad, todo muy tranquilo, el bebe patea fuerte", "gold": [], "otro": False},

    # 11-15: Dolores y Síntomas fuera del catálogo estricto (OTRO)
    {"text": "llevo tres noches de insomnio fatal, no puedo dormir nada", "gold": [], "otro": True},
    {"text": "me pica toda la barriga, es una comezon que no aguanto", "gold": [], "otro": True},
    {"text": "la acidez en la garganta me esta matando", "gold": [], "otro": True}, # Acidez = OTRO
    {"text": "me duelen las coyunturas de las manos", "gold": [], "otro": True}, # Dolor articular = OTRO
    {"text": "tengo un dolor insoportable en los dientes", "gold": [], "otro": True},

    # 16-20: Múltiples negaciones en cascada
    {"text": "ni dolor de cabeza, ni mareo, ni veo borroso", "gold": [["CEFALEA", True], ["MAREO", True], ["VISION_BORROSA", True]], "otro": False},
    {"text": "hoy no me duele la panza ni tengo colicos", "gold": [["DOLOR_ABDOMINAL", True]], "otro": False},
    {"text": "fui al bano y no me ardio, tampoco tuve sangrado", "gold": [["DISURIA", True], ["SANGRADO", True]], "otro": False},
    {"text": "sin escalofrios ni calentura", "gold": [["FIEBRE", True]], "otro": False}, # escalofrios = fiebre, calentura = fiebre.
    {"text": "no se me hincharon los pies ni las manos", "gold": [["EDEMA", True]], "otro": False},

    # 21-25: Zonas del dolor confusas
    {"text": "me duele muchisimo la boca del estomago, justo bajo las costillas", "gold": [["DOLOR_EPIGASTRICO", False]], "otro": False},
    {"text": "tengo un dolor en la pelvis y vientre bajo muy fuerte", "gold": [["DOLOR_ABDOMINAL", False]], "otro": False},
    {"text": "me duele la mitad de la cara, como migraña", "gold": [["CEFALEA", False]], "otro": False},
    {"text": "ardor horrible al orinar desde la manana", "gold": [["DISURIA", False]], "otro": False},
    {"text": "estoy muy cansada, fatigada, no me quiero ni parar de la cama", "gold": [["DEBILIDAD", False]], "otro": False},

    # 26-30: Casos híbridos y raros
    {"text": "se me esta nublando la vista y tengo mareos", "gold": [["VISION_BORROSA", False], ["MAREO", False]], "otro": False},
    {"text": "vomite todo lo que comi ayer, pero no tengo dolor", "gold": [["NAUSEA_VOMITO", False]], "otro": False},
    {"text": "ya no tengo fiebre, pero sigo con debilidad", "gold": [["FIEBRE", True], ["DEBILIDAD", False]], "otro": False},
    {"text": "tengo el vientre un poco tenso pero creo que es normal, patea bien", "gold": [["CONTRACCIONES", False]], "otro": False}, # "vientre tenso/duro" -> contracciones, aunque lo llame normal, es un sintoma extraido
    {"text": "todo normal", "gold": [], "otro": False}
]
