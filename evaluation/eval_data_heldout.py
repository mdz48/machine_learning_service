# Dataset de evaluacion held-out: bitacoras + gold standard (etiquetado a mano, proxy sin clinico).
# gold: lista de [CODIGO, negated]. otro=True: hay un sintoma que NO esta en el catalogo de 14.
ALARM = {"CEFALEA", "VISION_BORROSA", "EDEMA", "DOLOR_EPIGASTRICO", "SANGRADO",
         "DISMINUCION_MOVIMIENTO_FETAL", "CONTRACCIONES", "DIFICULTAD_RESPIRATORIA"}

DATA = [
    # 1. simples/alarmas
    {"text": "siento que la cabeza me va a explotar y no soporto la luz", "gold": [["CEFALEA", False]], "otro": True},
    {"text": "tengo la cara super hinchada, casi no puedo abrir los ojos", "gold": [["EDEMA", False]], "otro": False},
    {"text": "amaneci con dolor debajo de las costillas en el lado derecho", "gold": [["DOLOR_EPIGASTRICO", False]], "otro": False},
    {"text": "estoy manchando la ropa interior de color cafe oscuro", "gold": [["SANGRADO", False]], "otro": False},
    {"text": "no he sentido pataditas del bebe en toda la tarde", "gold": [["DISMINUCION_MOVIMIENTO_FETAL", False]], "otro": False},
    {"text": "siento que la panza se me pone dura y me duele cada 15 minutos", "gold": [["CONTRACCIONES", False]], "otro": False},
    {"text": "me falta el aire hasta para hablar, me canso rapido", "gold": [["DIFICULTAD_RESPIRATORIA", False], ["DEBILIDAD", False]], "otro": False},
    # 8. no alarmas / disuria
    {"text": "me arde bastante al hacer pis y voy muy seguido", "gold": [["DISURIA", False]], "otro": True},
    {"text": "tengo una calentura muy fuerte y tiemblo de frio", "gold": [["FIEBRE", False]], "otro": False},
    {"text": "no me para el vomito y no tolero ni el agua", "gold": [["NAUSEA_VOMITO", False]], "otro": False},
    {"text": "tengo un dolor tipo colico en el bajo vientre", "gold": [["DOLOR_ABDOMINAL", False]], "otro": False},
    {"text": "siento que todo me da vueltas si me levanto rapido", "gold": [["MAREO", False]], "otro": False},
    # 13. sin sintomas
    {"text": "hoy me siento muy bien, no he tenido ninguna molestia", "gold": [], "otro": False},
    {"text": "cero sintomas, me he sentido muy tranquila y descansada", "gold": [], "otro": False},
    # 15. negacion
    {"text": "no tengo dolor de cabeza ni tampoco sangrado", "gold": [["CEFALEA", True], ["SANGRADO", True]], "otro": False},
    {"text": "se me quito la hinchazon pero ahora me duele el vientre", "gold": [["EDEMA", True], ["DOLOR_ABDOMINAL", False]], "otro": False},
    {"text": "no he tenido fiebre pero si muchas nauseas", "gold": [["FIEBRE", True], ["NAUSEA_VOMITO", False]], "otro": False},
    # 18. trampa negacion
    {"text": "el bebe no se mueve y me cuesta mucho respirar", "gold": [["DISMINUCION_MOVIMIENTO_FETAL", False], ["DIFICULTAD_RESPIRATORIA", False]], "otro": False},
    # 19. typos / otro
    {"text": "me duele la cabesa y no puedo dormir", "gold": [["CEFALEA", False]], "otro": True},
    {"text": "veo como manchitas negras flotando y luces de colores", "gold": [["VISION_BORROSA", False]], "otro": False},
    {"text": "tengo mucha tos seca y dolor de garganta", "gold": [], "otro": True},
    {"text": "se me esta saliendo un liquido claro como agua por las piernas", "gold": [], "otro": True},
    {"text": "me duele la espalda baja desde hace dos dias", "gold": [], "otro": True},
    # 24. multi-sintoma
    {"text": "hoy amaneci con jaqueca, los tobillos inflados y ardor cuando voy al bano", "gold": [["CEFALEA", False], ["EDEMA", False], ["DISURIA", False]], "otro": False},
    # 25. hipoteticos/trampas
    {"text": "me dijo la enfermera que si tengo contracciones vaya a urgencias", "gold": [], "otro": False},
    {"text": "sin dolor de cabeza hoy, el bebe se esta moviendo bien", "gold": [["CEFALEA", True]], "otro": False},
]
