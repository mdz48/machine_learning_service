"""Catalogo clinico para la normalizacion semantica de sintomas."""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SymptomConcept:
    code: str
    label: str
    anchors: List[str] = field(default_factory=list)
    alarm: bool = False


CATALOG: List[SymptomConcept] = [
    SymptomConcept(
        "CEFALEA", "Cefalea",
        ["dolor de cabeza", "cefalea", "jaqueca", "migrana", "me duele la cabeza"],
        alarm=True,
    ),
    SymptomConcept(
        "VISION_BORROSA", "Alteraciones visuales",
        ["vision borrosa", "veo borroso", "luces o destellos en la vista",
         "veo lucecitas", "manchas en la vista", "escotomas"],
        alarm=True,
    ),
    SymptomConcept(
        "EDEMA", "Edema",
        ["hinchazon", "estoy hinchada", "pies hinchados", "manos hinchadas",
         "retencion de liquidos", "cara hinchada"],
        alarm=True,
    ),
    SymptomConcept(
        "DOLOR_EPIGASTRICO", "Dolor epigastrico",
        ["dolor en la boca del estomago", "dolor epigastrico",
         "dolor debajo de las costillas", "dolor arriba del estomago"],
        alarm=True,
    ),
    SymptomConcept(
        "SANGRADO", "Sangrado vaginal",
        ["sangrado", "hemorragia", "perdida de sangre", "manchado", "sangre vaginal"],
        alarm=True,
    ),
    SymptomConcept(
        "DISMINUCION_MOVIMIENTO_FETAL", "Disminucion de movimientos fetales",
        ["el bebe se mueve menos", "no siento al bebe", "el bebe no se mueve",
         "disminucion de los movimientos del bebe"],
        alarm=True,
    ),
    SymptomConcept(
        "CONTRACCIONES", "Contracciones",
        ["contracciones", "dolores de parto", "se me endurece el vientre",
         "el vientre se pone duro"],
        alarm=True,
    ),
    SymptomConcept(
        "DIFICULTAD_RESPIRATORIA", "Disnea",
        ["dificultad para respirar", "falta de aire", "me ahogo", "me falta el aire"],
        alarm=True,
    ),
    SymptomConcept(
        "MAREO", "Mareo",
        ["mareo", "me siento mareada", "vertigo", "aturdimiento", "me da vueltas la cabeza"],
    ),
    SymptomConcept(
        "DEBILIDAD", "Debilidad / fatiga",
        ["debilidad", "me siento debil", "fatiga", "cansancio", "sin fuerzas",
         "me siento agotada", "malestar general"],
    ),
    SymptomConcept(
        "FIEBRE", "Fiebre",
        ["fiebre", "temperatura alta", "calentura", "tengo temperatura", "escalofrios"],
    ),
    SymptomConcept(
        "NAUSEA_VOMITO", "Nauseas / vomito",
        ["nauseas", "ganas de vomitar", "vomito", "asco", "ganas de devolver"],
    ),
    SymptomConcept(
        "DOLOR_ABDOMINAL", "Dolor abdominal",
        ["dolor abdominal", "dolor de vientre", "colicos", "dolor en el bajo vientre",
         "dolor de barriga"],
    ),
    SymptomConcept(
        "DISURIA", "Disuria / molestias urinarias",
        ["ardor al orinar", "dolor al orinar", "molestias al orinar", "me arde al hacer pipi"],
    ),
]

CATALOG_BY_CODE = {c.code: c for c in CATALOG}


@dataclass(frozen=True)
class BodyZone:
    code: str
    label: str
    anchors: List[str] = field(default_factory=list)


BODY_ZONE_CATALOG: List[BodyZone] = [
    BodyZone("CABEZA", "Cabeza", ["cabeza", "craneo", "cráneo", "nuca", "sien", "sienes"]),
    BodyZone("OJOS", "Ojos / vista", ["ojos", "ojo", "la vista", "vista", "vision", "visión"]),
    BodyZone("CARA", "Cara / rostro", ["cara", "rostro"]),
    BodyZone("CUELLO", "Cuello", ["cuello", "garganta"]),
    BodyZone("PECHO", "Pecho / tórax", ["pecho", "torax", "tórax"]),
    BodyZone("ESPALDA", "Espalda / zona lumbar", ["espalda", "lumbar", "zona lumbar", "rinones", "riñones"]),
    BodyZone("EPIGASTRIO", "Boca del estómago", ["boca del estomago", "boca del estómago", "epigastrio", "debajo de las costillas"]),
    BodyZone("ABDOMEN", "Vientre / abdomen", ["vientre", "abdomen", "barriga", "estomago", "estómago"]),
    BodyZone("BAJO_VIENTRE", "Bajo vientre / pelvis", ["bajo vientre", "pelvis", "ingle", "bajo abdomen"]),
    BodyZone("MANOS", "Manos", ["manos", "mano", "dedos"]),
    BodyZone("PIES", "Pies / tobillos", ["pies", "pie", "tobillos", "tobillo"]),
    BodyZone("PIERNAS", "Piernas", ["piernas", "pierna", "pantorrillas", "pantorrilla"]),
    BodyZone("BRAZOS", "Brazos", ["brazos", "brazo"]),
    BodyZone("ZONA_GENITAL", "Zona genital / vaginal", ["vagina", "zona genital", "partes intimas", "partes íntimas", "genitales"]),
]

BODY_ZONE_BY_CODE = {z.code: z for z in BODY_ZONE_CATALOG}

EXTRA_STOPWORDS = set()
KEEP_WORDS = set()
