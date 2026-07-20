import os
from fastapi import APIRouter, HTTPException

from app.core.config import NLP_NER_MODEL, OLLAMA_MODEL
from app.features.nlp_symptom_extraction.domain.nlp_schemas import (
    SymptomExtractionRequest,
    SymptomExtractionResponse,
    ExtractedSymptom,
)
from app.features.nlp_symptom_extraction.domain.nlp_catalog import CATALOG_BY_CODE
from app.features.nlp_symptom_extraction.services.onnx_extractor_service import get_pipeline
from app.features.nlp_symptom_extraction.services import llm_extractor_service

router = APIRouter(prefix="/nlp", tags=["NLP Symptom Extraction"])


@router.post("/extract-symptoms", response_model=SymptomExtractionResponse)
def extract_symptoms(req: SymptomExtractionRequest):
    try:
        pipeline = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Servicio NLP no disponible: {e}")
    try:
        result = pipeline.extract(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando texto: {e}")
    return {
        "symptoms": result["symptoms"],
        "body_zones": result["body_zones"],
        "model_version": NLP_NER_MODEL,
    }


@router.post("/extract-symptoms-llm", response_model=SymptomExtractionResponse)
def extract_symptoms_llm(req: SymptomExtractionRequest):
    try:
        result = llm_extractor_service.extract(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en LLM extractor: {e}")
    
    symptoms_list = []
    for s in result.get("symptoms", []):
        code = s.get("code")
        concept = CATALOG_BY_CODE.get(code)
        label = concept.label if concept else code
        alarm = concept.alarm if concept else False
        symptoms_list.append(
            ExtractedSymptom(
                code=code,
                label=label,
                raw_text=s.get("raw_text", ""),
                negated=s.get("negated", False),
                score=1.0,
                alarm=alarm,
                zones=[]
            )
        )
    return {
        "symptoms": symptoms_list,
        "body_zones": [],
        "model_version": OLLAMA_MODEL,
    }
