from pydantic import BaseModel
from typing import List, Optional


class SymptomExtractionRequest(BaseModel):
    text: str


class ExtractedZone(BaseModel):
    code: str
    label: str
    raw_text: str
    negated: bool
    score: float


class ExtractedSymptom(BaseModel):
    code: str
    label: str
    raw_text: str
    negated: bool
    score: float
    alarm: bool
    zones: List[ExtractedZone] = []


class SymptomExtractionResponse(BaseModel):
    symptoms: List[ExtractedSymptom]
    body_zones: List[ExtractedZone]
    model_version: str
