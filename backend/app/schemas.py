"""Pydantic response/request models for the KisanMitra API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class TopK(BaseModel):
    label: str
    name: str
    confidence: float


class Economics(BaseModel):
    crop_value_at_risk: int = Field(..., description="Rupees of crop protected by treating")
    treatment_cost: int
    net_benefit: int
    yield_value_per_acre: int
    currency: str = "INR"


class Advisory(BaseModel):
    medicine: str
    dose: str
    frequency: str
    pre_harvest_interval_days: int
    safety: str
    note: str


class DiagnosisResponse(BaseModel):
    label: str
    disease: str
    disease_local: Optional[str] = None
    crop: str
    healthy: bool
    confidence: float
    severity: int
    severity_word: str
    advisory: Advisory
    economics: Economics
    recommendation: str
    spoken_advice: str
    language: str
    topk: List[TopK]
    model_version: str


class ReportIn(BaseModel):
    label: str
    crop: str
    severity: int
    lat: float
    lon: float
    village: Optional[str] = None


class ReportOut(BaseModel):
    id: int
    label: str
    crop: str
    severity: int
    lat: float
    lon: float
    village: Optional[str]
    risk: str
    created_at: str


class MapStats(BaseModel):
    reports_total: int
    villages: int
    alerts: int
    reports: List[ReportOut]
