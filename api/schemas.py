from pydantic import BaseModel, Field
from typing import List, Optional

class HorizonForecast(BaseModel):
    horizon: str = Field(..., example="24h")
    pm25: float = Field(..., example=42.5)
    aqi: int = Field(..., example=118)
    category: str = Field(..., example="Unhealthy for Sensitive Groups")
    recommendation: str = Field(..., example="Sensitive groups should limit outdoor exertion.")

class ForecastResponse(BaseModel):
    city: str = Field(..., example="Islamabad")
    latitude: float = Field(..., example=33.6844)
    longitude: float = Field(..., example=73.0479)
    current_aqi: int = Field(..., example=118)
    forecast: List[HorizonForecast]
    model_version: str = Field(..., example="v1")
    timestamp: str = Field(..., example="2026-08-30T02:20:00Z")

class FeatureContribution(BaseModel):
    feature: str = Field(..., example="pm25_lag_24h")
    importance: float = Field(..., example=0.35)

class ExplainabilityResponse(BaseModel):
    city: str = Field(..., example="Islamabad")
    top_features: List[FeatureContribution]
