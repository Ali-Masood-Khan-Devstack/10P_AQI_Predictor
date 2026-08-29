import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, DEFAULT_CITY, MODELS_DIR
from api.schemas import ForecastResponse, HorizonForecast, ExplainabilityResponse, FeatureContribution
from scripts.hopsworks_feature_pipeline import fetch_hourly_city_data
from scripts.feature_engineering import engineer_features
from scripts.shap_explainer import calculate_shap_contributions

app = FastAPI(
    title="Multi-City Pakistan AQI Forecasting REST API",
    description="Serverless MLOps REST API serving 24h, 48h, & 72h AQI predictions across 5 cities in Pakistan.",
    version="2.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_aqi(pm25):
    """EPA standard formula for PM2.5 -> AQI."""
    if pm25 < 0: return 0
    if pm25 <= 12.0: return round(((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0)
    elif pm25 <= 35.4: return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4: return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4: return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4: return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    else: return 500

def get_aqi_info(aqi_val):
    if aqi_val <= 50: return ("Good", "🌿 Air quality is satisfactory. Enjoy outdoor activities!")
    elif aqi_val <= 100: return ("Moderate", "⚠️ Air quality is acceptable. Sensitive groups should limit prolonged outdoor exertion.")
    elif aqi_val <= 150: return ("Unhealthy for Sensitive Groups", "😷 Sensitive groups should wear masks and reduce outdoor activity.")
    elif aqi_val <= 200: return ("Unhealthy", "🏠 Everyone may experience health effects. Wear N95 masks outdoors.")
    elif aqi_val <= 300: return ("Very Unhealthy", "🚨 Health alert: Avoid physical outdoor activities.")
    else: return ("Hazardous", "☢️ Emergency conditions. Remain indoors.")

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to Multi-City Pakistan AQI Forecasting REST API",
        "docs": "/docs",
        "supported_cities": list(CITIES.keys())
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cities": list(CITIES.keys())
    }

@app.get("/cities", tags=["Metadata"])
def get_supported_cities():
    return CITIES

@app.get("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def get_aqi_forecast(city: str = Query(DEFAULT_CITY, description="City key: islamabad, lahore, rawalpindi, quetta, karachi")):
    city_key = city.lower()
    if city_key not in CITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported city '{city}'. Valid choices: {list(CITIES.keys())}")

    city_info = CITIES[city_key]

    # Load City Model
    model_path = os.path.join(MODELS_DIR, f"{city_key}_model.pkl")
    model = None
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Fetch live features
    raw_df = fetch_hourly_city_data(city_key)
    feat_df = engineer_features(raw_df)
    latest_feat = feat_df.iloc[-1].to_dict()

    if model is not None:
        drop_cols = ['_id', 'datetime', 'city', 'target_h24', 'target_h48', 'target_h72']
        input_df = pd.DataFrame([latest_feat]).drop(columns=drop_cols, errors='ignore')
        try:
            preds = np.maximum(model.predict(input_df).flatten(), 0)
        except Exception:
            cur_pm = latest_feat.get("pm2_5", 35.0)
            preds = np.array([cur_pm * 1.02, cur_pm * 1.05, cur_pm * 0.98])
    else:
        cur_pm = latest_feat.get("pm2_5", 35.0)
        preds = np.array([cur_pm * 1.02, cur_pm * 1.05, cur_pm * 0.98])

    cur_aqi = calculate_aqi(latest_feat.get("pm2_5", 35.0))
    horizons = ["24h", "48h", "72h"]
    forecasts = []
    for i in range(3):
        f_pm = float(preds[i])
        f_aqi = calculate_aqi(f_pm)
        cat, rec = get_aqi_info(f_aqi)
        forecasts.append(HorizonForecast(
            horizon=horizons[i],
            pm25=round(f_pm, 2),
            aqi=f_aqi,
            category=cat,
            recommendation=rec
        ))

    return ForecastResponse(
        city=city_info["name"],
        latitude=city_info["lat"],
        longitude=city_info["lon"],
        current_aqi=cur_aqi,
        forecast=forecasts,
        model_version="v2.0-champion",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/explain", response_model=ExplainabilityResponse, tags=["Explainable AI"])
def get_shap_attributions(city: str = Query(DEFAULT_CITY, description="City key: islamabad, lahore, rawalpindi, quetta, karachi")):
    city_key = city.lower()
    if city_key not in CITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported city '{city}'")

    shap_df = calculate_shap_contributions(city_key)
    top_features = [
        FeatureContribution(feature=str(row["feature"]), importance=float(row["importance"]))
        for _, row in shap_df.head(6).iterrows()
    ]
    return ExplainabilityResponse(city=CITIES[city_key]["name"], top_features=top_features)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
