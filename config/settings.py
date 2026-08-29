import os
from dotenv import load_dotenv

load_dotenv()

# MULTI-CITY GEOLOCATION METADATA
CITIES = {
    "islamabad": {
        "name": "Islamabad",
        "lat": 33.6844,
        "lon": 73.0479,
        "fg_name": "aqi_islamabad_fg",
        "fv_name": "aqi_islamabad_fv",
        "model_name": "AQI_Predictor_Islamabad"
    },
    "lahore": {
        "name": "Lahore",
        "lat": 31.5204,
        "lon": 74.3587,
        "fg_name": "aqi_lahore_fg",
        "fv_name": "aqi_lahore_fv",
        "model_name": "AQI_Predictor_Lahore"
    },
    "rawalpindi": {
        "name": "Rawalpindi",
        "lat": 33.5989,
        "lon": 73.0441,
        "fg_name": "aqi_rawalpindi_fg",
        "fv_name": "aqi_rawalpindi_fv",
        "model_name": "AQI_Predictor_Rawalpindi"
    },
    "quetta": {
        "name": "Quetta",
        "lat": 30.1798,
        "lon": 66.9750,
        "fg_name": "aqi_quetta_fg",
        "fv_name": "aqi_quetta_fv",
        "model_name": "AQI_Predictor_Quetta"
    },
    "karachi": {
        "name": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011,
        "fg_name": "aqi_karachi_fg",
        "fv_name": "aqi_karachi_fv",
        "model_name": "AQI_Predictor_Karachi"
    }
}

DEFAULT_CITY = "islamabad"
BACKFILL_DAYS = 1825  # 5 Years of Historical Data (5 * 365)

# HOPSWORKS CONFIGURATION
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor_multi_city")

# DATABASE & MLFLOW CONFIGURATION
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("DB_NAME", "multi_city_aqi_db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MODEL_ALIAS = "champion"

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
