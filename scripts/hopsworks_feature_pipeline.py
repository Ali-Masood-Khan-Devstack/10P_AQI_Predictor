import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME
from scripts.feature_engineering import engineer_features

def fetch_hourly_city_data(city_key):
    """Fetches latest hourly payload for a city from Open-Meteo API."""
    city_info = CITIES[city_key]
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=7)).date()
    end_date = now_utc.date()

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto"
    }

    w_resp = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params={
        **params, "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "winddirection_10m", "uv_index"]
    })[0]

    a_resp = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        **params, "hourly": ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth"]
    })[0]

    h_w = w_resp.Hourly()
    time_range = pd.date_range(
        start=pd.to_datetime(h_w.Time(), unit="s", utc=True),
        end=pd.to_datetime(h_w.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h_w.Interval()),
        inclusive="left"
    )

    df = pd.DataFrame({
        "city": city_key,
        "datetime": time_range,
        "temperature": h_w.Variables(0).ValuesAsNumpy(),
        "humidity": h_w.Variables(1).ValuesAsNumpy(),
        "wind_speed": h_w.Variables(2).ValuesAsNumpy(),
        "wind_dir": h_w.Variables(3).ValuesAsNumpy(),
        "uv_index": h_w.Variables(4).ValuesAsNumpy(),
        "pm2_5": a_resp.Hourly().Variables(0).ValuesAsNumpy(),
        "pm10": a_resp.Hourly().Variables(1).ValuesAsNumpy(),
        "co": a_resp.Hourly().Variables(2).ValuesAsNumpy(),
        "no2": a_resp.Hourly().Variables(3).ValuesAsNumpy(),
        "so2": a_resp.Hourly().Variables(4).ValuesAsNumpy(),
        "o3": a_resp.Hourly().Variables(5).ValuesAsNumpy(),
        "dust": a_resp.Hourly().Variables(6).ValuesAsNumpy(),
        "aod": a_resp.Hourly().Variables(7).ValuesAsNumpy()
    })
    return df

def run_hourly_pipeline():
    print("=== Running Hourly Multi-City Feature Pipeline ===")
    for city_key in CITIES:
        print(f"\nProcessing {CITIES[city_key]['name']}...")
        raw_df = fetch_hourly_city_data(city_key)
        feat_df = engineer_features(raw_df)
        
        if HOPSWORKS_API_KEY:
            try:
                import hopsworks
                project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
                fs = project.get_feature_store()
                fg_name = CITIES[city_key]["fg_name"]
                aqi_fg = fs.get_or_create_feature_group(
                    name=fg_name,
                    version=1,
                    primary_key=["city", "datetime"],
                    event_time="datetime",
                    time_travel_format="HUDI",
                    description=f"5-Year AQI & Weather Features for {CITIES[city_key]['name']}"
                )
                aqi_fg.insert(feat_df.tail(1))
                print(f"[SUCCESS] Pushed latest feature row for {city_key} to Hopsworks FG: {fg_name}")
            except Exception as e:
                print(f"[WARNING] Hopsworks push failed: {e}")
        else:
            print(f"[INFO] Latest feature calculated for {city_key} (PM2.5: {feat_df['pm2_5'].iloc[-1]:.1f} ug/m3).")

if __name__ == "__main__":
    run_hourly_pipeline()
