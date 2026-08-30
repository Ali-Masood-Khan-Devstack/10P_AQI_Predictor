import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def fetch_hourly_city_data(city_key):
    """Fetches latest real-time hourly payload for a city from Open-Meteo Forecast & Air Quality APIs in UTC."""
    city_info = CITIES[city_key]
    cache_session = requests_cache.CachedSession('.cache', expire_after=300)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "UTC"
    }

    w_resp = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params={
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
        "city": str(city_key),
        "datetime": time_range,
        "temperature": h_w.Variables(0).ValuesAsNumpy().astype("float64"),
        "humidity": h_w.Variables(1).ValuesAsNumpy().astype("float64"),
        "wind_speed": h_w.Variables(2).ValuesAsNumpy().astype("float64"),
        "wind_dir": h_w.Variables(3).ValuesAsNumpy().astype("float64"),
        "uv_index": h_w.Variables(4).ValuesAsNumpy().astype("float64"),
        "pm2_5": a_resp.Hourly().Variables(0).ValuesAsNumpy().astype("float64"),
        "pm10": a_resp.Hourly().Variables(1).ValuesAsNumpy().astype("float64"),
        "co": a_resp.Hourly().Variables(2).ValuesAsNumpy().astype("float64"),
        "no2": a_resp.Hourly().Variables(3).ValuesAsNumpy().astype("float64"),
        "so2": a_resp.Hourly().Variables(4).ValuesAsNumpy().astype("float64"),
        "o3": a_resp.Hourly().Variables(5).ValuesAsNumpy().astype("float64"),
        "dust": a_resp.Hourly().Variables(6).ValuesAsNumpy().astype("float64"),
        "aod": a_resp.Hourly().Variables(7).ValuesAsNumpy().astype("float64")
    })
    return df

def run_hourly_pipeline():
    print("=== Running Hourly Multi-City Feature Ingestion Pipeline ===")
    now_utc = pd.Timestamp.now(tz='UTC').floor('h')
    
    for city_key in CITIES:
        print(f"\nProcessing {CITIES[city_key]['name']}...")
        raw_df = fetch_hourly_city_data(city_key)
        
        # Filter for exact current UTC hour or latest past observation
        past_df = raw_df[raw_df['datetime'] <= now_utc]
        if past_df.empty:
            latest_row = raw_df.head(1)
        else:
            latest_row = past_df.tail(1)

        print(f"Ingesting latest observation timestamp: {latest_row['datetime'].iloc[-1]} for {city_key}")
        
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
                aqi_fg.insert(latest_row)
                print(f"[SUCCESS] Pushed latest hourly row for {city_key} to Hopsworks FG: {fg_name}")
            except Exception as e:
                print(f"[WARNING] Hopsworks push failed for {city_key}: {e}")
        else:
            print(f"[INFO] Latest feature calculated for {city_key} (PM2.5: {latest_row['pm2_5'].iloc[-1]:.1f} ug/m3).")

if __name__ == "__main__":
    run_hourly_pipeline()
