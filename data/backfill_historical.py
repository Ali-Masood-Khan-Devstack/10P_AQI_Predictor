import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, BACKFILL_DAYS, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def fetch_city_historical(city_key, days=BACKFILL_DAYS):
    """Fetches 5 years of historical weather & 9 pollution parameters from Open-Meteo API."""
    city_info = CITIES[city_key]
    out_dir = os.path.join(os.path.dirname(__file__), "raw_backfill")
    csv_path = os.path.join(out_dir, f"{city_key}_5yr_backfill.csv")

    if os.path.exists(csv_path):
        print(f"Loading cached 5-year CSV dataset for {city_info['name']}...")
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    print(f"\n--- Backfilling {days} Days (5 Years) for {city_info['name']} ---")
    cache_session = requests_cache.CachedSession('.cache', expire_after=86400)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    now_utc = datetime.now(timezone.utc)
    end_date = now_utc.date()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto"
    }

    w_resp = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            **params,
            "hourly": ["temperature_2m", "relative_humidity_2m", "windspeed_10m", "winddirection_10m", "uv_index"]
        }
    )[0]

    a_resp = openmeteo.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            **params,
            "hourly": ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth"]
        }
    )[0]

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

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Successfully extracted & saved {len(df)} hourly rows for {city_info['name']}.")
    return df

def upload_to_hopsworks(city_key, df):
    if not HOPSWORKS_API_KEY:
        print(f"[INFO] HOPSWORKS_API_KEY not set. Using local CSV backup.")
        return

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
        aqi_fg.insert(df)
        print(f"[SUCCESS] Uploaded {len(df)} rows to Hopsworks Feature Group: {fg_name}")
    except Exception as e:
        print(f"[WARNING] Hopsworks upload notice for {city_key}: {e}")

def run_backfill_all_cities():
    print("=== Starting 5-Year Historical Backfill for 5 Pakistan Cities ===")
    for city_key in CITIES:
        df = fetch_city_historical(city_key, days=BACKFILL_DAYS)
        upload_to_hopsworks(city_key, df)

if __name__ == "__main__":
    run_backfill_all_cities()
