import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pymongo import MongoClient

# Add parent directory to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LATITUDE, LONGITUDE, MONGO_URI, DB_NAME, RAW_COLLECTION

def get_mongo_collection():
    if not MONGO_URI:
        print("[WARNING] MONGO_URI not found in environment. Data will be printed but not persisted.")
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        return db[RAW_COLLECTION]
    except Exception as e:
        print(f"[WARNING] MongoDB connection failed: {e}")
        return None

def fetch_raw_data(days=730):
    """Fetch weather and air quality data for Islamabad from Open-Meteo API."""
    print(f"Fetching Islamabad weather and air quality data for last {days} days...")
    
    # API Client Setup with cache and retry logic
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    now_utc = datetime.now(timezone.utc)
    end_date = now_utc.date()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto"
    }

    # Fetch Weather Archive Data
    w_resp = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            **params,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "windspeed_10m",
                "winddirection_10m"
            ]
        }
    )[0]

    # Fetch Air Quality Archive Data
    a_resp = openmeteo.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            **params,
            "hourly": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "dust"
            ]
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
        "datetime": time_range,
        "temperature": h_w.Variables(0).ValuesAsNumpy(),
        "humidity": h_w.Variables(1).ValuesAsNumpy(),
        "wind_speed": h_w.Variables(2).ValuesAsNumpy(),
        "wind_dir": h_w.Variables(3).ValuesAsNumpy(),
        "pm2_5": a_resp.Hourly().Variables(0).ValuesAsNumpy(),
        "pm10": a_resp.Hourly().Variables(1).ValuesAsNumpy(),
        "co": a_resp.Hourly().Variables(2).ValuesAsNumpy(),
        "no2": a_resp.Hourly().Variables(3).ValuesAsNumpy(),
        "so2": a_resp.Hourly().Variables(4).ValuesAsNumpy(),
        "o3": a_resp.Hourly().Variables(5).ValuesAsNumpy(),
        "dust": a_resp.Hourly().Variables(6).ValuesAsNumpy()
    })

    return df

def run_data_extraction():
    raw_col = get_mongo_collection()
    
    if raw_col is not None:
        count = raw_col.count_documents({})
        days_to_fetch = 730 if count == 0 else 3
        print(f"Database has {count} records. Fetching last {days_to_fetch} days...")
    else:
        days_to_fetch = 7  # Fetch 7 days for local inspection if Mongo not present

    df = fetch_raw_data(days=days_to_fetch)
    print(f"Extracted {len(df)} records for Islamabad (Lat: {LATITUDE}, Lon: {LONGITUDE}).")

    if raw_col is None:
        print("Data extraction complete. Preview:")
        print(df.tail())
        return df

    now_utc = datetime.now(timezone.utc)
    existing_times = set()
    for dt in raw_col.distinct("datetime"):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        existing_times.add(dt.astimezone(timezone.utc).isoformat())

    records_to_insert = []
    for record in df.to_dict("records"):
        dt_obj = record["datetime"]
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        
        current_time_iso = dt_obj.isoformat()
        if current_time_iso not in existing_times and dt_obj <= now_utc:
            record["datetime"] = dt_obj
            records_to_insert.append(record)

    if records_to_insert:
        raw_col.insert_many(records_to_insert)
        print(f"Success! Inserted {len(records_to_insert)} new records into MongoDB.")
    else:
        print("Everything is up to date. 0 records inserted.")

    return df

if __name__ == "__main__":
    run_data_extraction()
