import os
import sys
import pandas as pd
import hopsworks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def insert_missing_cities():
    print("=== Ingesting 5-Year Data for All Cities into Hopsworks Cloud ===")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()

    for city_key in ["karachi", "quetta", "islamabad", "lahore", "rawalpindi"]:
        city_info = CITIES[city_key]
        fg_name = city_info["fg_name"]
        csv_path = os.path.join("data", "raw_backfill", f"{city_key}_5yr_backfill.csv")

        if os.path.exists(csv_path):
            print(f"\nReading {csv_path} for {city_info['name']}...")
            df = pd.read_csv(csv_path)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df['city'] = city_key

            float_cols = ["temperature", "humidity", "wind_speed", "wind_dir", "uv_index", 
                          "pm2_5", "pm10", "co", "no2", "so2", "o3", "dust", "aod"]
            df[float_cols] = df[float_cols].astype("float64")

            try:
                fg = fs.get_or_create_feature_group(
                    name=fg_name,
                    version=1,
                    primary_key=["city", "datetime"],
                    event_time="datetime",
                    time_travel_format="HUDI",
                    description=f"5-Year AQI & Weather Features for {city_info['name']}"
                )
                print(f"Streaming {len(df):,} rows to Hopsworks Feature Group: {fg_name}...")
                fg.insert(df)
                print(f"[SUCCESS] Ingested 5-year data into Hopsworks Feature Group: {fg_name}")
            except Exception as e:
                print(f"[ERROR] Failed to ingest {city_key}: {e}")
        else:
            print(f"[WARNING] CSV file {csv_path} not found.")

if __name__ == "__main__":
    insert_missing_cities()
