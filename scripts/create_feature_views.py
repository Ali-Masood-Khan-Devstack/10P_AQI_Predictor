import os
import sys
import hopsworks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def create_all_feature_views():
    if not HOPSWORKS_API_KEY:
        print("[ERROR] HOPSWORKS_API_KEY not configured.")
        return

    print("=== Connecting to Hopsworks Cloud to Register Feature Views ===")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()

    for city_key, city_info in CITIES.items():
        fg_name = city_info["fg_name"]
        fv_name = city_info["fv_name"]
        print(f"\nProcessing Feature View for {city_info['name']}...")
        
        try:
            fg = fs.get_feature_group(name=fg_name, version=1)
            query = fg.select_all()
            
            fv = fs.get_or_create_feature_view(
                name=fv_name,
                version=1,
                query=query,
                description=f"Point-in-time Feature View for {city_info['name']} AQI Forecast"
            )
            print(f"[SUCCESS] Feature View registered: {fv.name} (Version 1)")
        except Exception as e:
            print(f"[WARNING] Feature View creation for {city_key} notice: {e}")

if __name__ == "__main__":
    create_all_feature_views()
