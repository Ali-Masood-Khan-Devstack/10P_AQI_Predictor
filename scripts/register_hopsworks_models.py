import os
import sys
import pickle
import hopsworks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, MODELS_DIR, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def register_models_to_hopsworks():
    if not HOPSWORKS_API_KEY:
        print("[ERROR] HOPSWORKS_API_KEY not configured.")
        return

    print("=== Connecting to Hopsworks Model Registry ===")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    mr = project.get_model_registry()

    for city_key, city_info in CITIES.items():
        model_filename = f"{city_key}_model.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(model_path):
            model_filename = f"{city_key}_aqi_model.pkl"
            model_path = os.path.join(MODELS_DIR, model_filename)

        if os.path.exists(model_path):
            model_name = f"aqi_{city_key}_model"
            print(f"\nRegistering {model_name} in Hopsworks Model Registry...")
            try:
                hw_model = mr.python.create_model(
                    name=model_name,
                    description=f"Champion Machine Learning AQI Forecast Model for {city_info['name']}"
                )
                hw_model.save(model_path)
                print(f"[SUCCESS] Registered {model_name} in Hopsworks Model Registry!")
            except Exception as e:
                print(f"[NOTICE] Model registration result for {city_key}: {e}")
        else:
            print(f"[WARNING] Model file not found for {city_key} at {model_path}")

if __name__ == "__main__":
    register_models_to_hopsworks()
