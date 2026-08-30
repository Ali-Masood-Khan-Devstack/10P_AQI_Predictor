import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

def verify_hopsworks():
    print("=== Verifying Live Hopsworks Cloud Resources ===")
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("\n1. Feature Groups Status:")
    for city_key in CITIES:
        fg_name = CITIES[city_key]["fg_name"]
        try:
            fg = fs.get_feature_group(fg_name, version=1)
            df = fg.read()
            print(f"   ✅ {fg_name}: {len(df):,} rows, {len(df.columns)} features. Latest timestamp: {df['datetime'].max()}")
        except Exception as e:
            print(f"   ⚠️ {fg_name}: Error reading group ({e})")

    print("\n2. Feature Views Status:")
    for city_key in CITIES:
        fv_name = CITIES[city_key]["fv_name"]
        try:
            fv = fs.get_feature_view(fv_name, version=1)
            print(f"   ✅ {fv_name}: Registered (Version 1)")
        except Exception as e:
            print(f"   ⚠️ {fv_name}: Error ({e})")

    print("\n3. Model Registry Status:")
    models = mr.get_models()
    for m in models:
        print(f"   ✅ Model Registered: '{m.name}' (Version {m.version})")

if __name__ == "__main__":
    verify_hopsworks()
