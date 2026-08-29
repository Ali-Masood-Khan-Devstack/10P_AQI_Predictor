import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD,
    MODEL_NAME, MODEL_ALIAS
)

def promote_model():
    """Promotes latest model version to champion alias if MLflow is configured."""
    if not MLFLOW_TRACKING_URI:
        print("[INFO] MLflow URI not configured. Local fallback model in models/ directory will be used.")
        return

    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            print(f"No versions found for model '{MODEL_NAME}'.")
            return

        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, latest_version.version)
        print(f"[SUCCESS] Promoted version {latest_version.version} of '{MODEL_NAME}' to alias '{MODEL_ALIAS}'.")

    except Exception as e:
        print(f"[WARNING] Model promotion skipped: {e}")

if __name__ == "__main__":
    promote_model()
