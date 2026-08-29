import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# Optional TensorFlow / Keras for Deep Learning LSTM
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES, MODELS_DIR
from data.backfill_historical import fetch_city_historical
from scripts.feature_engineering import engineer_features

def build_tensorflow_lstm_wrapper(input_dim, output_dim=3):
    """Builds a Keras Sequential LSTM Neural Network wrapped for scikit-learn interface."""
    if not HAS_TENSORFLOW:
        return None

    class KerasLSTMWrapper:
        def __init__(self, input_dim, output_dim=3, epochs=10, batch_size=64):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None

        def fit(self, X, y):
            X_mat = np.array(X)
            y_mat = np.array(y)
            X_3d = X_mat.reshape((X_mat.shape[0], 1, X_mat.shape[1]))
            
            inputs = tf.keras.Input(shape=(1, self.input_dim))
            x = LSTM(32, activation='relu', return_sequences=False)(inputs)
            x = Dropout(0.2)(x)
            outputs = Dense(self.output_dim)(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X_3d, y_mat, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict(self, X):
            X_mat = np.array(X)
            X_3d = X_mat.reshape((X_mat.shape[0], 1, X_mat.shape[1]))
            return self.model.predict(X_3d, verbose=0)

    return KerasLSTMWrapper(input_dim=input_dim, output_dim=output_dim)

def train_city_models(city_key):
    city_info = CITIES[city_key]
    print(f"\n=======================================================")
    print(f"  Multi-Model Training Engine for {city_info['name']}")
    print(f"=======================================================")

    raw_df = fetch_city_historical(city_key, days=365)
    feat_df = engineer_features(raw_df)

    target_cols = ["target_h24", "target_h48", "target_h72"]
    feat_df = feat_df.dropna(subset=target_cols)

    exclude_cols = target_cols + ["datetime", "city"]
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

    X = feat_df[feature_cols].fillna(0)
    y = feat_df[target_cols].fillna(0)

    split_idx = int(len(feat_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Dataset Split: Train={len(X_train)} samples, Test={len(X_test)} samples across {len(feature_cols)} features.")

    # Candidate ML Models Competition
    models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42))
    }

    if HAS_TENSORFLOW:
        lstm_model = build_tensorflow_lstm_wrapper(input_dim=len(feature_cols))
        if lstm_model:
            models["TensorFlow_LSTM"] = lstm_model

    best_model = None
    best_name = ""
    best_r2 = -float("inf")
    summary = []

    print("\n--- Running Model Evaluation Tournament ---")
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2_list = [r2_score(y_test.iloc[:, i], preds[:, i]) for i in range(3)]
            rmse_list = [root_mean_squared_error(y_test.iloc[:, i], preds[:, i]) for i in range(3)]
            mae_list = [mean_absolute_error(y_test.iloc[:, i], preds[:, i]) for i in range(3)]

            avg_r2 = float(np.mean(r2_list))
            avg_rmse = float(np.mean(rmse_list))
            avg_mae = float(np.mean(mae_list))

            summary.append({
                "model": name, "avg_r2": avg_r2, "avg_rmse": avg_rmse, "avg_mae": avg_mae
            })

            print(f"Model: {name:<16} | R2: {avg_r2:.4f} | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f}")

            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_name = name
                best_model = model
        except Exception as e:
            print(f"[WARNING] Model {name} evaluation failed: {e}")

    print(f"\n[CHAMPION WINNER] for {city_info['name']}: {best_name} (Avg R2: {best_r2:.4f})")

    # Retrain Champion Model on full dataset and save artifact
    print(f"Retraining {best_name} on full city dataset...")
    best_model.fit(X, y)

    city_model_path = os.path.join(MODELS_DIR, f"{city_key}_model.pkl")
    with open(city_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[SUCCESS] Production model saved locally to: {city_model_path}")

    return best_model, summary

def train_all_cities():
    print("=== Starting Multi-City Multi-Model ML Training Pipeline ===")
    results = {}
    for city_key in CITIES:
        model, summary = train_city_models(city_key)
        results[city_key] = summary
    print("\n[COMPLETE] All 5 City Models Trained & Saved Successfully!")
    return results

if __name__ == "__main__":
    train_all_cities()
