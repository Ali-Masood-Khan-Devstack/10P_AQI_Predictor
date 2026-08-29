# 🏔️ Islamabad AQI Predictor (`test 10p AQI`)

An end-to-end production MLOps system built specifically for **Islamabad, Pakistan** (Lat: `33.6844° N`, Lon: `73.0479° E`) to forecast PM2.5 concentrations and Air Quality Index (AQI) for **24-hour, 48-hour, and 72-hour horizons**.

---

## 📂 Project Structure

```
test 10p AQI/
├── .env.example                    # Template for environment credentials
├── requirements.txt                # Streamlit app and general dependencies
├── requirements-ci.txt                 # CI/CD and automation script dependencies
├── README.md                       # Documentation and usage guide
├── config/
│   ├── __init__.py
│   └── settings.py                 # Centralized configuration (Islamabad coords, DB, MLflow)
├── scripts/
│   ├── __init__.py
│   ├── data_extraction.py          # Fetches weather & pollution data for Islamabad from Open-Meteo
│   ├── feature_engineering.py      # Lags, rolling stats, wind vectors, cyclical time features
│   ├── model_train.py              # MultiOutput XGBoost/LightGBM/RandomForest model trainer
│   └── promote_model.py            # Model promotion & alias manager
├── automation_scripts/
│   ├── __init__.py
│   ├── hourly_data_pipeline.py     # Automation entry point: Data extraction + Feature engineering
│   └── daily_model_pipeline.py     # Automation entry point: Model training + Promotion
├── app/
│   └── app.py                      # Interactive Streamlit forecast dashboard
└── models/
    └── .gitkeep                    # Directory for local fallback model artifacts
```

---

## 🔑 Credentials & Integration Setup

Create a `.env` file in the `test 10p AQI` directory by copying `.env.example`:

```bash
cp .env.example .env
```

### 1. MongoDB Atlas (Default Feature Store)
- **`MONGO_URI`**: MongoDB connection string (e.g. `mongodb+srv://<user>:<password>@<cluster>.mongodb.net/`).
- **`DB_NAME`**: Set to `islamabad_aqi_db`.
- *Note:* If `MONGO_URI` is not provided, the Streamlit app automatically fetches live data directly from Open-Meteo API as a fallback.

### 2. Hopsworks Feature Store (Optional)
If you want to use **Hopsworks** instead of MongoDB:
- **`HOPSWORKS_API_KEY`**: Your API key from [Hopsworks.ai](https://c.app.hopsworks.ai/).
- **`HOPSWORKS_PROJECT_NAME`**: Your Hopsworks project name (e.g., `islamabad_aqi`).

### 3. DagsHub / MLflow Tracking (Model Registry)
- **`MLFLOW_TRACKING_URI`**: e.g., `https://dagshub.com/<username>/<repo_name>.mlflow`.
- **`MLFLOW_TRACKING_USERNAME`**: Your DagsHub username.
- **`MLFLOW_TRACKING_PASSWORD`**: Your DagsHub access token.
- *Note:* If MLflow credentials are not provided, the pipeline automatically saves local `.pkl` model artifacts inside `models/`.

### 4. Streamlit Sharing / Cloud Deployment
When deploying to **Streamlit Cloud** or **Hugging Face Spaces**:
- Go to App Settings -> **Secrets**.
- Paste the contents of your `.env` file into the Secrets Manager.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Hourly Extraction + Feature Store Update)
Fetches historical weather and air pollution metrics for Islamabad from Open-Meteo API:
```bash
python automation_scripts/hourly_data_pipeline.py
```

### 3. Run Model Pipeline (Daily Retraining + Champion Promotion)
Trains XGBoost, LightGBM, and Random Forest models on Islamabad features and selects the best model:
```bash
python automation_scripts/daily_model_pipeline.py
```

### 4. Launch Streamlit Web Application
Start the interactive Islamabad AQI Forecast Dashboard:
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.
