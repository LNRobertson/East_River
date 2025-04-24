![image](https://github.com/user-attachments/assets/cbe5e2f7-2fbc-48f0-999c-fbdfacc7e890)
 

## East River General Load Prediction POC 

Contributors:
- [Lindsey Robertson](https://linkedin.com/in/lnrobertson)
- [Aman Singh](https://www.linkedin.com/in/iamasr999)

## Objectives 

This project develops an AI-powered load forecasting model for East River General that aims to: 

- ✅Predict Estimated Online Load (MW) for the next 24–72 hours using historical SCADA and regional weather data. 
- ✅Inform settings for the Control Threshold (MW)
- ✅Incorporate periodic or regularly updated weather data sources to improve forecast accuracy. (Roadmap)

Business Goal: Reduce manual load control interventions and optimize grid stability while preventing costly peak demand charges—even with limited resources. 

- Presentation here: [slide deck here](https://docs.google.com/presentation/d/1pJottihB1ASlbE_PxFcr8F8a7vs-HiHBUzuatYwBiy0/edit?usp=sharing)


## Problem Statement 

Operators manually control load (Load_Control_MW) when Estimated_OnLine_Load_MW approaches the Control Threshold (Control_Threshold_MW). 

- ❌ Existing forecasting models are not accurate enough, leading to unexpected peaks and last-minute interventions. 
- ❌ Traditional methods do not capture dependencies well and leads to gaps in control threshold to online load leading to costly inefficiencies in managing the grid for the business and customers. 
- ❌ Weather significantly affects load, but previous methods struggled to incorporate it effectively. 


## 📁 Project organization 

East_River/
├─ [data/](./data/)  
│  └─ [sample/](./data/sample/)  
├─ [schema/](./schema/)  
│  └─ [generated_schema.json](./schema/generated_schema.json)  
├─ [notebooks/](./notebooks/)
│  ├─ [EDA/](./notebooks/EDA/)
│  │─ [feature_engineering/](./notebooks/feature_egineering/)
│  └─ [modeling/](./notebooks/modeling/)  
│     ├─ [baseline_xgboost](./notebooks/modeling/baseline xgboost/)  
│     ├─ [experimentation](./notebooks/modeling/experimentation/)  
│     └─ [lstm](./notebooks/modeling/lstm)  
│     └─ [wrangling](./notebooks/modeling/wrangling/)
├─ [scripts/](./scripts/)  
│  ├─ [train_lstm.py](./scripts/train_lstm.py)  
│  ├─ [predict_lstm.py](./scripts/predict_lstm].py)  
│  ├─ [predict.py](./scripts/predict.py)  
│  └─ [sample_data.py](./scripts/sample_data.py)  
├─ [models/](./models/)  
│  ├─ [multi_lstm/](./models/multi_lstm)  
│  └─ [xgb_multi_horizon_models/](./models/xgb_multi_horizon_models)  
├─ [environment.yml](./environment.yml)  
└─ [README.md](./README.md)



## 📦 Dependencies

All required Python packages and versions are specified in:

- [environment.yml](./environment.yml) (for Conda users)
- [requirements.txt](./requirements.txt) (for pip users)
- 

## 🚀 Setup 

```bash
# Using Conda (recommended)
conda env create -f [environment.yml](http://_vscodecontentref_/0)
conda activate er_lstm

# Or using pip
pip install -r [requirements.txt](http://_vscodecontentref_/1)


## ▶️ Run the project 
### 1) Install dependencies
```bash
conda env create -f environment.yml
pip install -r requirements.txt
conda activate er_lstm
```

### 2) Prepare data  
– Place your full HDF5 under `data/training/east_river_training-v2.h5` (this folder is in `.gitignore`).  
– Or generate a tiny sample for quick experiments:
```bash
python scripts/sample_data.py
```

### 3) (Optional) Generate & Validate JSON Schema  
```bash
python scripts/inspect_schema.py    # writes schema/generated_schema.json
python scripts/validate_schema.py   # checks one record against it
```

### 4) Train the model  
```bash
python scripts/train_lstm.py
```

### 5) Run inference  
Point `DATA_PATH` in `scripts/predict_lstm.py` at your new file and then:
```bash
python scripts/predict_lstm.py
```
Outputs will land in `outputs/predictions.csv`.

### 6) Launch notebooks  
```bash
jupyter notebook notebooks/
```
Open any notebook in `notebooks/` (EDA, feature_engineering, modeling) to step through analyses and visualizations.



## 🛠️ Technology Stack

**Programming**
- Python
- Jupyter Notebooks
- VS Code
- GitHub Copilot

**ML Libraries**
- scikit-learn
- xgboost
- Keras (with TensorFlow backend)
- NumPy
- pandas

**Data Integration**
- CSV, HDF5
- ipynb (notebooks)
- JSON Schema (for validation)
- Automated data ingestion in `predict.py`

**Deployment**
- Python scripts
- joblib & pickle (model serialization)



## ✅ Solution

Integrate historical SCADA load data, weather observations, and external event calendars to deliver robust, multi-horizon load forecasts for East River General. The workflow is designed for reproducibility, operational efficiency, and extensibility.

**1. Data Integration & Feature Engineering**
- Merge historical SCADA and weather data, aligning on timestamp and location.
- Fill missing weather values using forward-fill and alternate sources.
- Engineer features: lagged loads, rolling weather stats, calendar/holiday/event flags.
- Construct multi-horizon targets (24h, 48h, 72h ahead) for supervised learning.

**2. Modeling**
- Baseline: XGBoost and LightGBM models for fast, interpretable forecasting.
- Advanced: LSTM neural network to capture sequential dependencies and long-range patterns.
- All models trained with strong regularization, early stopping, and robust validation (60/20/20 chronological split).

**3. Validation & Evaluation**
- Hold-out test set (final 20%) used for unbiased MAE/RMSE scoring.
- Residual analysis and feature importance to guide further improvements.
- Comparison with existing Control Threshold logic and persistence baselines.

**4. Operationalization**
- Model pipelines exported as Python scripts for reproducible training and inference.
- Predict script validates new data against a JSON schema, applies scaling, and generates 24/48/72h forecasts.
- Outputs are ready for integration into dashboards or automated operator alerts.

**5. Extensibility & Roadmap**
- Modular design allows easy retraining with new data or features.
- Future enhancements: advanced models (TFT, GNNs), uncertainty quantification, integration of market/renewable data, and real-time API deployment.

This approach balances accuracy, transparency, and maintainability—delivering actionable forecasts to reduce costly peak events and manual interventions.

 


## 📊 Data Sources  

1️⃣ SCADA Data (Load & Control Information): 

- Source: East River General SCADA System 

- Key Variables: 

 - OnLine_Load_MW: Actual measured online load. 

 - Load_Control_MW: Load manually controlled by operators. 

 - Estimated_OnLine_Load_MW: Prediction target variable. 

 - Control_Threshold_MW: Threshold above which manual intervention is triggered. 

Note: Ingest both historical and live SCADA data to ensure real-time forecasting and model validation. 

2️⃣ Weather Data: 

- Historical Weather Data: 

 - Source: Provided by East River 

 - Usage: Used for training, validating, and feature engineering (e.g., lagged weather effects, seasonal trends) to understand how past weather events impacted load. 

 - Key variables like; temperature, humidity, precipitation. 

Forecast Weather Data: 

- Source: Free weather forecast APIs (e.g., Open-Mateo API Documentation, visual crossing) 

- Key Features: temperature, humidity, precipitation, etc., used to predict future conditions. 

 

3️⃣ External Factors – Public Holidays & Events: 

- Source: Public APIs (e.g., HolidayAPI) or available CSV calendars. 

- Key Features: 

 - Binary/categorical flags to indicate weekends, public holidays, or significant local events (Roadmap). These flags help detect usage spikes that are not solely weather-dependent. 

 

## 🏗 Model Architecture 

1️⃣ Baseline Models  
   - **XGBoost (multi‑horizon)**  
     • Fast, interpretable gradient‑boosted trees for simultaneous 24/48/72 h forecasts.  

2️⃣ Neural Network Models  
   - **Multi‑Horizon LSTM**  
     • Keras Sequential: LSTM(32) → Dropout(0.5) → Dense(16) → Dense(3)  
     • Strong L2 weight decay, EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.2, patience=3)  
     • Trained on 60/20/20 split via TimeseriesGenerator for robust sequence learning.  

3️⃣ Future Exploration  
   - **GRU / Bi‑LSTM** for lightweight recurrent alternatives  
   - **Temporal Fusion Transformer (TFT)** for long‑range attention  
   - **Graph Neural Networks (GNNs)** to model spatial/grid dependencies  

Why This Approach?  
✔ Leverages a quick, reliable tree‑based baseline alongside a regularized LSTM for temporal patterns  
✔ Combines weather, calendar and event features for resilient, multi‑horizon forecasts  
✔ Keeps the pipeline modular and resource‑efficient; advanced deep models (TFT, GNN) are in our roadmap for next‑stage improvements  

 

## 🧹 Preprocessing

1️⃣ Data Ingestion & Parsing  
   • Load SCADA (`east_river_training-v2.h5`) via `pd.read_hdf`.  
   • Parse `local_time` and `last_control_time` to `datetime`.

2️⃣ Historical Weather Integration  
   • Merge NOAA observations (`add_NOAA_weather.ipynb`) and Open‑Mateo forecasts (`add_Open_Mateo_Weather.ipynb`).  
   • Resample to hourly intervals & forward/back‑fill missing values (`all_hist_weather_time-interpolate.ipynb`).

3️⃣ Event & Calendar Features  
   • Load public holiday CSV and flag weekends/holidays (`add_holiday_scada_weather.ipynb`).  

4️⃣ Feature Engineering  
   • Compute lag features (1h, 6h, 24h) and rolling statistics (mean, std) in `feature_engineering.ipynb`.  
   • Add control‐action flags, drop unused columns (`OnLine_Load_MW`, `Load_Control_MW`).

5️⃣ Scaling & Sequencing  
   • Fit and persist `StandardScaler` to `multi_lstm_scaler.pkl`.  
   • Generate 60/20/20 train/val/test sequences with `TimeseriesGenerator` (SEQ_LEN=48).
 
 

## 🔍 Research 

Inspired by insights from recent research on AI forecasting and dynamic data integration in grid demand management. 
- Hybrid forecasting with weather-lag correlations 
- Advanced deep learning time-series architectures 

## ⚠️ Challenges
- Data Quality & Gaps

- Incomplete SCADA and weather time series; required interpolation and imputation (e.g., KNN for missing load, forward-fill for weather).
30-min intervals and diverse sources needed careful timestamp alignment.
Widespread zeros and missing values required robust cleaning and validation.
Feature selection was challenging due to uncertainty about which engineered features (lags, peaks, rolling stats) best enhance the dataset.
Domain & Time Constraints

- Limited weather forecasting experience on the team.
Only ~5 hours/week available for project work.
Weather & Event Variability

- Extreme weather, sensor reliability, and microclimate effects increase prediction complexity.
Uncertainty in capturing event-driven or behavioral load spikes.
Integration Complexity

- Merging diverse data sources (SCADA, weather, events) with different formats and update frequencies.
Modeling & Accuracy

- Balancing computational efficiency with the need for accurate, real-time forecasts.
Setting optimal thresholds to minimize false positives/negatives in peak detection.
Operationalization

- Managing environment dependencies (e.g., GPU/VM issues).
Ensuring reproducibility and maintainability for future retraining and scaling.


## ❓ Key Open Questions

- How can we best quantify uncertainty (e.g. via prediction intervals or Bayesian methods) to enhance the reliability of peak flagging?  
- What is the incremental value of including public holiday/event data versus weather data alone?  
- How robust is our model to noisy or missing SCADA/weather inputs, and what imputation strategies minimize degradation?  
- How often should the model be retrained to adapt to seasonal shifts or long‑term grid changes without overfitting?   
- How transferable is this forecasting pipeline to other substations or regions with different load/weather dynamics?  
- Can we incorporate real‑time streaming data and automated alerts into a production workflow while maintaining reliability?  


 
## 🗺️ Possible Next Steps & Roadmap

1️⃣ Develop ETL Pipelines  
   • Architect end‑to‑end Extract‑Transform‑Load jobs to ingest SCADA, historical weather data (ex. Open‑Meteo), and holiday/event calendars. Explore addional untapped sources. 
   • Implement data quality checks, missing‑value imputation, timestamp alignment and persisting cleaned data to HDF5 or a centralized data store.  
   • Build incremental update logic and a sample‑data generator (`scripts/sample_data.py`) for lightweight onboarding.

2️⃣ Deepen Exploratory Data Analysis  

3️⃣ Educated Feature Engineering & Selection  
   • Automate creation of domain‑informed covariates: dynamic weather lags, rolling statistics, control‑action deltas.  
   • Integrate SHAP‑based importance ranking and recursive feature elimination for targeted pruning.

4️⃣ Integrate Live Forecast Feeds  
   • Connect to a real‑time weather API for dynamic future feature injection.  
   • Schedule periodic refresh of exogenous data ahead of each retrain/inference cycle.

5️⃣ Model Fine‑Tuning & Validation  
   • Execute hyperparameter sweeps for XGBoost and LSTM.  
   • Validate across multiple hold‑out windows; measure MAE, RMSE, pinball loss and operational KPIs (peak‑flag accuracy, cost savings).

6️⃣ Advanced Model Prototyping  
   • Prototype GRU/Bi‑LSTM and Temporal Fusion Transformer (TFT) for long‑range dependencies.  
   • Investigate Graph Neural Networks (GNNs) to capture spatial correlations between substations.

7️⃣ Productionization & Automation  
   • Containerize training & inference scripts; establish CI/CD pipelines.  
   • Orchestrate data ingestion, model retraining and prediction jobs.  
   • Expose real‑time forecasts through a REST API or integrate into operator dashboards.

8️⃣ Economic Impact & Monitoring  
   • Quantify value: compare manual intervention costs, peak‑penalty fees and energy wastage against model‑driven forecasts.  
   • Define a retraining cadence (e.g. monthly or drift‑triggered) and monitor data/model drift with automated alerts.

9️⃣ Continuous Improvement & Roadmap  
   • Revisit advanced architectures (TFT, GNN, hybrid ensembles) as resource and performance needs evolve.  
   • Incorporate renewable injection and market price signals to further boost forecast robustness.

✨ This roadmap guides us from POC through scalable production, ensuring data integrity, model accuracy, and operational efficiency while laying the foundation for future innovation.  

 

📬 Contact 

For questions or collaboration inquiries, please contact: 
[Contact Information] 

 
✨ Acknowledgements

- Code suggestions and snippets generated with AI assistance from GitHub Copilot in VSCode.
