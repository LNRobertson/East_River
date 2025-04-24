import json
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator

# --- CONFIG ---
DATA_PATH   = r"C:\Users\Linds\Repos\East_River\data\new_data\to_predict.h5"
SCALER_PATH = r"C:\Users\Linds\Repos\East_River\scripts\multi_lstm_scaler.pkl"
MODEL_PATH  = r"C:\Users\Linds\Repos\East_River\scripts\multi_lstm.h5"
SCHEMA_PATH = r"C:\Users\Linds\Repos\East_River\schema\generated_schema.json"
SEQ_LEN     = 48
HORIZONS    = [24,48,72]

# 1) (Optional) load schema for your own reference
with open(SCHEMA_PATH) as f:
    schema = json.load(f)
# you could inspect `schema["properties"]` here or just keep it for docs

# 2) Load new data
df = pd.read_hdf(DATA_PATH, key="df").sort_values("local_time")
# drop unneeded cols, same as training:
drop = ["local_time","last_control_time","OnLine_Load_MW",
        "Load_Control_MW","Control_Threshold_MW","location"]
X_df = df.drop(columns=drop, errors="ignore")

# 3) Scale
scaler   = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X_df.values.astype(np.float32))

# 4) Build sequences & predict
gen = TimeseriesGenerator(X_scaled, np.zeros((len(X_scaled),len(HORIZONS))),
                          length=SEQ_LEN, batch_size=256)
model = load_model(MODEL_PATH)
y_pred = model.predict(gen)

# 5) Convert to a DataFrame and save
out_df = pd.DataFrame(
    y_pred,
    columns=[f"pred_{h}h" for h in HORIZONS],
    index=df.index[SEQ_LEN:SEQ_LEN+len(y_pred)]
)
out_df.to_csv(r"C:\Users\Linds\Repos\East_River\outputs\predictions.csv")

print("✅ Predictions saved to outputs\\predictions.csv")