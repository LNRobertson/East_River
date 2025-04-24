import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import regularizers, optimizers
from keras.preprocessing.sequence import TimeseriesGenerator

def main():
    # 1) Config
    DATA_PATH   = r"C:\Users\Linds\Repos\East_River\data\training\east_river_training-v2.h5"
    SCALER_PATH = "multi_lstm_scaler.pkl"
    MODEL_PATH  = "multi_lstm.h5"
    HORIZONS    = [24, 48, 72]
    SEQ_LEN     = 48
    EPOCHS      = 50

    # 2) Load & scale
    df      = pd.read_hdf(DATA_PATH, key="df").sort_values("local_time")
    drop    = ["local_time","last_control_time","OnLine_Load_MW","Load_Control_MW",
               "Control_Threshold_MW","location"]
    X_df    = df.drop(columns=drop, errors="ignore")
    y_df    = pd.DataFrame({h: df.OnLine_Load_MW.shift(-h) for h in HORIZONS})
    mask    = y_df.notna().all(axis=1)
    X, y    = X_df.loc[mask].values.astype(np.float32), y_df.loc[mask].values.astype(np.float32)

    scaler  = StandardScaler().fit(X)
    X_scaled= scaler.transform(X)
    joblib.dump(scaler, SCALER_PATH)

    # 3) Prepare sequences
    n   = len(X_scaled)
    t0  = int(0.6 * n);  t1 = int(0.8 * n)
    train_gen = TimeseriesGenerator(X_scaled[:t1], y[:t1], length=SEQ_LEN, batch_size=256)
    val_gen   = TimeseriesGenerator(X_scaled[t0:], y[t0:], length=SEQ_LEN, batch_size=256)

    # 4) Build & compile model
    model = Sequential([
        LSTM(32,
             input_shape=(SEQ_LEN, X_scaled.shape[1]),
             kernel_regularizer=regularizers.l2(1e-3),
             recurrent_regularizer=regularizers.l2(1e-3)),
        Dropout(0.5),
        Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3)),
        Dense(len(HORIZONS))
    ])
    model.compile(optimizer=optimizers.Adam(5e-4), loss="mse")

    # 5) Callbacks & training
    es       = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ckpt     = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
    reduce_lr= ReduceLROnPlateau("val_loss", factor=0.2, patience=3, min_lr=1e-6)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[es, ckpt, reduce_lr],
        verbose=1
    )

if __name__ == "__main__":
    main()