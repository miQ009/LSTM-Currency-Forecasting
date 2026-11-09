# Run this in Google Colab (paste into a cell). For each pair it trains
# LSTM, GRU and BiLSTM on Close/High/Low and saves everything to "models/".
# Then download models.zip, unzip it next to app.py.
!pip install -q yfinance scikit-learn

import os, json, joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense

PAIRS = ["USDINR=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
START = "2020-01-01"
TIME_STEP = 60
EPOCHS = 20
FEATURES = ["Close", "High", "Low"]   # Close MUST stay first (it is the target)

tf.keras.utils.set_random_seed(42)   # same results on every run
os.makedirs("models", exist_ok=True)

def make_windows(arr, ts):
    X = np.array([arr[i:i + ts] for i in range(len(arr) - ts)])
    return X, arr[ts:, 0]

def build(arch, ts, n_feat):
    if arch == "LSTM":
        body = [LSTM(50, return_sequences=True), LSTM(50)]
    elif arch == "GRU":
        body = [GRU(50, return_sequences=True), GRU(50)]
    else:  # BiLSTM
        body = [Bidirectional(LSTM(50, return_sequences=True)), Bidirectional(LSTM(50))]
    m = Sequential([Input(shape=(ts, n_feat))] + body + [Dense(1)])
    m.compile(optimizer="adam", loss="mean_squared_error")
    return m

ARCHS = ["LSTM", "GRU", "BiLSTM"]

for t in PAIRS:
    df = yf.download(t, start=START, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[FEATURES].dropna()

    split = int(len(df) * 0.8)
    scaler = MinMaxScaler().fit(df.iloc[:split])          # train only (no leakage)
    X, y = make_windows(scaler.transform(df.iloc[:split]), TIME_STEP)

    name = t.replace("=", "_")
    for arch in ARCHS:
        model = build(arch, TIME_STEP, len(FEATURES))
        model.fit(X, y, epochs=EPOCHS, batch_size=64, verbose=0)
        model.save(f"models/{name}_{arch}.keras")
        print("saved", t, arch)

    joblib.dump(scaler, f"models/{name}_scaler.pkl")
    json.dump({"train_end": str(df.index[split - 1].date()), "time_step": TIME_STEP,
               "features": FEATURES, "archs": ARCHS},
              open(f"models/{name}.json", "w"))

!zip -r models.zip models
from google.colab import files
files.download("models.zip")
