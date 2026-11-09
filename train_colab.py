# Run this in Google Colab (paste into a cell). It trains one model per pair
# and saves them into a "models" folder. Download the folder, put it next to app.py.
!pip install -q yfinance scikit-learn

import os, json, joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

PAIRS = ["USDINR=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
START = "2020-01-01"
TIME_STEP = 60
EPOCHS = 20

os.makedirs("models", exist_ok=True)

def make_windows(arr, ts):
    X, y = [], []
    for i in range(len(arr) - ts):
        X.append(arr[i:i + ts, 0]); y.append(arr[i + ts, 0])
    X = np.array(X)
    return X.reshape(X.shape[0], X.shape[1], 1), np.array(y)

for t in PAIRS:
    df = yf.download(t, start=START, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Close"]].dropna()

    split = int(len(df) * 0.8)
    scaler = MinMaxScaler().fit(df.iloc[:split])          # train only
    scaled = scaler.transform(df.iloc[:split])
    X, y = make_windows(scaled, TIME_STEP)

    model = Sequential([Input(shape=(TIME_STEP, 1)),
                        LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=EPOCHS, batch_size=64, verbose=0)

    name = t.replace("=", "_")
    model.save(f"models/{name}.keras")
    joblib.dump(scaler, f"models/{name}_scaler.pkl")
    json.dump({"train_end": str(df.index[split - 1].date()),
               "time_step": TIME_STEP}, open(f"models/{name}.json", "w"))
    print("saved", t)

!zip -r models.zip models
from google.colab import files
files.download("models.zip")
