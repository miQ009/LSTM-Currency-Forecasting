import json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Currency Predictor (LSTM)", page_icon="💱", layout="wide")
st.title("💱 Currency Prediction using LSTM")
st.caption("Educational demo only. Not financial advice.")

MODEL_DIR = "models"
pairs = sorted(f[:-5].replace("_X", "=X") for f in os.listdir(MODEL_DIR) if f.endswith(".json")) \
    if os.path.isdir(MODEL_DIR) else []
if not pairs:
    st.error("No trained models found in the 'models' folder. Run train_colab.py first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Currency pair", pairs)
    future_days = st.slider("Forecast days", 1, 30, 7)


@st.cache_resource(show_spinner="Loading model...")
def load_model(ticker):
    from tensorflow.keras.models import load_model as lm  # lazy import
    name = ticker.replace("=", "_")
    model = lm(f"{MODEL_DIR}/{name}.keras", compile=False)
    scaler = joblib.load(f"{MODEL_DIR}/{name}_scaler.pkl")
    meta = json.load(open(f"{MODEL_DIR}/{name}.json"))
    return model, scaler, meta


@st.cache_data(ttl=3600, show_spinner="Downloading latest data...")
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[["Close"]].dropna()


def windows(arr, ts):
    X = np.array([arr[i:i + ts, 0] for i in range(len(arr) - ts)])
    return X.reshape(X.shape[0], ts, 1), arr[ts:, 0]


model, scaler, meta = load_model(ticker)
ts = meta["time_step"]
data = load_data(ticker)
scaled = scaler.transform(data)

c1, c2, c3 = st.columns(3)
c1.metric("Latest close", f"{data['Close'].iloc[-1]:.4f}")
c2.metric("Last date", data.index[-1].strftime("%Y-%m-%d"))
c3.metric("Model trained up to", meta["train_end"])

# Out-of-sample evaluation: only data AFTER the training cutoff
cut = int((data.index <= pd.Timestamp(meta["train_end"])).sum())
X_test, y_test = windows(scaled[cut - ts:], ts)
pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
test_idx = data.index[cut:]

rmse = float(np.sqrt(mean_squared_error(actual, pred)))
mae = float(mean_absolute_error(actual, pred))
mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

st.subheader("Out-of-sample performance")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.4f}")
m2.metric("MAE", f"{mae:.4f}")
m3.metric("MAPE", f"{mape:.2f}%")

fig = go.Figure()
fig.add_scatter(x=test_idx, y=actual, name="Actual")
fig.add_scatter(x=test_idx, y=pred, name="Predicted")
fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Next {future_days} business days")
window = scaled[-ts:].reshape(1, ts, 1)
out = []
for _ in range(future_days):
    p = model.predict(window, verbose=0)[0][0]
    out.append([p])
    window = np.append(window[:, 1:, :], [[[p]]], axis=1)
preds = scaler.inverse_transform(out).ravel()
dates = pd.bdate_range(data.index[-1] + pd.Timedelta(days=1), periods=future_days)
fdf = pd.DataFrame({"Date": dates.date, "Predicted rate": preds.round(4)})

fig = go.Figure(go.Scatter(x=dates, y=preds, mode="lines+markers"))
fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(fdf, use_container_width=True, hide_index=True)
st.download_button("Download forecast CSV", fdf.to_csv(index=False), "forecast.csv", "text/csv")
st.info("Multi-step forecasts feed predictions back as input, so errors compound. "
        "Later days are increasingly unreliable.")
