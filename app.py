import json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Currency Predictor", page_icon="💱", layout="wide")
st.title("💱 Currency Prediction: LSTM vs GRU vs BiLSTM")
st.caption("Educational demo only. Not financial advice.")

MODEL_DIR = "models"
pairs = sorted(f[:-5].replace("_X", "=X") for f in os.listdir(MODEL_DIR) if f.endswith(".json")) \
    if os.path.isdir(MODEL_DIR) else []
if not pairs:
    st.error("No trained models found in the 'models' folder. Run train_colab.py first.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Downloading latest data...")
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[["Close", "High", "Low"]].dropna()


def windows(arr, ts):
    X = np.array([arr[i:i + ts] for i in range(len(arr) - ts)])
    return X, arr[ts:, 0]


@st.cache_resource(show_spinner="Loading models & evaluating...")
def load_and_eval(ticker, last_date):
    from tensorflow.keras.models import load_model  # lazy import
    name = ticker.replace("=", "_")
    meta = json.load(open(f"{MODEL_DIR}/{name}.json"))
    scaler = joblib.load(f"{MODEL_DIR}/{name}_scaler.pkl")
    ts, feats = meta["time_step"], meta["features"]

    data = load_data(ticker)[feats]
    scaled = scaler.transform(data)
    lo, rng = scaler.data_min_[0], scaler.data_range_[0]
    inv = lambda v: np.asarray(v) * rng + lo   # scaled Close -> real Close

    cut = int((data.index <= pd.Timestamp(meta["train_end"])).sum())
    X_test, y_test = windows(scaled[cut - ts:], ts)
    actual = inv(y_test)

    models, preds, rows = {}, {}, []
    for arch in meta["archs"]:
        m = load_model(f"{MODEL_DIR}/{name}_{arch}.keras", compile=False)
        p = inv(m.predict(X_test, verbose=0).ravel())
        models[arch], preds[arch] = m, p
        rows.append({"Model": arch,
                     "RMSE": float(np.sqrt(mean_squared_error(actual, p))),
                     "MAE": float(mean_absolute_error(actual, p)),
                     "MAPE %": float(np.mean(np.abs((actual - p) / actual)) * 100)})
    return meta, scaler, models, preds, actual, data.index[cut:], pd.DataFrame(rows)


with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Currency pair", pairs)
    future_days = st.slider("Forecast days", 1, 30, 7)

data_all = load_data(ticker)
meta, scaler, models, preds, actual, test_idx, metrics = load_and_eval(
    ticker, str(data_all.index[-1].date()))
feats, ts = meta["features"], meta["time_step"]

with st.sidebar:
    best = int(metrics["RMSE"].idxmin())
    arch = st.selectbox("Model", meta["archs"], index=best)
    show_band = st.checkbox("Show uncertainty band (95%)", value=True)

data = data_all[feats]
c1, c2, c3 = st.columns(3)
c1.metric("Latest close", f"{data['Close'].iloc[-1]:.4f}")
c2.metric("Last date", data.index[-1].strftime("%Y-%m-%d"))
c3.metric("Trained up to", meta["train_end"])

# ---- Model comparison ----
st.subheader("Model comparison (out-of-sample)")
st.dataframe(metrics.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE %": "{:.2f}"})
             .highlight_min(subset=["RMSE", "MAE", "MAPE %"], color="#1f6f43"),
             use_container_width=True, hide_index=True)
st.caption(f"Best by RMSE: **{metrics.loc[best, 'Model']}**. Metrics use only dates after the training cutoff.")

# ---- Actual vs predicted ----
fig = go.Figure()
fig.add_scatter(x=test_idx, y=actual, name="Actual")
fig.add_scatter(x=test_idx, y=preds[arch], name=f"Predicted ({arch})")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# ---- Forecast ----
st.subheader(f"Next {future_days} business days ({arch})")
model = models[arch]
scaled = scaler.transform(data)
lo, rng = scaler.data_min_[0], scaler.data_range_[0]
hi_sp = float(data["High"].iloc[-1] - data["Close"].iloc[-1])   # keep last High/Low spread
lo_sp = float(data["Close"].iloc[-1] - data["Low"].iloc[-1])

window = scaled[-ts:].reshape(1, ts, len(feats))
out = []
for _ in range(future_days):
    c = float(model.predict(window, verbose=0)[0][0]) * rng + lo
    out.append(c)
    row = pd.DataFrame([[c, c + hi_sp, c - lo_sp]], columns=feats)
    window = np.append(window[:, 1:, :], scaler.transform(row).reshape(1, 1, -1), axis=1)
out = np.array(out)

sigma = float(np.std(actual - preds[arch]))
half = 1.96 * sigma * np.sqrt(np.arange(1, future_days + 1))
dates = pd.bdate_range(data.index[-1] + pd.Timedelta(days=1), periods=future_days)

fig = go.Figure()
if show_band:
    fig.add_scatter(x=dates, y=out + half, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=dates, y=out - half, fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(99,110,250,0.2)", name="95% band", hoverinfo="skip")
fig.add_scatter(x=dates, y=out, mode="lines+markers", name="Forecast")
fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

fdf = pd.DataFrame({"Date": dates.date, "Predicted rate": out.round(4),
                    "Lower (95%)": (out - half).round(4), "Upper (95%)": (out + half).round(4)})
st.dataframe(fdf, use_container_width=True, hide_index=True)
st.download_button("Download forecast CSV", fdf.to_csv(index=False), "forecast.csv", "text/csv")
st.info("The band is an approximation based on the model's past test errors, widening with the "
        "forecast horizon. Future High/Low are assumed to keep the latest spread from Close. "
        "Multi-day forecasts compound errors, so later days are less reliable.")
