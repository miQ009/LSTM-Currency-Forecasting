# 💱 Currency Prediction Using LSTM

A class project that uses an LSTM (Long Short-Term Memory) neural network to forecast foreign exchange rates from historical data, with an interactive Streamlit web app.

**Live demo:** _add your Streamlit link here_

## Overview

Historical closing prices are downloaded from Yahoo Finance and used to train an LSTM model. The app shows how well the model performs on unseen data and forecasts the next few business days.

Supported pairs: USD/INR, EUR/USD, GBP/USD, USD/JPY (easy to extend).

Three architectures (LSTM, GRU, BiLSTM) are trained on Close, High and Low prices and compared side by side.

## Features

* Pulls live historical data with `yfinance`
* Normalizes data with `MinMaxScaler` (fit on training data only, to avoid data leakage)
* 60-day sliding window sequences
* Multi-feature input: Close, High and Low prices
* Three models compared: LSTM, GRU and Bidirectional LSTM (TensorFlow / Keras)
* Out-of-sample evaluation with RMSE, MAE and MAPE, with a model comparison table
* 95% uncertainty band on forecasts (approximate, based on past test errors)
* Interactive charts (Plotly): actual vs. predicted, plus 1–30 day forecast
* Downloadable forecast as CSV

## Project Structure

```
├── app.py              # Streamlit web app (loads pre-trained models)
├── train_colab.py      # Training script (run in Google Colab)
├── models/             # Saved .keras models, scalers and metadata
├── requirements.txt
└── README.md
```

## How It Works

1. **Train:** `train_colab.py` downloads data, trains LSTM, GRU and BiLSTM models per pair, and saves the model, scaler and training cutoff date to `models/`.
2. **Serve:** `app.py` loads the saved models, fetches the latest data, evaluates on dates after the training cutoff, and forecasts ahead.

Training is done separately so the deployed app stays lightweight and starts quickly.

## Installation & Running Locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

To retrain or add currency pairs, edit `PAIRS` in `train_colab.py`, run it in Colab, and replace the `models/` folder with the downloaded one.

## Technologies Used

Python, Streamlit, TensorFlow / Keras, yfinance, Pandas, NumPy, Scikit-learn, Plotly

## Future Improvements

* Try Transformer and Encoder-Decoder LSTM models
* Automate periodic retraining (e.g. weekly with GitHub Actions)
* Add external features (interest rates, inflation, news sentiment)

## Limitations

* Future High/Low values are assumed to keep the latest spread from Close when forecasting.
* Exchange rates are driven by news, policy and market events that price history alone cannot capture.
* LSTMs on this kind of data mostly learn "tomorrow ≈ today", so a good-looking chart does not mean real predictive power.
* Multi-day forecasts feed predictions back as inputs, so errors compound and later days are less reliable.

## Disclaimer

This project is for educational purposes only and is **not** financial advice.

## Data Source

[Yahoo Finance – Currencies](https://finance.yahoo.com/markets/currencies/)

## Author

_Your name_
