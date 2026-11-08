# Currency Prediction Using LSTM

A simple class project that uses an LSTM (Long Short-Term Memory) neural network to predict the USD/INR currency exchange rate using historical data.

## Overview

This project uses historical USD/INR exchange rate data obtained from Yahoo Finance and applies an LSTM model to predict future exchange rates. 

The model is trained on historical closing prices and generates predictions for the next 7 days.

## Technologies Used

* **Python**
* **yfinance**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **TensorFlow / Keras**

## Features

* Fetches historical USD/INR exchange rate data.
* Normalizes data using `MinMaxScaler`.
* Creates sequences using a 60-day sliding window.
* Trains an LSTM neural network.
* Visualizes actual vs. predicted exchange rates.
* Predicts exchange rates for the next 7 days.

## Installation

Install the required libraries:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

## How to Run

Run the Python file:

```bash
python currency_prediction.py
```

## Data Source

* **Yahoo Finance** – Currency Markets

## Note

This project was developed as a small academic/class project for learning and experimenting with LSTM-based time-series forecasting. Predictions are for educational purposes only and should not be considered financial advice.
