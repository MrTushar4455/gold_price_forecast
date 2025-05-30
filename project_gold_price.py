import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

# Load dataset
df = pd.read_csv("Gold_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

# Load models

arima_model = joblib.load("arima model.pkl")

# Helper: create lag features
def create_lag_features(series, n_lags=30):
    return np.array(series[-n_lags:]).reshape(1, -1)

# Forecast functions


def forecast_arima(model, df, days=30):
    forecast_values = model.forecast(steps=days)
    start_date = df['date'].max() + timedelta(days=1)
    forecast_dates = [start_date + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({'date': forecast_dates, 'forecast_price': forecast_values})

# Sidebar menu


# XGB Forecast Page

# Holt-Winters Forecast Page


# ARIMA Forecast Page

st.title(" Gold Price Forecast - ARIMA")
forecast_df = forecast_arima(arima_model, df)
st.line_chart(forecast_df.set_index("date")["forecast_price"])
st.dataframe(forecast_df, use_container_width=True)
st.download_button(" Download Forecast", forecast_df.to_csv(index=False), "arima_forecast.csv")



