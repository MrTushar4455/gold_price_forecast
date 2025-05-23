import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

# --- CUSTOM BACKGROUND ---
def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({main_bg});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

bg_url = "https://images.unsplash.com/photo-1622994609188-1978c5b1c51e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80"
set_bg_hack(bg_url)

# --- LOAD DATA ---
df = pd.read_csv("C:/Users/tusha/Downloads/Gold_data.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- LOAD MODELS ---
xgb_model = joblib.load("xgb_model.pkl")
hw_model=joblib.load("hw_model.pkl")

# --- SIDEBAR ---
menu = st.sidebar.selectbox("Navigation", ["📄 View Data", "📈 XGB Forecast", "📉 Holt-Winters Forecast", "📊 Compare Forecasts"])
forecast_days = st.sidebar.slider("Forecast days", 7, 60, 30)

# --- FUNCTIONS ---
def forecast_with_xgb(df, model, days=30, n_lags=30):
    df_copy = df.copy()
    for lag in range(1, n_lags + 1):
        df_copy[f"lag_{lag}"] = df_copy["price"].shift(lag)
    df_copy.dropna(inplace=True)

    # RMSE on training data
    X = df_copy[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df_copy["price"]
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)

    # Forecast future
    input_seq = df_copy.iloc[-1, -n_lags:].values.tolist()
    preds = []
    for _ in range(days):
        X_input = np.array(input_seq[-n_lags:]).reshape(1, -1)
        pred = model.predict(X_input)[0]
        preds.append(pred)
        input_seq.append(pred)
    future_dates = pd.date_range(start=df["date"].max() + timedelta(days=1), periods=days)
    return pd.DataFrame({"date": future_dates, "forecast_xgb": preds}), rmse

def forecast_with_hw(model, last_date, days=30):
    forecast = model.forecast(days)
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    return pd.DataFrame({"date": future_dates, "forecast_hw": forecast})

# --- PAGES ---
if menu == "📄 View Data":
    st.title("Gold Price Dataset")
    st.dataframe(df)

elif menu == "📈 XGB Forecast":
    st.title("Forecast using XGBRegressor")
    xgb_df, rmse_xgb = forecast_with_xgb(df, xgb_model, forecast_days)
    st.write(f"📉 RMSE on training data: `{rmse_xgb:.2f}`")
    st.line_chart(xgb_df.set_index("date"))
    st.dataframe(xgb_df)

    # CSV download
    csv = xgb_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download XGB Forecast CSV", csv, "xgb_forecast.csv", "text/csv")

elif menu == "📉 Holt-Winters Forecast":
    st.title("Forecast using Holt-Winters")
    hw_df = forecast_with_hw(hw_model, df["date"].max(), forecast_days)
    st.line_chart(hw_df.set_index("date"))
    st.dataframe(hw_df)

    # CSV download
    csv = hw_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Holt-Winters Forecast CSV", csv, "hw_forecast.csv", "text/csv")

elif menu == "📊 Compare Forecasts":
    st.title("Compare XGB vs Holt-Winters Forecast")
    xgb_df, _ = forecast_with_xgb(df, xgb_model, forecast_days)
    hw_df = forecast_with_hw(hw_model, df["date"].max(), forecast_days)
    merged = pd.merge(xgb_df, hw_df, on="date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["price"], label="Actual", color="black")
    ax.plot(merged["date"], merged["forecast_xgb"], label="XGB Forecast", linestyle="--")
    ax.plot(merged["date"], merged["forecast_hw"], label="HW Forecast", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (₹)")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(merged)
    # Download merged
    csv = merged.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Comparison CSV", csv, "comparison.csv", "text/csv")
