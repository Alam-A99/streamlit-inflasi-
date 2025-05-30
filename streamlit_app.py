# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulasi & Forecast Inflasi", layout="wide")

st.title("📈 Simulasi dan Forecasting Inflasi Bulanan (2010–2023)")

# --------- Simulasi Data Inflasi Bulanan ---------
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', end='2023-12-01', freq='MS')
n = len(dates)

trend = np.linspace(3, 5, n)
seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 0.4, n)
inflation = trend + seasonal + noise

df = pd.DataFrame({'Date': dates, 'Inflation': inflation})
df.set_index('Date', inplace=True)

# --------- Forecasting Horizon Slider ---------
horizon = st.slider("Berapa bulan ke depan ingin diprediksi?", 6, 36, 12)
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')

# --------- Model Forecasting ---------
model_arima = ARIMA(df['Inflation'], order=(1,1,1)).fit()
forecast_arima = model_arima.forecast(steps=horizon)

model_es = ExponentialSmoothing(df['Inflation'], trend='add', seasonal='add', seasonal_periods=12).fit()
forecast_es = model_es.forecast(horizon)

df_lr = df.copy()
df_lr['t'] = np.arange(len(df_lr))
X = df_lr[['t']]
y = df_lr['Inflation']
model_lr = LinearRegression().fit(X, y)
future_t = np.arange(len(df_lr), len(df_lr) + horizon).reshape(-1, 1)
forecast_lr = model_lr.predict(future_t)

# --------- Plotly Chart ---------
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['Inflation'], mode='lines+markers', name='Data Aktual'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_arima, mode='lines+markers', name='ARIMA Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_es, mode='lines+markers', name='Exp. Smoothing Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lr, mode='lines+markers', name='Linear Regression Forecast'))

fig.update_layout(title='Simulasi & Forecasting Inflasi Bulanan',
                  xaxis_title='Tanggal',
                  yaxis_title='Inflasi (%)',
                  template='plotly_white',
                  hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)

st.caption("Model: ARIMA (1,1,1), Holt-Winters, Linear Regression. Data yang digunakan Data percuma, kata orang Malaysia :) #Cheeersss.")
