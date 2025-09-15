from prophet import Prophet
import pandas as pd
import streamlit as st

# Prepare data
df_forecast = df_clean[[date_column, numeric_column]].rename(
    columns={date_column: 'ds', numeric_column: 'y'}
)

# Fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df_forecast)

# Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
st.line_chart(forecast.set_index('ds')['yhat'])

# Identify hype and low dates
hype_threshold = forecast['yhat'].quantile(0.75)
low_threshold = forecast['yhat'].quantile(0.25)

hype_dates = forecast[forecast['yhat'] >= hype_threshold]['ds']
low_dates = forecast[forecast['yhat'] <= low_threshold]['ds']

st.write("**Predicted Hype Dates:**", hype_dates.dt.date.tolist())
st.write("**Predicted Low Dates:**", low_dates.dt.date.tolist())
