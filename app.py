import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from prophet import Prophet

# --- 1. CONFIGURATION AND DATA LOADING ---
channel_map = {"Online": 0, "Phone": 1, "In-Person": 2}
service_type_map = {"Consultation": 0, "Follow-up": 1, "Emergency": 2}
weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Storm": 3}
tags_map = {"New": 0, "Returning": 1, "VIP": 2, "Other": 3}
label_map = {0: "No-Show", 1: "Completed", 2: "Cancelled", 3: "Rescheduled", 4: "Confirmed"}

# --- 2. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Demand & Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Demand & Appointment Status Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Appointment Status Prediction", "🗓️ Daily Demand Heatmap"])

# --- TAB 1: APPOINTMENT STATUS PREDICTION ---
with tab1:
    st.header("Predict an Appointment's Status")
    
    model_filename = "gradient_boosting_model.pkl"
    try:
        gb_model = joblib.load(model_filename)
        model_loaded = True
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found.")
        st.info("Please ensure the model file is in the same directory as the script.")
        model_loaded = False
        
    if model_loaded:
        st.write("Choose booking and appointment times to calculate lead time ⏱️")

        now = datetime.now()
        col1, col2 = st.columns(2)
        with col1:
            booking_date = st.date_input("📅 Booking Date", now.date())
            booking_time = st.time_input("⏰ Booking Time", now.time())
        with col2:
            appointment_date = st.date_input("📅 Appointment Date", now.date())
            appointment_time = st.time_input("⏰ Appointment Time", (now + timedelta(hours=1)).time())

        booking_datetime = datetime.combine(booking_date, booking_time)
        appointment_datetime = datetime.combine(appointment_date, appointment_time)
        if appointment_datetime < booking_datetime:
            st.warning("⚠️ Appointment datetime is before booking datetime. Lead time set to 0.")
        lead_time_minutes = max(0, int((appointment_datetime - booking_datetime).total_seconds() / 60))
        st.info(f"⏱️ Lead Time: **{lead_time_minutes} minutes**")

        st.subheader("Other Appointment Details")
        col3, col4, col5 = st.columns(3)
        with col3:
            reschedule_count = st.number_input("🔄 Number of reschedules", min_value=0, max_value=20, step=1)
            channel_enc = channel_map[st.selectbox("📡 Channel", list(channel_map.keys()))]
        with col4:
            service_type_enc = service_type_map[st.selectbox("💼 Service Type", list(service_type_map.keys()))]
            weather_enc = weather_map[st.selectbox("🌦️ Weather", list(weather_map.keys()))]
        with col5:
            holiday_flag = st.radio("🎉 Is a Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            tags_enc = tags_map[st.selectbox("🏷️ Tags", list(tags_map.keys()))]

        if st.button("🔮 Predict Appointment Status", use_container_width=True):
            user_data = np.array([[lead_time_minutes, reschedule_count, channel_enc,
                                   service_type_enc, holiday_flag, weather_enc, tags_enc]])
            prediction = gb_model.predict(user_data)[0]
            predicted_status = label_map.get(prediction, f"Unknown Class ({prediction})")
            st.success(f"✅ Predicted Status: **{predicted_status}**")

# --- TAB 2: DAILY DEMAND HEATMAP FROM UPLOADED DATA ---
# --- TAB 2: DAILY DEMAND HEATMAP ---
# --- TAB 2: DAILY DEMAND HEATMAP ---
with tab2:
        # --- Forecasting using Prophet with interactive 3D-like plot ---
        st.subheader("Forecasting Future Hype & Lows (Interactive)")
        
        # Prepare data for Prophet
        df_forecast = df_clean[[date_column, numeric_column]].rename(columns={
            date_column: 'ds', numeric_column: 'y'
        })
        df_forecast = df_forecast.groupby('ds').sum().reset_index()  # Aggregate daily
        
        # Fit Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df_forecast)
        
        # Forecast next 30 days
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Identify hype and low dates
        hype_threshold = forecast['yhat'].quantile(0.75)
        low_threshold = forecast['yhat'].quantile(0.25)
        hype_dates = forecast[forecast['yhat'] >= hype_threshold]['ds']
        low_dates = forecast[forecast['yhat'] <= low_threshold]['ds']
        
        # --- Create interactive Plotly plot ---
        fig = go.Figure()
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines',
            name='Forecast', line=dict(color='blue')
        ))
        
        # Forecast uncertainty
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
            name='Upper Bound', line=dict(color='lightblue'), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
            name='Lower Bound', line=dict(color='lightblue'), fill='tonexty', showlegend=False
        ))
        
        # Actual data points
        fig.add_trace(go.Scatter(
            x=df_forecast['ds'], y=df_forecast['y'], mode='markers',
            name='Actual', marker=dict(color='red', size=6)
        ))
        
        # Add hype and low dates
        fig.add_trace(go.Scatter(
            x=hype_dates, y=forecast.loc[forecast['ds'].isin(hype_dates), 'yhat'],
            mode='markers', name='Hype Dates',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
        fig.add_trace(go.Scatter(
            x=low_dates, y=forecast.loc[forecast['ds'].isin(low_dates), 'yhat'],
            mode='markers', name='Low Dates',
            marker=dict(color='orange', size=8, symbol='triangle-down')
        ))
        
        fig.update_layout(
            title=f"Forecast of {numeric_column} (Interactive)",
            xaxis_title="Date",
            yaxis_title=numeric_column,
            hovermode="x unified",
            width=1200,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display hype/low dates
        st.write("**Predicted Hype Dates (High Values):**")
        st.dataframe(hype_dates.dt.date.reset_index(drop=True))
        st.write("**Predicted Low Dates (Low Values):**")
        st.dataframe(low_dates.dt.date.reset_index(drop=True))
