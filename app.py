import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calplot
import matplotlib.pyplot as plt

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
with tab2:
    st.header("Daily Demand Heatmap from Excel/CSV")
    st.write("Upload your time-series data to visualize daily patterns.")

    uploaded_file = st.file_uploader("Upload your file", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            # Load Excel/CSV
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_csv(uploaded_file)
            
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Ask user to select date and numeric columns
            col_options = df.columns.tolist()
            default_date_col = 'booking_date' if 'booking_date' in col_options else col_options[0]
            default_value_col = 'lead_time_minutes' if 'lead_time_minutes' in col_options else col_options[1]
            
            date_col = st.selectbox("Select Date Column", options=col_options, index=col_options.index(default_date_col))
            value_col = st.selectbox("Select Numeric Column for Heatmap", options=col_options, index=col_options.index(default_value_col))

            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, value_col])

            # Aggregate numeric values per day
            daily_data = df.groupby(df[date_col].dt.date)[value_col].sum()

            # Convert index back to datetime for calplot
            daily_data.index = pd.to_datetime(daily_data.index)
            daily_data = daily_data.sort_index()

            # Generate calplot heatmap (fixed: removed ax parameter)
            st.write("### Heatmap")
            calplot.calplot(
                daily_data,
                suptitle=f"Daily {value_col} Heatmap",
                cmap="YlGnBu",
                edgecolor="gray",
                linewidth=0.5,
                monthlabels=True,
                figsize=(15, 6)
            )
            st.pyplot(plt.gcf())  # get current figure

        except Exception as e:
            st.error(f"Error loading or processing file: {e}")
    else:
        st.info("Please upload an Excel or CSV file to visualize the heatmap.")
