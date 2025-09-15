import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# --- Load the trained model ---
model_filename = "gradient_boosting_model.pkl"
gb_model = joblib.load(model_filename)

# --- Define feature order ---
features = [
    'lead_time_minutes', 'reschedule_count', 'channel_enc',
    'service_type_enc', 'holiday_flag', 'weather_enc', 'tags_enc'
]

# --- Example category mappings (replace with your dataset categories if different) ---
channel_map = {"Online": 0, "Phone": 1, "In-Person": 2}
service_type_map = {"Consultation": 0, "Follow-up": 1, "Emergency": 2}
weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Storm": 3}
tags_map = {"New": 0, "Returning": 1, "VIP": 2, "Other": 3}

# Labels (update with your training target classes)
label_map = {
    0: "No-Show",
    1: "Completed",
    2: "Cancelled",
    3: "Rescheduled",
    4: "Confirmed"
}


# --- Streamlit UI ---
st.title("🗓️ Appointment Status Prediction (Gradient Boosting)")
st.write("Choose booking and appointment times to calculate lead time ⏱️")

# Booking & Appointment Date/Time
booking_date = st.date_input("📅 Booking Date", datetime.today())
booking_time = st.time_input("⏰ Booking Time", datetime.now().time())
appointment_date = st.date_input("📅 Appointment Date", datetime.today())
appointment_time = st.time_input("⏰ Appointment Time", (datetime.now().replace(minute=0, second=0)))

# Convert to datetime
booking_datetime = datetime.combine(booking_date, booking_time)
appointment_datetime = datetime.combine(appointment_date, appointment_time)

# Calculate lead time in minutes
lead_time_minutes = max(0, int((appointment_datetime - booking_datetime).total_seconds() / 60))
st.info(f"⏱️ Lead Time: **{lead_time_minutes} minutes**")

# Other Inputs
reschedule_count = st.number_input("🔄 Number of reschedules", min_value=0, max_value=20, step=1)
channel_enc = channel_map[st.selectbox("📡 Channel", list(channel_map.keys()))]
service_type_enc = service_type_map[st.selectbox("💼 Service Type", list(service_type_map.keys()))]
holiday_flag = st.radio("🎉 Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
weather_enc = weather_map[st.selectbox("🌦️ Weather", list(weather_map.keys()))]
tags_enc = tags_map[st.selectbox("🏷️ Tags", list(tags_map.keys()))]

# --- Predict Button ---
if st.button("🔮 Predict Appointment Status"):
    user_data = np.array([[
        lead_time_minutes, reschedule_count, channel_enc,
        service_type_enc, holiday_flag, weather_enc, tags_enc
    ]])

    prediction = gb_model.predict(user_data)[0]

    # Safely handle unknown predictions
    predicted_status = label_map.get(prediction, f"Unknown Class ({prediction})")

    st.success(f"✅ Predicted Status: **{predicted_status}**")
