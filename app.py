import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration and Mappings (Modify these to match your data) ---
# NOTE: The model must be trained on data with these exact numerical encodings.

# Define the order of features the model expects
FEATURES = [
    'lead_time_minutes', 'reschedule_count', 'channel_enc',
    'service_type_enc', 'holiday_flag', 'weather_enc', 'tags_enc'
]

# Mapping from user-friendly names to numerical encodings
CHANNEL_MAP = {"Online": 0, "Phone": 1, "In-Person": 2}
SERVICE_TYPE_MAP = {"Consultation": 0, "Follow-up": 1, "Emergency": 2}
WEATHER_MAP = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Storm": 3}
TAGS_MAP = {"New": 0, "Returning": 1, "VIP": 2, "Other": 3}

# Mapping from model's numerical output to human-readable labels
LABEL_MAP = {
    0: "No-Show",
    1: "Completed",
    2: "Cancelled",
    3: "Rescheduled",
    4: "Confirmed"
}

# --- Load the trained model ---
MODEL_FILENAME = "gradient_boosting_model.pkl"
try:
    gb_model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"Error: The model file '{MODEL_FILENAME}' was not found.")
    st.error("Please ensure the trained model is in the same directory as this script.")
    st.stop()  # Stop the app if the model isn't found

# --- Streamlit UI ---
st.set_page_config(
    page_title="Appointment Status Predictor",
    page_icon="🗓️",
    layout="centered"
)

st.title("🗓️ Appointment Status Prediction")
st.markdown("### Powered by a Gradient Boosting Model")

st.write("Enter the details of a new appointment to predict its status.")

# --- User Inputs ---
with st.container(border=True):
    st.subheader("Appointment Details")

    # Booking & Appointment Date/Time
    col1, col2 = st.columns(2)
    with col1:
        booking_date = st.date_input("📅 Booking Date", datetime.today())
        booking_time = st.time_input("⏰ Booking Time", datetime.now().time())

    with col2:
        appointment_date = st.date_input("📅 Appointment Date", datetime.today() + timedelta(days=7))
        appointment_time = st.time_input("⏰ Appointment Time", datetime.now().time().replace(minute=0, second=0))

    # Convert to datetime objects for calculation
    try:
        booking_datetime = datetime.combine(booking_date, booking_time)
        appointment_datetime = datetime.combine(appointment_date, appointment_time)
    except Exception as e:
        st.error(f"An error occurred with the date/time inputs: {e}")
        st.stop()
    
    # Calculate lead time in minutes
    if appointment_datetime < booking_datetime:
        st.warning("⚠️ The appointment date/time cannot be before the booking date/time.")
        lead_time_minutes = 0
    else:
        lead_time_minutes = int((appointment_datetime - booking_datetime).total_seconds() / 60)
    
    st.info(f"⏱️ Calculated Lead Time: **{lead_time_minutes} minutes**")

    # Other inputs
    reschedule_count = st.number_input("🔄 Number of reschedules", min_value=0, max_value=20, value=0, step=1)
    
    channel_enc = st.selectbox("📡 Channel", list(CHANNEL_MAP.keys()), index=0)
    service_type_enc = st.selectbox("💼 Service Type", list(SERVICE_TYPE_MAP.keys()), index=0)
    
    holiday_flag = st.radio("🎉 Is the appointment day a public holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    weather_enc = st.selectbox("🌦️ Weather", list(WEATHER_MAP.keys()), index=0)
    tags_enc = st.selectbox("🏷️ Tags", list(TAGS_MAP.keys()), index=0)

# --- Prediction and Output ---
if st.button("🔮 Predict Appointment Status", use_container_width=True):
    # Create the feature vector in the correct order
    user_data = np.array([[
        lead_time_minutes,
        reschedule_count,
        CHANNEL_MAP[channel_enc],
        SERVICE_TYPE_MAP[service_type_enc],
        holiday_flag,
        WEATHER_MAP[weather_enc],
        TAGS_MAP[tags_enc]
    ]])

    # Make the prediction
    try:
        prediction = gb_model.predict(user_data)[0]
        predicted_status = LABEL_MAP.get(prediction, f"Unknown Class ({prediction})")
        st.success(f"✅ Predicted Appointment Status: **{predicted_status}**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and try again.")
