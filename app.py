import streamlit as st
import joblib
import numpy as np
from datetime import datetime, time

# Set Streamlit page configuration
st.set_page_config(page_title="🕒 Appointment Lead Time & Prediction", layout="centered")

# --- Load the trained model ---
try:
    model_filename = "gradient_boosting_model.pkl"
    gb_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Error: The model file 'gradient_boosting_model.pkl' was not found.")
    st.stop()

# --- Example category mappings (replace with your dataset categories if different) ---
channel_map = {"Online": 0, "Phone": 1, "In-Person": 2}
service_type_map = {"Consultation": 0, "Follow-up": 1, "Emergency": 2}
weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Storm": 3}
tags_map = {"New": 0, "Returning": 1, "VIP": 2, "Other": 3}
label_map = {0: "No-Show", 1: "Completed", 2: "Cancelled"}

# Initialize session state for the pop-up
if 'show_booking_time_popup' not in st.session_state:
    st.session_state.show_booking_time_popup = False
if 'show_appointment_time_popup' not in st.session_state:
    st.session_state.show_appointment_time_popup = False

# =================================
#       Streamlit UI Components
# =================================

st.title("📅 Appointment Status Prediction")
st.write("This app predicts the status of an appointment based on several key features.")

# --- Time Picker Modals (pop-ups) ---
def booking_time_popup():
    """Renders the booking time picker pop-up."""
    with st.container():
        st.subheader("Booking Time Picker")
        col_h, col_m = st.columns(2)
        with col_h:
            booking_hour = st.selectbox("Booking Hour", list(range(0, 24)), index=11, key='popup_b_hour')
        with col_m:
            booking_minute = st.selectbox("Booking Minute", list(range(0, 60, 5)), index=0, key='popup_b_minute')
        
        if st.button("Confirm Booking Time"):
            st.session_state.booking_time = time(booking_hour, booking_minute)
            st.session_state.show_booking_time_popup = False
            st.experimental_rerun()

def appointment_time_popup():
    """Renders the appointment time picker pop-up."""
    with st.container():
        st.subheader("Appointment Time Picker")
        col_h, col_m = st.columns(2)
        with col_h:
            appointment_hour = st.selectbox("Appointment Hour", list(range(0, 24)), index=12, key='popup_a_hour')
        with col_m:
            appointment_minute = st.selectbox("Appointment Minute", list(range(0, 60, 5)), index=0, key='popup_a_minute')

        if st.button("Confirm Appointment Time"):
            st.session_state.appointment_time = time(appointment_hour, appointment_minute)
            st.session_state.show_appointment_time_popup = False
            st.experimental_rerun()


# --- Display the main app UI ---
if st.session_state.show_booking_time_popup:
    booking_time_popup()
elif st.session_state.show_appointment_time_popup:
    appointment_time_popup()
else:
    # --- Main inputs ---
    st.subheader("📌 Booking Date & Time")
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        booking_date = st.date_input("Booking Date", datetime.today())
    with col2:
        if st.button("Select Booking Time"):
            st.session_state.show_booking_time_popup = True
            st.experimental_rerun()
    
    # Display the selected booking time
    if 'booking_time' in st.session_state:
        st.info(f"Booking Time: **{st.session_state.booking_time.strftime('%H:%M')}**")
    
    st.subheader("📌 Appointment Date & Time")
    col3, col4 = st.columns([0.7, 0.3])
    with col3:
        appointment_date = st.date_input("Appointment Date", datetime.today())
    with col4:
        if st.button("Select Appointment Time"):
            st.session_state.show_appointment_time_popup = True
            st.experimental_rerun()

    # Display the selected appointment time
    if 'appointment_time' in st.session_state:
        st.info(f"Appointment Time: **{st.session_state.appointment_time.strftime('%H:%M')}**")
    
    # --- Calculate Lead Time ---
    lead_time_minutes = 0
    if 'booking_time' in st.session_state and 'appointment_time' in st.session_state:
        booking_datetime = datetime.combine(booking_date, st.session_state.booking_time)
        appointment_datetime = datetime.combine(appointment_date, st.session_state.appointment_time)
        
        lead_time_seconds = (appointment_datetime - booking_datetime).total_seconds()
        lead_time_minutes = int(lead_time_seconds / 60)

        if lead_time_minutes < 0:
            st.error("⚠️ Appointment time must be after booking time.")
            lead_time_minutes = 0
        else:
            st.info(f"⏳ Lead Time: **{lead_time_minutes} minutes**")

    # --- Other Inputs ---
    st.subheader("📊 Other Appointment Details")
    reschedule_count = st.number_input("Number of reschedules", min_value=0, max_value=20, step=1)
    channel_enc = channel_map[st.selectbox("Channel", list(channel_map.keys()))]
    service_type_enc = service_type_map[st.selectbox("Service Type", list(service_type_map.keys()))]
    holiday_flag = st.radio("Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    weather_enc = weather_map[st.selectbox("Weather", list(weather_map.keys()))]
    tags_enc = tags_map[st.selectbox("Tags", list(tags_map.keys()))]

    # --- Predict Button ---
    if st.button("🔮 Predict Appointment Status"):
        if 'booking_time' not in st.session_state or 'appointment_time' not in st.session_state:
            st.warning("Please select both booking and appointment times first.")
        else:
            user_data = np.array([[
                lead_time_minutes, 
                reschedule_count, 
                channel_enc,
                service_type_enc, 
                holiday_flag, 
                weather_enc, 
                tags_enc
            ]])

            try:
                prediction = gb_model.predict(user_data)[0]
                predicted_status = label_map.get(prediction, f"Unknown Class ({prediction})")
                st.success(f"✅ Predicted Status: **{predicted_status}**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
