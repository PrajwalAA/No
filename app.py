import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
from datetime import datetime, time

# Set Streamlit page configuration
st.set_page_config(page_title="🕒 Appointment Lead Time & Prediction", layout="centered")

# --- Load the trained model ---
# Ensure you have the 'gradient_boosting_model.pkl' file in the same directory
try:
    model_filename = "gradient_boosting_model.pkl"
    gb_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Error: The model file 'gradient_boosting_model.pkl' was not found.")
    st.stop() # Stop the app if the model is not found

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
label_map = {0: "No-Show", 1: "Completed", 2: "Cancelled"}

# =================================
#       Streamlit UI Components
# =================================

st.title("📅 Appointment Status Prediction with Analog Clock Picker")
st.write("This app predicts the status of an appointment based on several key features.")

# =================================
#    CLOCK PICKER (VISUAL ONLY)
# =================================
# This is a visual component and does not directly provide input to the model.
# The user must use the selectboxes below for actual data entry.
st.subheader("🕒 Visual Clock Picker (Demo)")
st.caption("ℹ️ The analog clocks below are for visual effect only. Use the dropdowns for actual time input.")

# Note: The HTML/JS for the clock picker is for demonstration. It does not
# pass values back to the Streamlit app's Python script.
clock_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/clockpicker/0.0.7/bootstrap-clockpicker.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/clockpicker/0.0.7/bootstrap-clockpicker.min.js"></script>

<style>
/* Custom CSS to improve layout and visibility */
.clockpicker-html-container {
    display: flex;
    flex-wrap: wrap; /* Allows items to wrap on smaller screens */
    justify-content: center; /* Center the items horizontally */
    align-items: center;
    gap: 40px; /* Add space between the two clock pickers */
    padding: 20px 0;
}
.clockpicker-html-container .clockpicker {
    width: 200px; /* Give the container a fixed width */
    text-align: center;
}
.clockpicker-html-container input {
    font-size: 1.25rem; /* Make the input text larger */
    text-align: center;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 8px;
}
</style>

<div class="clockpicker-html-container">
    <div class="clockpicker" data-placement="bottom" data-align="left" data-autoclose="true">
        <b>Booking Time</b>
        <input type="text" value="11:00">
    </div>
    <div class="clockpicker" data-placement="bottom" data-align="left" data-autoclose="true">
        <b>Appointment Time</b>
        <input type="text" value="12:00">
    </div>
</div>

<script type="text/javascript">
    $('.clockpicker').clockpicker();
</script>
"""
components.html(clock_html, height=250)

# =================================
#       STREAMLIT INPUTS (WORKING)
# =================================

# --- Booking Date & Time ---
st.subheader("📌 Booking Date & Time")
col1, col2, col3 = st.columns(3)
with col1:
    booking_date = st.date_input("Booking Date", datetime.today())
with col2:
    booking_hour = st.selectbox("Booking Hour", list(range(0, 24)), index=11)
with col3:
    booking_minute = st.selectbox("Booking Minute", list(range(0, 60, 5)), index=0)
booking_time = time(booking_hour, booking_minute)

# --- Appointment Date & Time ---
st.subheader("📌 Appointment Date & Time")
col4, col5, col6 = st.columns(3)
with col4:
    appointment_date = st.date_input("Appointment Date", datetime.today())
with col5:
    appointment_hour = st.selectbox("Appointment Hour", list(range(0, 24)), index=12)
with col6:
    appointment_minute = st.selectbox("Appointment Minute", list(range(0, 60, 5)), index=0)
appointment_time = time(appointment_hour, appointment_minute)

# --- Calculate Lead Time ---
booking_datetime = datetime.combine(booking_date, booking_time)
appointment_datetime = datetime.combine(appointment_date, appointment_time)

# Calculate lead time in minutes
lead_time_seconds = (appointment_datetime - booking_datetime).total_seconds()
lead_time_minutes = int(lead_time_seconds / 60)

if lead_time_minutes < 0:
    st.error("⚠️ Appointment time must be after booking time.")
    lead_time_minutes = 0 # Reset to 0 to prevent negative values in the model input
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
    # Create a NumPy array from the user inputs
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
        # Make the prediction
        prediction = gb_model.predict(user_data)[0]

        # Get the predicted status from the label map
        predicted_status = label_map.get(prediction, f"Unknown Class ({prediction})")

        st.success(f"✅ Predicted Status: **{predicted_status}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
