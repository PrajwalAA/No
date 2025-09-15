import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
from datetime import datetime, time

st.set_page_config(page_title="🕒 Appointment Lead Time & Prediction", layout="centered")

# --- Load the trained model ---
model_filename = "gradient_boosting_model.pkl"
gb_model = joblib.load(model_filename)

# --- Feature order ---
features = ['lead_time_minutes', 'reschedule_count', 'channel_enc',
            'service_type_enc', 'holiday_flag', 'weather_enc', 'tags_enc']

# --- Category mappings ---
channel_map = {"Online": 0, "Phone": 1, "In-Person": 2}
service_type_map = {"Consultation": 0, "Follow-up": 1, "Emergency": 2}
weather_map = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Storm": 3}
tags_map = {"New": 0, "Returning": 1, "VIP": 2, "Other": 3}

st.title("📅 Appointment Status Prediction with Analog Clock Picker")

# =============================
#   CLOCK PICKER (VISUAL ONLY)
# =============================
st.subheader("🕒 Visual Clock Picker (Demo)")

clock_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/clockpicker/0.0.7/bootstrap-clockpicker.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/clockpicker/0.0.7/bootstrap-clockpicker.min.js"></script>

<b>Booking Time</b>
<div class="clockpicker" data-placement="bottom" data-align="left" data-autoclose="true">
    <input type="text" value="11:44">
</div>

<br><b>Appointment Time</b>
<div class="clockpicker" data-placement="bottom" data-align="left" data-autoclose="true">
    <input type="text" value="12:44">
</div>

<script type="text/javascript">
    $('.clockpicker').clockpicker();
</script>
"""
components.html(clock_html, height=250)

st.caption("ℹ️ The above analog clocks are for visual effect. Use the dropdowns below for actual input.")

# =============================
#   STREAMLIT INPUTS (WORKING)
# =============================

# --- Booking Date & Time ---
st.subheader("📌 Booking Date & Time")
booking_date = st.date_input("Booking Date", datetime.today())
booking_hour = st.selectbox("Booking Hour", list(range(0, 24)), index=11)
booking_minute = st.selectbox("Booking Minute", list(range(0, 60, 5)), index=44)
booking_time = time(booking_hour, booking_minute)

# --- Appointment Date & Time ---
st.subheader("📌 Appointment Date & Time")
appointment_date = st.date_input("Appointment Date", datetime.today())
appointment_hour = st.selectbox("Appointment Hour", list(range(0, 24)), index=12)
appointment_minute = st.selectbox("Appointment Minute", list(range(0, 60, 5)), index=44)
appointment_time = time(appointment_hour, appointment_minute)

# --- Calculate Lead Time ---
booking_datetime = datetime.combine(booking_date, booking_time)
appointment_datetime = datetime.combine(appointment_date, appointment_time)
lead_time_minutes = int((appointment_datetime - booking_datetime).total_seconds() / 60)

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
    user_data = np.array([[lead_time_minutes, reschedule_count, channel_enc,
                           service_type_enc, holiday_flag, weather_enc, tags_enc]])

    prediction = gb_model.predict(user_data)[0]

    # Example label map
    label_map = {0: "No-Show", 1: "Completed", 2: "Cancelled"}
    predicted_status = label_map.get(prediction, f"Class {prediction}")

    st.success(f"✅ Predicted Status: **{predicted_status}**")
