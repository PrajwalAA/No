import streamlit as st
import pandas as pd
import calplot
import matplotlib.pyplot as plt

st.header("Daily Demand Heatmap from Excel Dataset")

# Upload the file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "csv"])
if uploaded_file:
    try:
        # Load Excel/CSV
        if uploaded_file.name.endswith("appointments.xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("Preview of dataset:", df.head())

        # Ask user to select date and numeric columns
        date_col = st.selectbox("Select Date Column", options=df.columns, index=df.columns.get_loc("booking_date"))
        value_col = st.selectbox("Select Numeric Column for Heatmap", options=df.columns, index=df.columns.get_loc("lead_time_minutes"))

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Aggregate numeric values per day
        daily_data = df.groupby(df[date_col])[value_col].sum()
        
        # Sort by date
        daily_data = daily_data.sort_index()

        # Generate calplot heatmap
        fig, ax = calplot.calplot(
            daily_data,
            suptitle=f"Daily {value_col} Heatmap",
            cmap="YlGnBu",
            edgecolor="gray",
            linewidth=0.5,
            figsize=(15, 6),
            monthlabels=True
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.info("Upload an Excel or CSV file to visualize the heatmap.") combine import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
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

tab1, tab2 = st.tabs(["🔮 Appointment Status Prediction", "🗓️ Demand Forecast Heatmap"])

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
            appointment_time = st.time_input("⏰ Appointment Time", (now + pd.Timedelta(hours=1)).time())

        booking_datetime = datetime.combine(booking_date, booking_time)
        appointment_datetime = datetime.combine(appointment_date, appointment_time)
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

# --- TAB 2: DEMAND FORECAST HEATMAP ---
with tab2:
    st.header("Daily Demand Forecast Heatmap")
    st.write("Visualize demand patterns over time to identify trends and seasonality.")

    @st.cache_data
    def generate_demand_data(start_year, end_year):
        dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31')
        np.random.seed(42)
        base_demand = np.random.normal(loc=100, scale=15, size=len(dates))
        weekly_pattern = np.cos(2 * np.pi * dates.dayofweek / 7) * 20
        yearly_pattern = np.sin(2 * np.pi * dates.dayofyear / 365.25) * 30
        demand_values = base_demand + weekly_pattern + yearly_pattern
        demand_series = pd.Series(demand_values, index=dates)
        spike_dates = pd.to_datetime([f'{2025}-07-04', f'{2025}-12-25'])
        demand_series.loc[demand_series.index.isin(spike_dates)] += 150
        return demand_series.astype(int).clip(lower=0)

    col6, col7 = st.columns(2)
    with col6:
        start_year = st.selectbox("Start Year", options=list(range(2024, 2027)), index=1)
    with col7:
        end_year = st.selectbox("End Year", options=list(range(2024, 2027)), index=1)

    if start_year > end_year:
        st.warning("The start year must be less than or equal to the end year.")
    else:
        demand_data = generate_demand_data(start_year, end_year)
        try:
            fig = calplot.calplot(
                demand_data,
                suptitle=f'Daily Demand Forecast for {start_year}-{end_year}',
                cmap='YlGnBu',
                colorbar=True,
                edgecolor='gray',
                linewidth=0.5,
                figsize=(15, 6),
                yearlabel_kws={'fontsize': 14, 'color': 'gray'},
                monthlabels=True
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while generating the heatmap: {e}")
this 
