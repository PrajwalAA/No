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
label_map = {0: "No-Show", 1: "Show"}

# --- 2. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Demand & Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Demand & Appointment Status Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Appointment Status Prediction", "🗓️ Daily Demand Heatmap"])

# --- TAB 1: APPOINTMENT STATUS PREDICTION ---
with tab1:
    st.header("Predict an Appointment's Status")
    
    model_filename = "gradient_boosting_model2.pkl"
    try:
        gb_model = joblib.load(model_filename)
        model_loaded = True
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found.")
        st.info("Please ensure the model file is in the same directory as the script.")
        model_loaded = False
        
    if model_loaded:
        # Direct input for lead time
        st.subheader("Appointment Lead Time")
        lead_time_minutes = st.number_input(
            "⏱️ Lead Time (minutes)", 
            min_value=0, max_value=10080, step=10, value=60
        )

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
with tab2:    
    # --- Streamlit page configuration ---
    st.set_page_config(page_title="Appointments Analytics & Forecasting", layout="wide")
    st.title("Appointments Analytics & Date-wise Forecasting")
    
    # --- Upload Excel file ---
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        
        # --- Select columns ---
        date_column = st.selectbox(
            "Select Date Column", df.select_dtypes(include=['datetime','object']).columns
        )
        numeric_column = st.selectbox(
            "Select Numeric Column", df.select_dtypes(include=['int','float']).columns
        )
    
        if date_column and numeric_column:
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df_clean = df.dropna(subset=[date_column, numeric_column])
    
            if df_clean.empty:
                st.warning("No valid data to analyze. Please check your selected columns.")
            else:
                # ==============================================================
                # 1. Business Metrics + Descriptive Statistics
                # ==============================================================
                st.subheader("Business Metrics & Statistics Heatmap")

                grouped = df_clean.groupby(df_clean[date_column].dt.date)[numeric_column]

                # Business-oriented metrics
                daily_total = grouped.count().rename("Total Appointments")
                if "price" in df_clean.columns:
                    daily_revenue = df_clean.groupby(df_clean[date_column].dt.date)["price"].sum().rename("Revenue")
                else:
                    daily_revenue = pd.Series(dtype=float)

                if "status" in df_clean.columns:
                    status_counts = df_clean.groupby([df_clean[date_column].dt.date, "status"]).size().unstack(fill_value=0)
                else:
                    status_counts = pd.DataFrame()

                # Descriptive statistics
                stats_df = pd.DataFrame({
                    "Mean": grouped.mean(),
                    "Median": grouped.median(),
                    "Mode": grouped.apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan),
                    "25th Percentile": grouped.quantile(0.25),
                    "50th Percentile": grouped.quantile(0.50),
                    "75th Percentile": grouped.quantile(0.75)
                })

                # Combine business + descriptive
                business_df = pd.concat([daily_total, daily_revenue, status_counts, stats_df], axis=1).fillna(0)

                # Heatmap
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.heatmap(business_df.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, 
                            cbar_kws={'label': 'Value'})
                ax.set_xlabel("Date")
                ax.set_ylabel("Metrics & Statistics")
                st.pyplot(fig)

                # ==============================================================
                # 2. Insights
                # ==============================================================
                st.subheader("Business Insights")
                st.write("**Top 5 Days by Appointments:**")
                st.dataframe(business_df.sort_values("Total Appointments", ascending=False).head(5))

                if "Revenue" in business_df.columns:
                    st.write("**Top 5 Days by Revenue:**")
                    st.dataframe(business_df.sort_values("Revenue", ascending=False).head(5))

                st.write("**Top 5 Days by Mean Value:**")
                st.dataframe(business_df.sort_values("Mean", ascending=False).head(5))

                st.write("**Top 5 Dates with Lowest Median:**")
                st.dataframe(business_df.sort_values("Median").head(5))

                # Overall stats
                st.subheader("Overall Statistics")
                st.write(f"**Overall Mean:** {df_clean[numeric_column].mean():.2f}")
                st.write(f"**Overall Median:** {df_clean[numeric_column].median():.2f}")
                st.write(f"**Overall Mode:** {df_clean[numeric_column].mode()[0]}")
                st.write(f"**25th Percentile:** {df_clean[numeric_column].quantile(0.25):.2f}")
                st.write(f"**50th Percentile:** {df_clean[numeric_column].quantile(0.50):.2f}")
                st.write(f"**75th Percentile:** {df_clean[numeric_column].quantile(0.75):.2f}")

                # ==============================================================
                # 3. Forecasting with Prophet
                # ==============================================================
                st.subheader("Forecasting Future Hype & Lows (Interactive)")

                df_forecast = df_clean[[date_column, numeric_column]].rename(columns={
                    date_column: 'ds', numeric_column: 'y'
                })
                df_forecast = df_forecast.groupby('ds').sum().reset_index().dropna(subset=['ds', 'y'])

                model = Prophet(daily_seasonality=True)
                model.fit(df_forecast)

                # Forecast next 30 days
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                # Identify hype/low
                hype_threshold = forecast['yhat'].quantile(0.75)
                low_threshold = forecast['yhat'].quantile(0.25)
                hype_dates = pd.to_datetime(forecast[forecast['yhat'] >= hype_threshold]['ds'])
                low_dates = pd.to_datetime(forecast[forecast['yhat'] <= low_threshold]['ds'])

                # Plot forecast
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'], mode='lines',
                    name='Forecast', line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                    line=dict(color='lightblue'), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                    fill='tonexty', line=dict(color='lightblue'), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=df_forecast['ds'], y=df_forecast['y'], mode='markers',
                    name='Actual', marker=dict(color='red', size=6)
                ))
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

                # Show hype/low dates
                st.write("**Predicted Hype Dates (High Values):**")
                st.dataframe(hype_dates.dt.date.reset_index(drop=True))
                st.write("**Predicted Low Dates (Low Values):**")
                st.dataframe(low_dates.dt.date.reset_index(drop=True))

    else:
        st.info("Please upload an Excel file to continue.")
