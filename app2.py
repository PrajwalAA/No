import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet

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
            # --- 1. Descriptive Statistics Heatmap ---
            grouped = df_clean.groupby(df_clean[date_column].dt.date)[numeric_column]
            stats_df = pd.DataFrame({
                "Mean": grouped.mean(),
                "Median": grouped.median(),
                "Mode": grouped.apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan),
                "25th Percentile": grouped.quantile(0.25),
                "50th Percentile": grouped.quantile(0.50),
                "75th Percentile": grouped.quantile(0.75)
            })

            st.subheader("Descriptive Statistics Heatmap")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(stats_df.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Statistic")
            st.pyplot(fig)

            # --- Overall insights ---
            st.subheader("Overall Insights")
            st.write(f"**Overall Mean:** {df_clean[numeric_column].mean():.2f}")
            st.write(f"**Overall Median:** {df_clean[numeric_column].median():.2f}")
            st.write(f"**Overall Mode:** {df_clean[numeric_column].mode()[0]}")
            st.write(f"**25th Percentile:** {df_clean[numeric_column].quantile(0.25):.2f}")
            st.write(f"**50th Percentile:** {df_clean[numeric_column].quantile(0.50):.2f}")
            st.write(f"**75th Percentile:** {df_clean[numeric_column].quantile(0.75):.2f}")

            # Top 5 dates with highest mean
            st.write("**Top 5 Dates with Highest Mean:**")
            st.dataframe(stats_df['Mean'].sort_values(ascending=False).head(5))
            # Top 5 dates with lowest median
            st.write("**Top 5 Dates with Lowest Median:**")
            st.dataframe(stats_df['Median'].sort_values().head(5))

            # --- 2. Forecasting using Prophet ---
            st.subheader("Forecasting Future Hype & Lows")

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

            # Plot forecast
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(forecast['ds'], forecast['yhat'], label='Forecast')
            ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
            ax2.scatter(df_forecast['ds'], df_forecast['y'], color='red', label='Actual')
            ax2.set_xlabel("Date")
            ax2.set_ylabel(numeric_column)
            ax2.set_title("Forecast of Numeric Column")
            ax2.legend()
            st.pyplot(fig2)

            # Identify hype and low dates
            hype_threshold = forecast['yhat'].quantile(0.75)
            low_threshold = forecast['yhat'].quantile(0.25)
            hype_dates = forecast[forecast['yhat'] >= hype_threshold]['ds']
            low_dates = forecast[forecast['yhat'] <= low_threshold]['ds']

            st.write("**Predicted Hype Dates (High Values):**")
            st.dataframe(hype_dates.dt.date.reset_index(drop=True))
            st.write("**Predicted Low Dates (Low Values):**")
            st.dataframe(low_dates.dt.date.reset_index(drop=True))

else:
    st.info("Please upload an Excel file to continue.")
