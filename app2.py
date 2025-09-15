import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load data
df = pd.read_excel("appointments.xlsx")

# --- Select columns ---
date_column = st.selectbox("Select Date Column", df.select_dtypes(include=['datetime','object']).columns)
numeric_column = st.selectbox("Select Numeric Column", df.select_dtypes(include=['int','float']).columns)

if date_column and numeric_column:
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Drop rows with invalid data
    df_clean = df.dropna(subset=[date_column, numeric_column])

    if df_clean.empty:
        st.warning("No valid data to plot. Please check your selected columns.")
    else:
        # Group by date
        grouped = df_clean.groupby(df_clean[date_column].dt.date)[numeric_column]

        # Calculate statistics
        stats_df = pd.DataFrame({
            "Mean": grouped.mean(),
            "Median": grouped.median(),
            "Mode": grouped.apply(lambda x: stats.mode(x)[0][0] if not x.empty else np.nan),
            "25th Percentile": grouped.quantile(0.25),
            "50th Percentile": grouped.quantile(0.50),
            "75th Percentile": grouped.quantile(0.75)
        })

        st.subheader("Descriptive Statistics Heatmap")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(stats_df.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        # --- Extractable Insights ---
        st.subheader("Insights from Data")
        st.write(f"**Overall Mean:** {df_clean[numeric_column].mean():.2f}")
        st.write(f"**Overall Median:** {df_clean[numeric_column].median():.2f}")
        st.write(f"**Overall Mode:** {stats.mode(df_clean[numeric_column])[0][0]}")
        st.write(f"**Overall 25th Percentile:** {df_clean[numeric_column].quantile(0.25):.2f}")
        st.write(f"**Overall 50th Percentile:** {df_clean[numeric_column].quantile(0.50):.2f}")
        st.write(f"**Overall 75th Percentile:** {df_clean[numeric_column].quantile(0.75):.2f}")

        # Top 5 dates with highest mean
        top_mean = stats_df['Mean'].sort_values(ascending=False).head(5)
        st.write("**Top 5 Dates with Highest Mean:**")
        st.dataframe(top_mean)

        # Top 5 dates with lowest median
        low_median = stats_df['Median'].sort_values().head(5)
        st.write("**Top 5 Dates with Lowest Median:**")
        st.dataframe(low_median)
