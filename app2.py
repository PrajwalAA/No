import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Appointments Heatmap", layout="wide")

st.title("Appointments Statistics Heatmap")

# --- Load data ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Select columns ---
    date_column = st.selectbox(
        "Select Date Column", df.select_dtypes(include=['datetime','object']).columns
    )
    numeric_column = st.selectbox(
        "Select Numeric Column for Heatmap", df.select_dtypes(include=['int','float']).columns
    )

    if date_column and numeric_column:
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df_clean = df.dropna(subset=[date_column, numeric_column])

        if df_clean.empty:
            st.warning("No valid data to plot. Please check your selected columns.")
        else:
            # Group by date
            grouped = df_clean.groupby(df_clean[date_column].dt.date)[numeric_column]

            # Calculate statistics safely
            stats_df = pd.DataFrame({
                "Mean": grouped.mean(),
                "Median": grouped.median(),
                "Mode": grouped.apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan),
                "25th Percentile": grouped.quantile(0.25),
                "50th Percentile": grouped.quantile(0.50),
                "75th Percentile": grouped.quantile(0.75)
            })

            # --- Heatmap ---
            st.subheader("Descriptive Statistics Heatmap")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(stats_df.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Statistic")
            st.pyplot(fig)

            # --- Extractable Insights ---
            st.subheader("Overall Insights")
            st.write(f"**Overall Mean:** {df_clean[numeric_column].mean():.2f}")
            st.write(f"**Overall Median:** {df_clean[numeric_column].median():.2f}")
            st.write(f"**Overall Mode:** {df_clean[numeric_column].mode()[0]}")
            st.write(f"**Overall 25th Percentile:** {df_clean[numeric_column].quantile(0.25):.2f}")
            st.write(f"**Overall 50th Percentile:** {df_clean[numeric_column].quantile(0.50):.2f}")
            st.write(f"**Overall 75th Percentile:** {df_clean[numeric_column].quantile(0.75):.2f}")

            # Top 5 dates with highest mean
            st.write("**Top 5 Dates with Highest Mean:**")
            st.dataframe(stats_df['Mean'].sort_values(ascending=False).head(5))

            # Top 5 dates with lowest median
            st.write("**Top 5 Dates with Lowest Median:**")
            st.dataframe(stats_df['Median'].sort_values().head(5))
else:
    st.info("Please upload an Excel file to continue.")
