import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("appointments.xlsx")

# --- Select columns ---
date_column = st.selectbox("Select Date Column", df.select_dtypes(include=['datetime','object']).columns)
numeric_column = st.selectbox("Select Numeric Column for Heatmap", df.select_dtypes(include=['int','float']).columns)

if date_column and numeric_column:
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Drop rows with invalid data
    df_clean = df.dropna(subset=[date_column, numeric_column])

    if df_clean.empty:
        st.warning("No valid data to plot. Please check your selected columns.")
    else:
        # Group by date
        heatmap_data = df_clean.groupby(df_clean[date_column].dt.date)[numeric_column].sum().to_frame()
        
        if heatmap_data.empty:
            st.warning("No data after grouping. Check your date and numeric columns.")
        else:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(heatmap_data.T, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
