import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("appointments.xlsx")

# 1️⃣ Select Date Column
date_column = st.selectbox("Select Date Column", df.select_dtypes(include=['datetime','object']).columns)

# 2️⃣ Select Numeric Column for Heatmap
numeric_column = st.selectbox("Select Numeric Column for Heatmap", df.select_dtypes(include=['int','float']).columns)

if date_column and numeric_column:
    # Convert date column to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Group by date (or pivot for heatmap)
    heatmap_data = df.groupby(df[date_column].dt.date)[numeric_column].sum().to_frame()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)
