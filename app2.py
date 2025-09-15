import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Peak Hours Analysis", layout="wide")
st.title("Peak Hours Graph")

# --- Upload Excel file ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Select datetime column ---
    datetime_column = st.selectbox(
        "Select DateTime Column", df.select_dtypes(include=['datetime','object']).columns
    )

    if datetime_column:
        # Convert to datetime
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        df_clean = df.dropna(subset=[datetime_column])

        if df_clean.empty:
            st.warning("No valid datetime data found.")
        else:
            # Extract hour from datetime
            df_clean['Hour'] = df_clean[datetime_column].dt.hour

            # Count number of entries per hour
            hourly_counts = df_clean.groupby('Hour').size()

            # Plot Peak Hours
            st.subheader("Peak Hours of the Day")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette="viridis", ax=ax)
            ax.set_xlabel("Hour of the Day")
            ax.set_ylabel("Number of Appointments")
            ax.set_title("Peak Hours Analysis")
            st.pyplot(fig)

            # Optional: show table of counts
            st.write("Hourly Counts Table:")
            st.dataframe(hourly_counts.reset_index(name="Count"))

else:
    st.info("Please upload an Excel file to continue.")

