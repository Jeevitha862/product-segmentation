import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Title
st.title("🛍️ Product Segmentation using K-Means")

# Upload CSV
uploaded_file = st.file_uploader("Upload your inventory_monitoring.csv file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # Clean column names

    st.subheader("🔍 Preview of Data")
    st.write(data.head())

    # Check for required columns
    if 'Stock Levels' in data.columns and 'Supplier Lead Time (days)' in data.columns:
        features = data[['Stock Levels', 'Supplier Lead Time (days)']].fillna(0)

        # KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        data['Segment'] = kmeans.fit_predict(features)

        st.subheader("📊 Clustered Data")
        st.write(data)

        # Plot
        st.subheader("📈 Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(data['Stock Levels'], data['Supplier Lead Time (days)'],
                             c=data['Segment'], cmap='viridis')
        ax.set_xlabel('Stock Levels')
        ax.set_ylabel('Supplier Lead Time (days)')
        st.pyplot(fig)
    else:
        st.error("The CSV must contain 'Stock Levels' and 'Supplier Lead Time (days)' columns.")
else:
    st.info("Please upload a CSV file to proceed.")
