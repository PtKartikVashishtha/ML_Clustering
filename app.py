import streamlit as st
import joblib
import numpy as np

# Load the saved KMeans model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set the title for your web app
st.title("Customer Segmentation App")

# Create input fields for the user
income = st.slider("Annual Income (k$)", 0, 150, 60)
score = st.slider("Spending Score (1-100)", 0, 100, 50)

# Create a button for the user to predict their cluster
if st.button("Predict Cluster"):
    # Transform the input data using the scaler
    data = scaler.transform([[income, score]])
    
    # Predict the cluster using the model
    cluster = model.predict(data)[0]
    
    # Display the result to the user
    st.success(f"The customer belongs to Cluster #{cluster}")
