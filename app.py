import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('vehicle_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App
st.title("🚗 Fuel Efficiency Predictor")

# Inputs
origin = st.selectbox("Origin", [1, 2, 3])
cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8])
displacement = st.number_input("Displacement", 50.0, 500.0, 140.0)
horsepower = st.number_input("Horsepower", 40, 250, 100)
weight = st.number_input("Weight", 1500, 5500, 3000)
acceleration = st.number_input("Acceleration", 8.0, 25.0, 15.0)
year = st.slider("Year", 1970, 1982, 1976)

# Predict
if st.button("Predict"):
    # Prepare data
    data = [[origin, cylinders, displacement, horsepower, 
             weight, acceleration, year]]
    
    # Scale and predict
    scaled = scaler.transform(data)
    result = model.predict(scaled, verbose=0)[0][0]
    
    # Show result
    st.write(f"### {result:.2f} km/L")