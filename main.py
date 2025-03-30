import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Set page configuration
st.set_page_config(page_title="🌾 Crop Recommendation System",
                   layout="wide",
                   page_icon="🌱")

# Load the model
pkl_path = "crop_recommendation.pkl"
with open(pkl_path, "rb") as file:
    data = pickle.load(file)

model = data["model"]
minmax_scaler = data["minmax_scaler"]
standard_scaler = data["standard_scaler"]

# Crop label dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Sidebar navigation
with st.sidebar:
    selected = option_menu("Crop Recommendation",
                           ["Predict Crop"],
                           menu_icon="leaf",
                           icons=["search"],
                           default_index=0)

# Crop Prediction Page
if selected == "Predict Crop":
    st.title("🌾 Crop Recommendation System")
    st.write("Enter soil and weather parameters to get the best crop recommendation.")


    # User input form
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N) (kg/ha)", min_value=0, max_value=200)
        P = st.number_input("Phosphorus (P) (kg/ha)", min_value=0, max_value=200)
        K = st.number_input("Potassium (K) (kg/ha)", min_value=0, max_value=200)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
        pH = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0)

    with col3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0)

    # Predict Crop
    if st.button("🔍 Recommend Crop"):
        user_input = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Apply scalers
        user_input_scaled = minmax_scaler.transform(user_input)
        user_input_final = standard_scaler.transform(user_input_scaled)

        # Make prediction
        predicted_label = model.predict(user_input_final)[0]
        predicted_crop = crop_dict.get(predicted_label, "Unknown Crop")

        st.success(f"🌱 Recommended Crop: **{predicted_crop}**")
