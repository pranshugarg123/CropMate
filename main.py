import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import requests
from fpdf import FPDF
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="🌾 CropMate", layout="wide", page_icon="🌱")

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
    selected = option_menu(
        "CropMate",
        ["Predict Crop"],
        menu_icon="leaf",
        icons=["search"],
        default_index=0
    )

# Function to fetch weather data
def fetch_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return data["main"]["temp"], data["main"]["humidity"]
        else:
            return None, None
    except:
        return None, None

# Function to generate PDF
def generate_pdf(inputs, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Crop Recommendation Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, "Input Parameters:", ln=True)
    for key, val in inputs.items():
        pdf.cell(200, 10, f"{key}: {val}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, f"Recommended Crop: {prediction}", ln=True)

    file_path = "crop_recommendation_report.pdf"
    pdf.output(file_path)
    return file_path

# Crop Prediction Page
if selected == "Predict Crop":
    st.title("🌾 CropMate")
    st.subheader("Your Smart Assistant for Crop Recommendation")
    st.markdown(
        "CropMate helps farmers and agri-experts make better crop choices "
        "by analyzing soil nutrients, weather conditions, and rainfall. 🌱\n\n"
        "**Just enter the details below to get the best crop suggestion for your land.**"
    )
    st.divider()

    # Weather API option
    use_api = st.checkbox("Fetch Temperature & Humidity from City (Weather API)")
    temperature = None
    humidity = None

    if use_api:
        api_key = st.text_input("Enter your OpenWeatherMap API Key:")
        city = st.text_input("Enter your city name:")
        if api_key and city:
            temp, hum = fetch_weather(city, api_key)
            if temp is not None:
                temperature = temp
                humidity = hum
                st.success(f"Fetched Weather → Temperature: {temp}°C, Humidity: {hum}%")
            else:
                st.error("Failed to fetch weather data. Check API key or city name.")

    # User input form
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N) (kg/ha)", min_value=0, max_value=200)
        P = st.number_input("Phosphorus (P) (kg/ha)", min_value=0, max_value=200)
        K = st.number_input("Potassium (K) (kg/ha)", min_value=0, max_value=200)

    with col2:
        if temperature is None:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
        if humidity is None:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
        pH = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0)

    with col3:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0)

    st.markdown(
        "ℹ️ **Input Tips:**\n"
        "- Use recent soil test reports to get N, P, K, and pH values.\n"
        "- Use average temperature, humidity, and rainfall for your region.\n"
    )

    # Predict Crop
    if st.button("🔍 Recommend Crop"):
        user_input = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        # Apply scalers
        user_input_scaled = minmax_scaler.transform(user_input)
        user_input_final = standard_scaler.transform(user_input_scaled)

        # Make prediction
        predicted_label = model.predict(user_input_final)[0]
        predicted_crop = crop_dict.get(predicted_label, "Unknown Crop")

        st.success(f"🌱 **Recommended Crop: {predicted_crop}**")
        st.info("✅ This crop is best suited to your current soil and climate conditions.")

        # PDF Download
        inputs = {
            "Nitrogen": N,
            "Phosphorus": P,
            "Potassium": K,
            "Temperature": temperature,
            "Humidity": humidity,
            "pH": pH,
            "Rainfall": rainfall
        }
        pdf_path = generate_pdf(inputs, predicted_crop)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Recommendation Report",
                data=pdf_file,
                file_name="crop_recommendation_report.pdf",
                mime="application/pdf"
            )

    # Footer
    st.divider()
    st.markdown(
        "<center><small>🔬 Powered by Machine Learning | 🤝 Designed for Farmers | 🌍 Project: CropMate</small></center>",
        unsafe_allow_html=True
    )
