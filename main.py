import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import requests
from fpdf import FPDF
from datetime import datetime

# -------------------
# Streamlit page config
# -------------------
st.set_page_config(page_title="🌾 CropMate", layout="wide", page_icon="🌱")

# -------------------
# Load model & scalers
# -------------------
pkl_path = "crop_recommendation.pkl"
with open(pkl_path, "rb") as file:
    data = pickle.load(file)

model = data["model"]
minmax_scaler = data["minmax_scaler"]
standard_scaler = data["standard_scaler"]

# Crop labels
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# -------------------
# Sidebar navigation
# -------------------
with st.sidebar:
    selected = option_menu(
        "CropMate",
        ["Predict Crop"],
        menu_icon="leaf",
        icons=["search"],
        default_index=0
    )

# -------------------
# Weather API helper
# -------------------
def fetch_weather(city):
    """Fetch temperature and humidity for a city using OpenWeatherMap API key from secrets."""
    api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code == 200 and "main" in data:
            return data["main"]["temp"], data["main"]["humidity"]
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None, None


# -------------------
# PDF generation helper
# -------------------
def generate_pdf(inputs, prediction):
    crop_info = {
        "Rice": "Rice is a staple food crop with high demand worldwide. Requires flooded fields and moderate temperature.",
        "Maize": "Maize is versatile and used for food, fodder, and biofuel. Prefers warm climate with adequate rainfall.",
        "Jute": "Jute is a fiber crop, important for textile industries. Grows well in warm and humid climate.",
        "Cotton": "Cotton is used in textile manufacturing. Thrives in warm climates with moderate rainfall.",
        "Coconut": "Coconut palms require coastal tropical climates and well-drained sandy soils.",
        "Papaya": "Papaya is a tropical fruit crop with high vitamin content, grows best in warm regions.",
        "Orange": "Oranges require subtropical climates and well-drained soils for best fruit quality.",
        "Apple": "Apples thrive in temperate climates with cold winters and warm summers.",
        "Muskmelon": "Muskmelons grow best in warm climates with plenty of sunlight.",
        "Watermelon": "Watermelons require warm weather and sandy loam soils.",
        "Grapes": "Grapes are grown for fresh fruit and wine; prefer warm, dry climates.",
        "Mango": "Mangoes flourish in tropical and subtropical climates with dry seasons.",
        "Banana": "Bananas grow well in tropical areas with rich, well-drained soils.",
        "Pomegranate": "Pomegranates prefer semi-arid climates with cool winters.",
        "Lentil": "Lentils are cool-season legumes, important for protein-rich diets.",
        "Blackgram": "Blackgram is a pulse crop grown in warm climates, improves soil fertility.",
        "Mungbean": "Mungbeans are short-duration legumes grown in warm climates.",
        "Mothbeans": "Mothbeans are drought-tolerant legumes, suitable for arid regions.",
        "Pigeonpeas": "Pigeonpeas are drought-tolerant legumes, important in dry farming.",
        "Kidneybeans": "Kidneybeans are grown in temperate to subtropical climates.",
        "Chickpea": "Chickpeas prefer cool climates and are a major pulse crop.",
        "Coffee": "Coffee grows best in tropical highlands with rich soil."
    }
    class PDF(FPDF):
        def header(self):
            # Logo (optional, add your logo file if you have one)
            # self.image('logo.png', 10, 8, 33)
            self.set_font('Arial', 'B', 16)
            self.set_text_color(0, 102, 0)  # Dark green
            self.cell(0, 10, '🌾 CropMate Crop Recommendation Report', 0, 1, 'C')
            self.ln(5)
            self.set_draw_color(0, 102, 0)
            self.set_line_width(0.5)
            self.line(10, 25, 200, 25)
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()} | Powered by CropMate 🌱', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()

    # Date and time
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100)
    pdf.cell(0, 10, f'Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
    pdf.ln(5)

    # Input parameters as a table
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 51, 0)
    pdf.cell(0, 10, 'Input Parameters:', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    col_width = 50
    row_height = 8
    for key, val in inputs.items():
        pdf.set_fill_color(224, 235, 224)  # Light green background
        pdf.cell(col_width, row_height, str(key), border=1, fill=True)
        pdf.cell(col_width, row_height, str(val), border=1, ln=1, fill=False)

    pdf.ln(10)

    # Recommended Crop Highlight
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 102, 51)
    pdf.cell(0, 10, f"Recommended Crop: {prediction}", 0, 1, 'C')
    pdf.ln(5)

    # Crop Description
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    info_text = crop_info.get(prediction, "No additional information available for this crop.")
    pdf.multi_cell(0, 8, info_text)

    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(80)
    pdf.cell(0, 10, "Thank you for using CropMate! Wishing you a fruitful harvest. 🌱", 0, 1, 'C')

    file_path = "crop_recommendation_report.pdf"
    pdf.output(file_path)
    return file_path


# -------------------
# Crop Prediction Page
# -------------------
if selected == "Predict Crop":
    st.title("🌾 CropMate")
    st.subheader("Your Smart Assistant for Crop Recommendation")
    st.markdown(
        "CropMate helps farmers and agri-experts make better crop choices "
        "by analyzing soil nutrients, weather conditions, and rainfall. 🌱"
    )
    st.divider()

    # Weather autofill
    temperature, humidity = None, None
    use_api = st.checkbox("📍 Auto-fill Temperature & Humidity using City Name")

    if use_api:
        city = st.text_input("Enter your city name:")
        if city:
            temp, hum = fetch_weather(city)
            if temp is not None:
                temperature, humidity = temp, hum
                st.success(f"Weather fetched: 🌡 {temp}°C, 💧 {hum}%")
            else:
                st.error("❌ Could not fetch weather data. Please enter manually.")

    # User inputs
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

    # Prediction button
    if st.button("🔍 Recommend Crop"):
        try:
            user_input = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

            # Scale features
            user_input_scaled = minmax_scaler.transform(user_input)
            user_input_final = standard_scaler.transform(user_input_scaled)

            # Predict
            predicted_label = model.predict(user_input_final)[0]
            predicted_crop = crop_dict.get(predicted_label, "Unknown Crop")

            st.success(f"🌱 **Recommended Crop: {predicted_crop}**")

            # Generate PDF
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
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    st.divider()
    st.markdown(
        "<center><small>🔬 Powered by Machine Learning | 🤝 Designed for Farmers | 🌍 Project: CropMate</small></center>",
        unsafe_allow_html=True
    )
