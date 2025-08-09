# 🌾 CropMate

**Deployed App:** [🌱 Crop Recommendation System](https://pranshugarg-cropmate.streamlit.app/)

CropMate is a **machine learning-powered web app** that helps farmers and agricultural experts recommend the best crop to grow based on **soil nutrients, weather conditions, and rainfall**.

---

## 🖼 🚀 Live Preview

*Insert your latest app screenshot here*

---

## 🚀 Features

- 📊 Predict the best crop using:
  - **Soil nutrients:** Nitrogen (N), Phosphorus (P), Potassium (K)
  - **Weather data:** Temperature (°C), Humidity (%)
  - **Soil pH level** and **Rainfall (mm)**
- 🌍 **Auto-fetch Temperature and Humidity** by entering your city name via OpenWeatherMap API integration  
- 🔍 Input values are **scaled with MinMaxScaler and StandardScaler** before prediction for better accuracy  
- 📄 Generates a **downloadable PDF report** summarizing input parameters, recommended crop, and crop info  
- 🌱 Supports recommendations for **22 different crops**

---

## 🛠 Tech Stack

- Python  
- Streamlit (UI)  
- Scikit-learn (ML model)  
- NumPy (data processing)  
- Requests (API calls)  
- FPDF (PDF generation)  
- Pickle (model serialization)

---

## 📦 Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pranshugarg-dev/crop-recommendation.git
cd crop-recommendation
