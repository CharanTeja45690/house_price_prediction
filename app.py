import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ===== Load trained model =====
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "house_price_rf_model.joblib"

@st.cache_resource
def load_model():
    # Load the already-trained RandomForest model
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ===== UI =====
st.sidebar.title("Input Features")

MedInc = st.sidebar.number_input("Median Income (10k USD)", min_value=0.0, value=3.0, step=0.1)
HouseAge = st.sidebar.number_input("House Age", min_value=1.0, value=20.0, step=1.0)
AveRooms = st.sidebar.number_input("Average Rooms", min_value=1.0, value=5.0, step=0.1)
AveBedrms = st.sidebar.number_input("Average Bedrooms", min_value=0.5, value=1.0, step=0.1)
Population = st.sidebar.number_input("Population", min_value=1.0, value=1000.0, step=1.0)
AveOccup = st.sidebar.number_input("Average Occupancy", min_value=1.0, value=3.0, step=0.1)
Latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01)
Longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-118.0, step=0.01)

input_data = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude],
})

st.title("🏡 House Price Prediction App")
st.subheader("Input Summary")
st.write(input_data)

if st.button("Predict Price"):
    # model is already trained & loaded – just predict
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted median house value: ${prediction * 100000:,.2f}")
