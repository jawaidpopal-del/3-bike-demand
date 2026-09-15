import streamlit as st
import pandas as pd
import pickle
from src.llm_service import explain_prediction

MODEL_PATH = "models/gradient_boosting_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("Seoul Bike Demand Predictor")
st.write("Enter the conditions below to predict hourly bike rental demand.")

st.header("Enter Bike Demand Conditions")

hour = st.number_input("Hour", min_value=0, max_value=23, value=8)
temperature = st.number_input("Temperature (°C)", value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=2.0)
visibility = st.number_input("Visibility (10m)", min_value=0, value=1500)
solar_radiation = st.number_input("Solar Radiation (MJ/m2)", min_value=0.0, value=0.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
snowfall = st.number_input("Snowfall (cm)", min_value=0.0, value=0.0)

holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])


if st.button("Predict Demand"):

    if temperature < -30 or temperature > 50:
        st.error("Please enter a realistic temperature.")
        st.stop()

    with open("data/processed/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    input_data = pd.DataFrame([{
        "Hour": hour,
        "Temperature(°C)": temperature,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Solar Radiation (MJ/m2)": solar_radiation,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
    }])

    numeric_columns = [
        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
    ]

    input_data[numeric_columns] = scaler.transform(
        input_data[numeric_columns]
    )

    input_data["is_peak_hour"] = int(18 <= hour <= 22)
    input_data["is_night"] = int(hour < 6)
    input_data["is_working_day"] = 1
    input_data["is_holiday"] = int(holiday == "Holiday")

    input_data["season_Spring"] = int(season == "Spring")
    input_data["season_Summer"] = int(season == "Summer")
    input_data["season_Winter"] = int(season == "Winter")

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted bike rentals: {prediction:.0f}")

    try:
        explanation = explain_prediction(
            prediction,
            hour,
            temperature
        )

        st.subheader("AI Explanation")
        st.write(explanation)

    except Exception:
        st.warning(
            "The local LLM is unavailable, "
            "but the bike demand prediction is still available."
        )