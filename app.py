import streamlit as st
import pickle
import numpy as np

# --- Add background image using CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://acko-cms.ackoassets.com/how_to_sell_a_used_car_in_india_9ea7a4dc00.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🚗 Resale Value Car Price Predictor")

# User Inputs
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
distance = st.number_input("Distance Driven (in km)", min_value=0, step=500)
owner = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth or more"])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "MUV", "Luxury"])
drive_manual = st.selectbox("Transmission Type", ["Manual", "Automatic"])
brand = st.selectbox("Car Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "BMW", "Mercedes", "Audi", "Tata", "Mahindra"])

# Convert categorical inputs to numeric
owner_mapping = {"First": 1, "Second": 2, "Third": 3, "Fourth or more": 4}
fuel_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
type_mapping = {"Hatchback": 0, "Sedan": 1, "SUV": 2, "MUV": 3, "Luxury": 4}
transmission_mapping = {"Manual": 1, "Automatic": 0}
brand_mapping = {
    "Maruti": 0, "Hyundai": 1, "Honda": 2, "Toyota": 3, "Ford": 4,
    "BMW": 5, "Mercedes": 6, "Audi": 7, "Tata": 8, "Mahindra": 9
}

# Apply mappings
owner = owner_mapping[owner]
fuel = fuel_mapping[fuel]
car_type = type_mapping[car_type]
drive_manual = transmission_mapping[drive_manual]
brand = brand_mapping[brand]

# Prediction
if st.button("Predict Price 💰"):
    try:
        input_data = np.array([[year, distance, owner, fuel, car_type, drive_manual, brand]], dtype=np.float32)
        price = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹{price:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
