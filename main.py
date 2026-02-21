
import streamlit as st
import pandas as pd
import pickle
import base64
import requests
import os

st.set_page_config(page_title="Delivery Time Prediction", layout="centered")



MODEL_URL = "https://drive.google.com/uc?export=download&id=1ytdFFFN_4Eb4JAEn9KXicK6PYTg2mIxf"
MODEL_PATH = "delivery_time_pipeline67.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()



def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-container {{
        background-color: rgba(255, 255, 255, 0.90);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
    }}

    .stButton>button {{
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        border: none;
    }}

    .stButton>button:hover {{
        background-color: #e63e3e;
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Make sure sdcvbn.png is in same folder as app.py
set_bg("sdcvbn.png")



st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🚴 Food Delivery Time Prediction")
st.write("Enter order and delivery details to predict delivery time")

distance = st.number_input("Distance (km)", min_value=0.1, value=5.0)
ratings = st.slider("Restaurant Rating", 1.0, 5.0, 4.0)
age = st.number_input("Delivery Partner Age", min_value=18, value=30)
vehicle_condition = st.slider("Vehicle Condition (0–3)", 0, 3, 3)
multiple_deliveries = st.number_input("Multiple Deliveries", min_value=0, value=0)
order_time_hour = st.slider("Order Time (Hour)", 0, 23, 12)
order_day = st.slider("Order Day (1 = Mon, 7 = Sun)", 1, 7, 5)
pickup_time_minutes = st.number_input("Pickup Time (minutes)", min_value=0, value=10)

traffic = st.selectbox("Traffic", ["low", "medium", "high", "jam"])
city_type = st.selectbox("City Type", ["semi-urban", "urban", "metropolitian"])
festival = st.selectbox("Festival", ["no", "yes"])
weather = st.selectbox("Weather", ["sunny", "stormy", "windy", "sandstorms", "fog"])

if st.button("Predict Delivery Time"):
    input_df = pd.DataFrame([{
        "distance": distance,
        "ratings": ratings,
        "age": age,
        "vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "order_time_hour": order_time_hour,
        "order_day": order_day,
        "pickup_time_minutes": pickup_time_minutes,
        "traffic": traffic,
        "city_type": city_type,
        "festival": festival,
        "weather": weather
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")

st.markdown('</div>', unsafe_allow_html=True)
