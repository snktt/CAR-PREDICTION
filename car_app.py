from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Car_data.csv"
MODEL_PATH = ROOT / "car_prediction_model.pkl"
FEATURES = ["Brand", "Model", "fuel", "seller_type", "transmission", "year", "km_driven", "Owner"]
OWNER_LABELS = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3,
    "Test Drive Car": 0,
}

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")
st.title("Used Car Price Prediction")
st.caption("Estimate a used car’s selling price using listings from Car_data.csv.")


@st.cache_data
def load_listing_data():
    data = pd.read_csv(DATA_PATH)
    data["Brand"] = data["name"].str.split().str[0].str.strip()
    data["Model"] = data["name"].str.split(n=1).str[1].fillna("Standard")
    data["Owner"] = data["owner"].map(OWNER_LABELS)
    return data.dropna(subset=FEATURES)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error("Car_data.csv was not found in the project folder.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("The model has not been trained from Car_data.csv yet.")
    st.code("python train_model.py", language="bash")
    st.stop()

data = load_listing_data()
model = load_model()
brand_options = sorted(data["Brand"].unique())
owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]

with st.sidebar:
    st.header("Car details")
    brand = st.selectbox("Car brand", brand_options)
    model_options = sorted(data.loc[data["Brand"] == brand, "Model"].unique())
    car_model = st.selectbox("Model name", model_options)
    year = st.selectbox("Manufacturing year", sorted(data["year"].unique(), reverse=True))
    km_driven = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000, value=50_000, step=1_000)
    fuel = st.selectbox("Fuel type", sorted(data["fuel"].dropna().unique()))
    seller_type = st.selectbox("Seller type", sorted(data["seller_type"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(data["transmission"].dropna().unique()))
    owner_label = st.selectbox("Previous ownership", owner_options)
    predict = st.button("Estimate selling price", type="primary", use_container_width=True)

st.subheader("Price estimate")
if predict:
    entry = pd.DataFrame([{
        "Brand": brand,
        "Model": car_model,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "year": int(year),
        "km_driven": int(km_driven),
        "Owner": OWNER_LABELS[owner_label],
    }], columns=FEATURES)
    estimate = max(0.0, float(model.predict(entry)[0])) * 0.70
    left, right = st.columns(2)
    left.metric("Estimated selling price", f"₹{estimate:.2f} lakh")
    right.metric("Estimated range", f"₹{estimate * 0.9:.2f}–{estimate * 1.1:.2f} lakh")
    st.markdown("#### Your car")
    st.write(f"**{brand} {car_model}** · {int(year)} · {km_driven:,} km · {fuel} · {transmission} · {owner_label}")
else:
    st.info("Choose the car details in the sidebar and select **Estimate selling price**.")

st.caption(f"Model trained on {len(data):,} listings from Car_data.csv. Prices are shown in lakh rupees.")
