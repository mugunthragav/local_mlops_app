import sys
import os
import mlflow.pyfunc
import pandas as pd
import streamlit as st

# Add the path to your model files (adjust the path as per your directory structure)
sys.path.append(os.path.abspath("C:/Users/Admin/PycharmProjects/mlflow/local_mlops/models"))

# Load the latest production model from MLflow
def load_production_model():
    client = mlflow.tracking.MlflowClient()

    # Assuming the model name you registered
    model_name = "RandomForestModel"

    # Fetch the latest production version
    model_version = client.get_latest_versions(model_name, stages=["Production"])[0].version

    # Load the model using pyfunc from MLflow model registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model

# Initialize the model at startup
model = load_production_model()

# Streamlit app
st.title("House Price Prediction")

# Input fields for house features
area = st.number_input("Area (sq ft)", min_value=0, value=1500)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, value=2.0, step=0.5)
stories = st.number_input("Number of Stories", min_value=1, value=1)

# Categorical fields with dropdown options
mainroad = st.selectbox("Main Road Access", options=["Yes", "No"])
guestroom = st.selectbox("Guest Room", options=["Yes", "No"])
basement = st.selectbox("Basement", options=["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", options=["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", options=["Yes", "No"])
parking = st.number_input("Parking Spaces", min_value=0, value=1)
prefarea = st.selectbox("Preferred Area", options=["Yes", "No"])
furnishingstatus = st.selectbox("Furnishing Status", options=["Furnished", "Unfurnished", "Semi-Furnished"])

# Prepare input data for prediction
input_data = {
    "area": area,
    "bedrooms": num_bedrooms,
    "bathrooms": num_bathrooms,
    "stories": stories,
    "mainroad": 1 if mainroad == "Yes" else 0,
    "guestroom": 1 if guestroom == "Yes" else 0,
    "basement": 1 if basement == "Yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
    "airconditioning": 1 if airconditioning == "Yes" else 0,
    "parking": parking,
    "prefarea": 1 if prefarea == "Yes" else 0,
}

# Convert furnishing status to numerical if needed
if furnishingstatus == "Furnished":
    furnishing_encoded = 2
elif furnishingstatus == "Semi-Furnished":
    furnishing_encoded = 1
else:
    furnishing_encoded = 0

# Include the furnishing status in the input data
input_data["furnishingstatus"] = furnishing_encoded

# Create a DataFrame for prediction
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    # Make predictions
    predictions = model.predict(input_df)

    # Display predictions
    st.success(f"Predicted House Price: ${predictions[0]:,.2f}")



