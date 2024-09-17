import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and cache data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return None

# Function to preprocess data
def preprocess_data(df):
    # Convert datetime columns if they exist
    datetime_cols = ['Production Datetime', 'Inspection Datetime', 'Delivery Datetime', 'Quality Issue Datetime']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Extract day, month, year, and hour from datetime columns if needed
    if 'Production Datetime' in df.columns:
        df['Production Day'] = df['Production Datetime'].dt.day
        df['Production Month'] = df['Production Datetime'].dt.month
        df['Production Year'] = df['Production Datetime'].dt.year
        df['Production Hour'] = df['Production Datetime'].dt.hour

    # Drop original datetime columns
    df.drop(columns=[col for col in datetime_cols if col in df.columns], inplace=True)

    # Label Encoding for categorical columns
    categorical_cols = ['Production Line ID', 'Product Type', 'Operator ID', 'Inspector ID', 'Supplier ID',
                        'Material Type', 'Material Quality Grade', 'Sensor ID', 'Sensor Type',
                        'Quality Issue ID', 'Quality Issue Type']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

# Train model based on whether it's regression or classification
def train_model(data, model_type="regression"):
    st.write("Columns in the data:", data.columns.tolist())

    required_columns = ['Quality Metric 1', 'Quality Metric 2', 'Production Day', 'Time to Quality Issue']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return None

    # Prepare data
    X = data[['Quality Metric 1', 'Quality Metric 2', 'Production Day']]
    y = data['Time to Quality Issue']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    if model_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")

    return model

# Main Streamlit App
st.title("Predictive Model Training and Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load data
df = load_data(uploaded_file)

if df is not None:
    st.write("Data preview:")
    st.write(df.head())

    # Preprocess data
    df = preprocess_data(df)

    # Model type selection
    model_type = st.selectbox("Select Model Type", ["regression", "classification"])

    # Train model
    if st.button("Train Model"):
        model = train_model(df, model_type=model_type)
        if model:
            st.success("Model trained successfully!")

    # Prediction
    if model_type == "regression":
        st.subheader("Predict Quality Metric")
        quality_metric_1 = st.number_input("Quality Metric 1")
        quality_metric_2 = st.number_input("Quality Metric 2")
        production_day = st.number_input("Production Day")

        if st.button("Predict"):
            input_data = pd.DataFrame({
                'Quality Metric 1': [quality_metric_1],
                'Quality Metric 2': [quality_metric_2],
                'Production Day': [production_day]
            })
            prediction = model.predict(input_data)
            st.write(f"Predicted Time to Quality Issue: {prediction[0]:.2f}")
    elif model_type == "classification":
        st.subheader("Predict Quality Issue Type")
        quality_metric_1 = st.number_input("Quality Metric 1")
        quality_metric_2 = st.number_input("Quality Metric 2")
        production_day = st.number_input("Production Day")

        if st.button("Predict"):
            input_data = pd.DataFrame({
                'Quality Metric 1': [quality_metric_1],
                'Quality Metric 2': [quality_metric_2],
                'Production Day': [production_day]
            })
            prediction = model.predict(input_data)
            st.write(f"Predicted Quality Issue Type: {prediction[0]}")
else:
    st.write("Please upload a CSV file.")
