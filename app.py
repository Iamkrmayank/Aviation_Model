import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    datetime_cols = ['Production Date', 'Inspection Date', 'Delivery Date', 'Quality Issue Date']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Extract day, month, year, and hour from datetime columns if needed
    if 'Production Date' in df.columns:
        df['Production Day'] = df['Production Date'].dt.day
        df['Production Month'] = df['Production Date'].dt.month
        df['Production Year'] = df['Production Date'].dt.year
        df['Production Hour'] = df['Production Date'].dt.hour

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
def train_model(data):
    st.write("Columns in the data:", data.columns.tolist())

    required_columns = ['Production Line ID', 'Product Type', 'Quality Metric 1', 'Quality Metric 2', 
                        'Time to Inspection', 'Time from Delivery to Production', 'Production Day of Week',
                        'Production Hour', 'Production Month', 'Lagged Quality Metric 1', 
                        'Lagged Quality Metric 2', 'Rolling Avg Quality Metric 1', 
                        'Rolling Avg Quality Metric 2', 'Supplier ID', 'Material Type', 
                        'Material Quality Grade', 'Lead Time', 'Sensor ID', 'Sensor Type', 
                        'Sensor Reading', 'Quality Issue ID', 'Quality Issue Type', 
                        'Time to Quality Issue', 'Quality Issue Day of Week', 
                        'Quality Issue Hour', 'Quality Issue Month', 'Production Day', 'Production Year']
    missing_columns = [col for col in required_columns if col not in data.columns]


    X = data.drop(columns=['Quality Status'])
    y = data['Quality Status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose model
    model_type = st.session_state.get('model_type', 'regression')
    if model_type == "classification":
        from sklearn.ensemble import RandomForestClassifier
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
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")

    return model, scaler

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
    st.session_state['model_type'] = model_type

    # Train model
    if st.button("Train Model"):
        model, scaler = train_model(df)
        if model:
            st.success("Model trained successfully!")

    # Prediction
    if model_type == "regression":
        st.subheader("Predict Quality Metric")
        input_data = {col: st.number_input(col) for col in df.columns if col != 'Target'}
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            input_data_processed = preprocess_data(input_df)
            input_data_scaled = scaler.transform(input_data_processed)
            prediction = model.predict(input_data_scaled)
            st.write(f"Predicted Quality Metric: {prediction[0]:.2f}")
    elif model_type == "classification":
        st.subheader("Predict Quality Issue Type")
        input_data = {col: st.number_input(col) for col in df.columns if col != 'Target'}
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            input_data_processed = preprocess_data(input_df)
            input_data_scaled = scaler.transform(input_data_processed)
            prediction = model.predict(input_data_scaled)
            st.write(f"Predicted Quality Issue Type: {prediction[0]}")
else:
    st.write("Please upload a CSV file.")
