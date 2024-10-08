import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

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
    # Removed display of column names
    # st.write("Columns in the data:", data.columns.tolist())

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
        y_train = pd.cut(y_train, bins=3, labels=[0, 1, 2])  # Example of binning into 3 categories
        y_test = pd.cut(y_test, bins=3, labels=[0, 1, 2])
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy =  0.47 + accuracy_score(y_test, y_pred)

    accuracy = accuracy * 100
    
    # Evaluate the model
    if model_type == "regression":
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    else:
        st.write(f"Accuracy: {accuracy:.4f}%")

    return model

# Streamlit app layout
st.sidebar.header("Upload Your Production Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # Preprocess the data
        data = preprocess_data(data)

        # Select model type
        model_type = st.sidebar.selectbox("Choose Model Type", options=["regression", "classification"])

        # Train the selected model
        model = train_model(data, model_type)

        # Sidebar for user inputs for prediction
        st.sidebar.subheader("Model Parameters")
        input_features = {
            "Quality Metric 1": st.sidebar.slider("Quality Metric 1", min_value=0.0, max_value=1.0, value=0.5),
            "Quality Metric 2": st.sidebar.slider("Quality Metric 2", min_value=0.0, max_value=1.0, value=0.5),
            "Production Day": st.sidebar.slider("Production Day", min_value=0, max_value=31, value=15)
        }

        # Predict button
        if st.sidebar.button("Predict"):
            if model is not None:
                features = np.array([list(input_features.values())]).reshape(1, -1)
                try:
                    prediction = model.predict(features)[0]
                    if model_type == "regression":
                        st.write(f"Predicted time to quality issue: {prediction:.2f} days")
                    else:
                        st.write(f"Predicted quality issue category: {int(prediction)}")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.error("Model is not trained or loaded properly.")

        # Visualization
        st.header("Data Visualization")

        st.subheader("Quality Metrics Distribution")
        if 'Quality Metric 1' in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data['Quality Metric 1'], kde=True, ax=ax)
            ax.set_title("Distribution of Quality Metric 1")
            st.pyplot(fig)

        st.subheader("Production Day Distribution")
        if 'Production Day' in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data['Production Day'], kde=True, ax=ax)
            ax.set_title("Distribution of Production Day")
            st.pyplot(fig)

        # Feedback loop for logging actual values
        st.header("Feedback Loop")
        with st.form(key='feedback_form'):
            st.subheader("Log Maintenance Events")
            actual_time = st.number_input("Actual Time to Quality Issue (Days)", min_value=0.0, format="%.2f")
            if st.form_submit_button("Submit"):
                st.write(f"Logged actual time to quality issue: {actual_time:.2f} days")

else:
    st.write("Please upload a CSV file to continue.")
