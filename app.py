import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import joblib

@st.cache
def load_data():
    return pd.read_csv('Time_Series_5k_data.csv')

def train_model(data):
    # Prepare your data
    X = data[['Quality Metric 1', 'Quality Metric 2', 'Products per day']]
    y = data['Time to Quality Issue']
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Load or train the model
data = load_data()
model = train_model(data)  # or use joblib.load if you have a pre-trained model

# Sidebar for user inputs
st.sidebar.header("Maintenance Prediction Dashboard")

# Inputs for prediction
st.sidebar.subheader("Model Parameters")
input_features = {
    "Quality Metric 1": st.sidebar.slider("Quality Metric 1", min_value=0.0, max_value=1.0, value=0.5),
    "Quality Metric 2": st.sidebar.slider("Quality Metric 2", min_value=0.0, max_value=1.0, value=0.5),
    "Products per day": st.sidebar.slider("Products per day", min_value=0, max_value=1000, value=500)
}

# Predict Button
if st.sidebar.button("Predict"):
    features = np.array([list(input_features.values())]).reshape(1, -1)
    try:
        prediction = model.predict(features)[0]
        st.write(f"Predicted time to quality issue: {prediction:.2f} days")
    except NotFittedError:
        st.write("Model is not trained or fitted.")

# Visualization
st.header("Data Visualization")

# Plot some example visualizations
st.subheader("Quality Metrics Distribution")
if 'Quality Metric 1' in data.columns:
    fig, ax = plt.subplots()
    sns.histplot(data['Quality Metric 1'], kde=True, ax=ax)
    ax.set_title("Distribution of Quality Metric 1")
    st.pyplot(fig)
else:
    st.write("Column 'Quality Metric 1' not found in the data.")

st.subheader("Products Per Day")
if 'Production Day' in data.columns:
    fig, ax = plt.subplots()
    sns.histplot(data['Production Day'], kde=True, ax=ax)
    ax.set_title("Distribution of Production Day")
    st.pyplot(fig)
else:
    st.write("Column 'Production Day' not found in the data.")

# Feedback Loop
st.header("Feedback Loop")

# Placeholder for logging actual maintenance events
with st.form(key='feedback_form'):
    st.subheader("Log Maintenance Events")
    actual_time = st.number_input("Actual Time to Quality Issue (Days)", min_value=0.0, format="%.2f")
    if st.form_submit_button("Submit"):
        st.write(f"Logged actual time to quality issue: {actual_time:.2f} days")
        # Here, you would save this feedback to your database or file

# Deployment and Monitoring
st.header("Deployment and Monitoring")

st.write("Integrate this system with existing maintenance workflows. Set up monitoring to track performance and accuracy over time.")

# Continuous Improvement
st.header("Continuous Improvement")

st.write("Regularly retrain the model with new data. Refine features and algorithms based on real-world performance.")
