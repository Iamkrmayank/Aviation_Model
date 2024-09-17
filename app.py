import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return None

def train_model(data):
    # Display columns to check for correct names
    st.write("Columns in the data:", data.columns.tolist())
    
    required_columns = ['Quality Metric 1', 'Quality Metric 2', 'Production Day', 'Time to Quality Issue']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return None

    # Prepare your data
    X = data[['Quality Metric 1', 'Quality Metric 2', 'Production Day']]
    y = data['Time to Quality Issue']
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Sidebar for file upload
st.sidebar.header("Upload Your Production Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        model = train_model(data)

        # Sidebar for user inputs
        st.sidebar.subheader("Model Parameters")
        input_features = {
            "Quality Metric 1": st.sidebar.slider("Quality Metric 1", min_value=0.0, max_value=1.0, value=0.5),
            "Quality Metric 2": st.sidebar.slider("Quality Metric 2", min_value=0.0, max_value=1.0, value=0.5),
            "Production Day": st.sidebar.slider("Production Day", min_value=0, max_value=31, value=15)
        }

        # Predict Button
        if st.sidebar.button("Predict"):
            if model is not None:
                features = np.array([list(input_features.values())]).reshape(1, -1)
                try:
                    prediction = model.predict(features)[0]
                    st.write(f"Predicted time to quality issue: {prediction:.2f} days")
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

        st.subheader("Production Day")
        if 'Production Day' in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data['Production Day'], kde=True, ax=ax)
            ax.set_title("Distribution of Production Day")
            st.pyplot(fig)

        # Feedback Loop
        st.header("Feedback Loop")
        with st.form(key='feedback_form'):
            st.subheader("Log Maintenance Events")
            actual_time = st.number_input("Actual Time to Quality Issue (Days)", min_value=0.0, format="%.2f")
            if st.form_submit_button("Submit"):
                st.write(f"Logged actual time to quality issue: {actual_time:.2f} days")

        # Deployment and Monitoring
        st.header("Deployment and Monitoring")
        st.write("Integrate this system with existing maintenance workflows. Set up monitoring to track performance and accuracy over time.")

        # Continuous Improvement
        st.header("Continuous Improvement")
        st.write("Regularly retrain the model with new data. Refine features and algorithms based on real-world performance.")
else:
    st.write("Please upload a CSV file to continue.")
