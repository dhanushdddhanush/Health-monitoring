import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV data into a Pandas DataFrame
@st.cache
def load_data():
    heart_data = pd.read_csv('heart_disease_data.csv')
    return heart_data

heart_data = load_data()

# Sidebar
st.sidebar.subheader("Input Data")
age = st.sidebar.slider("Age", 0, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
trestbps = st.sidebar.slider("Resting Blood Pressure", 0, 200, 120)
chol = st.sidebar.slider("Cholesterol", 0, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.sidebar.slider("Resting Electrocardiographic Results", 0, 2, 0)
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 250, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 2.0)
slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment", 0, 2, 1)
ca = st.sidebar.slider("Number of Major Vessels (0-3) Colored by Flourosopy", 0, 3, 1)
thal = st.sidebar.slider("Thalassemia", 0, 3, 2)

# Preprocess sidebar inputs
sex = 0 if sex == "Male" else 1
fbs = 1 if fbs == "True" else 0
exang = 1 if exang == "Yes" else 0

# Prepare input data
input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Function to preprocess sidebar input and make prediction
def predict(input_data):
    model = LogisticRegression()
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model.fit(X_train, Y_train)
    prediction = model.predict([input_data])
    return prediction

# Streamlit app
def main():
    st.title("Heart Disease Prediction App")

    # Prediction
    prediction = predict(input_data)
    if prediction[0] == 0:
        st.write("The person does not have a heart disease.")
    else:
        st.write("The person has a heart disease.")

if __name__ == "__main__":
    main()
