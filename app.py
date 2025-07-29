import streamlit as st
import pandas as pd
import joblib

model = joblib.load('xgboost_model_pipeline.pkl')

st.title("Bank Marketing Subscription Prediction")
st.write("Enter customer details to predict if they will subscribe.")

age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 
                           'unemployed', 'entrepreneur', 'housemaid', 'self-employed', 'student', 'unknown'])
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
education = st.selectbox('Education', ['secondary', 'tertiary', 'primary', 'unknown'])
default = st.selectbox('Default Credit', ['yes', 'no'])
housing = st.selectbox('Housing Loan', ['yes', 'no'])
loan = st.selectbox('Personal Loan', ['yes', 'no'])
contact = st.selectbox('Contact Type', ['cellular', 'telephone'])
month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Call Duration (seconds)', min_value=0, max_value=5000, value=100)
campaign = st.number_input('Number of Contacts During Campaign', min_value=1, max_value=50, value=1)

input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign]
})

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: YES (Probability: {prob:.2f}) - Customer is likely to subscribe.")
    else:
        st.warning(f"Prediction: NO (Probability: {prob:.2f}) - Customer is unlikely to subscribe.")
