import streamlit as st
import pandas as pd
import joblib

model = joblib.load('final_model_pipeline.pkl')

st.title("Bank Marketing Term Deposit Prediction")
st.write("Enter customer details to predict if they will subscribe to a term deposit.")

age = st.number_input("Age", 18, 100, 30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'university.degree', 'professional.course', 'illiterate', 'unknown'])
default = st.selectbox("Has Credit in Default?", ['yes', 'no', 'unknown'])
housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
loan = st.selectbox("Has Personal Loan?", ['yes', 'no', 'unknown'])
contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox("Day of the Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
cons_price_idx = st.number_input("Consumer Price Index", 90.0, 95.0, 92.9)
cons_conf_idx = st.number_input("Consumer Confidence Index", -60.0, -20.0, -40.0)
euribor3m = st.number_input("Euribor 3 Month Rate", 0.0, 6.0, 4.8)

input_data = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'campaign': campaign,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m
}])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"The customer is likely to subscribe to a term deposit (Probability: {prob:.2f})")
    else:
        st.warning(f"The customer is unlikely to subscribe (Probability: {prob:.2f})")

