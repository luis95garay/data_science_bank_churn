from typing import Dict
import json
import streamlit as st
import requests


def predict_req(new_data: Dict):
    url = "http://ml_bankchurn:8000/predict"

    payload = json.dumps(new_data)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    details = json.loads(response.text)
    return details['data']


st.title("Bank Churn prediction :bank:")

col1, col2, col3 = st.columns(3)
with col1:
    customer_age = st.number_input('How old are you?', placeholder="Type a number", value=54, step=1)
    gender = st.selectbox(
    'What is your gender?',
    ('F', 'M'), index=1)
    dependent_count = st.number_input('How many people depends on you?', placeholder="Type a number", value=3, step=1)
    education_level = st.selectbox(
    'What is your education level?',
    ('Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'), index=4)
    marital_status = st.selectbox(
    'What is your marital status?',
    ('Married', 'Single', 'Unknown', 'Divorced'), index=0)
    income_category = st.selectbox(
    'What is your level of income?',
    ('Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'), index=3)

with col2:
    card_category = st.selectbox(
    'What is your card category?',
    ('Blue', 'Gold', 'Silver', 'Platinum'), index=2)
    months_on_book = st.number_input('Months_on_book?', placeholder="Type a number", value=39, step=1)
    total_relationship_count = st.number_input('Total_Relationship_Count?', placeholder="Type a number", value=5, step=1)
    months_inactive_12_mon = st.number_input('Months_Inactive_12_mon?', placeholder="Type a number", value=1, step=1)
    contacts_count_12_mon = st.number_input('Contacts_Count_12_mon?', placeholder="Type a number", value=3, step=1)
    credit_limit = st.number_input('Credit_Limit?', placeholder="Type a number", value=12691, step=1)
    
with col3:
    total_revolving_bal = st.number_input('Total_Revolving_Bal?', placeholder="Type a number", value=777, step=1)
    avg_open_to_buy = st.number_input('Avg_Open_To_Buy?', placeholder="Type a number", value=11914, step=1)
    total_amt_chng_q4_q1 = st.number_input('Total_Amt_Chng_Q4_Q1?', placeholder="Type a number", value=1.335, step=0.001)
    total_trans_amt = st.number_input('Total_Trans_Amt?', placeholder="Type a number", value=1144, step=1)
    total_trans_ct = st.number_input('Total_Trans_Ct?', placeholder="Type a number", value=42, step=1)
    total_ct_chng_q4_q1 = st.number_input('Total_Ct_Chng_Q4_Q1?', placeholder="Type a number", value=1.625, step=0.001)
    avg_utilization_ratio = st.number_input('Avg_Utilization_Ratio?', placeholder="Type a number", value=0.061, step=0.001)

submitted = st.button("Submit")
if submitted:
    new_data = {
        "customer_age": customer_age,
        "gender": gender,
        "dependent_count": dependent_count,
        "education_level": education_level,
        "marital_status": marital_status,
        "income_category": income_category,
        "card_category": card_category,
        "months_on_book": months_on_book,
        "total_relationship_count": total_relationship_count,
        "months_inactive_12_mon": months_inactive_12_mon,
        "contacts_count_12_mon": contacts_count_12_mon,
        "credit_limit": credit_limit,
        "total_revolving_Bal": total_revolving_bal,
        "avg_open_to_buy": avg_open_to_buy,
        "total_amt_chng_q4_q1": total_amt_chng_q4_q1,
        "total_trans_amt": total_trans_amt,
        "total_trans_ct": total_trans_ct,
        "total_ct_chng_q4_q1": total_ct_chng_q4_q1,
        "avg_utilization_ratio": avg_utilization_ratio
    }
    result = predict_req(new_data)
    st.write((f"The prediction is: {result}"))
