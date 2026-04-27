import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import load_and_train_model

# Load the model
model, _, _, _ = load_and_train_model()

# Load the data for insights
columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
df = pd.read_csv("sample_loan_data.csv", sep='\t', names=columns, header=0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

st.title("Loan Approval Prediction and Insights")

# Sidebar for prediction
st.sidebar.header("Loan Approval Prediction")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Probability of approval
    
    if prediction == 1:
        st.sidebar.success("Loan Approved")
    else:
        st.sidebar.error("Loan Rejected")
    
    # Risk classification
    if prob > 0.8:
        risk = "Low Risk"
    elif prob > 0.5:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    
    st.sidebar.write(f"Approval Probability: {prob:.2f}")
    st.sidebar.write(f"Risk Classification: {risk}")

# Main area for graphical insights
st.header("Graphical Insights")

# Distribution of Loan Status
st.subheader("Loan Status Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Loan_Status', data=df, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Rejected', 'Approved'])
st.pyplot(fig)

# Income distribution
st.subheader("Applicant Income Distribution")
fig, ax = plt.subplots()
sns.histplot(df['ApplicantIncome'], kde=True, ax=ax)
st.pyplot(fig)

# Loan Amount vs Income
st.subheader("Loan Amount vs Applicant Income")
fig, ax = plt.subplots()
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df, ax=ax)
st.pyplot(fig)

# Credit History impact
st.subheader("Credit History vs Loan Status")
fig, ax = plt.subplots()
sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax)
st.pyplot(fig)