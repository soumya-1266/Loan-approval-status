from flask import Flask, render_template_string, request
from main import load_and_train_model
import pandas as pd

app = Flask(__name__)
model, _, _, _ = load_and_train_model()

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Loan Approval Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f6f8; padding: 30px; }
        .container { max-width: 700px; margin: auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h2 { text-align: center; }
        input, select { width: 100%; padding: 10px; margin: 8px 0 16px; }
        button { background: #007bff; color: white; padding: 12px; border: none; width: 100%; border-radius: 8px; }
        .result { margin-top: 20px; font-size: 20px; font-weight: bold; text-align: center; }
    </style>
</head>
<body>
<div class="container">
    <h2>Loan Approval Prediction</h2>
    <form method="post">
        <label>Gender</label>
        <select name="Gender"><option>Male</option><option>Female</option></select>

        <label>Married</label>
        <select name="Married"><option>Yes</option><option>No</option></select>

        <label>Dependents</label>
        <select name="Dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select>

        <label>Education</label>
        <select name="Education"><option>Graduate</option><option>Not Graduate</option></select>

        <label>Self Employed</label>
        <select name="Self_Employed"><option>No</option><option>Yes</option></select>

        <label>Applicant Income</label>
        <input type="number" name="ApplicantIncome" required>

        <label>Coapplicant Income</label>
        <input type="number" name="CoapplicantIncome" required>

        <label>Loan Amount</label>
        <input type="number" name="LoanAmount" required>

        <label>Loan Amount Term</label>
        <input type="number" name="Loan_Amount_Term" required>

        <label>Credit History</label>
        <select name="Credit_History"><option value="1">1</option><option value="0">0</option></select>

        <label>Property Area</label>
        <select name="Property_Area"><option>Urban</option><option>Rural</option><option>Semiurban</option></select>

        <button type="submit">Predict</button>
    </form>
    {% if result %}
        <div class="result">{{ result }}</div>
    {% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }
        prediction = model.predict(pd.DataFrame([data]))[0]
        prob = model.predict_proba(pd.DataFrame([data]))[0][1]
        result = 'Loan Approved' if prediction == 1 else 'Loan Rejected'
        if prob > 0.8:
            risk = "Low Risk"
        elif prob > 0.5:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
        result += f"' 'Approval Probability: {prob:.2f} '  ' Risk Classification: {risk}"
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)