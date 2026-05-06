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
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #e6ecf0, #f7f9fb);
            padding: 40px;
        }

        .container {
            max-width: 750px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }

        h2 {
            text-align: center;
        }

        input, select {
            width: 100%;
            padding: 10px;
            margin: 8px 0 16px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }

        button {
            width: 100%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }

        .approved {
            background: #d4edda;
            color: #155724;
        }

        .rejected {
            background: #f8d7da;
            color: #721c24;
        }
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
        <select name="Dependents">
            <option>0</option><option>1</option><option>2</option><option>3+</option>
        </select>

        <label>Education</label>
        <select name="Education"><option>Graduate</option><option>Not Graduate</option></select>

        <label>Self Employed</label>
        <select name="Self_Employed"><option>No</option><option>Yes</option></select>

        <label>Applicant Income</label>
        <input type="number" name="ApplicantIncome" required>

        <label>Coapplicant Income</label>
        <input type="number" name="CoapplicantIncome" required>

        <label>Loan Amount (Enter like 120000)</label>
        <input type="number" name="LoanAmount" required>

        <label>Loan Term</label>
        <input type="number" name="Loan_Amount_Term" required>

        <label>Credit History</label>
        <select name="Credit_History"><option value="1">1</option><option value="0">0</option></select>

        <label>Property Area</label>
        <select name="Property_Area">
            <option>Urban</option><option>Rural</option><option>Semiurban</option>
        </select>

        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <div class="result {{ 'approved' if 'Approved' in result else 'rejected' }}">
            {{ result }}
        </div>
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
            'Dependents': '3' if request.form['Dependents'] == '3+' else request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']) / 1000,
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }

        df = pd.DataFrame([data])

        prob = model.predict_proba(df)[0][1]

        if prob >= 0.5:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        if prob >= 0.7:
            risk = "Low Risk"
        elif prob >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        result += f" | Probability: {prob:.2f} | Risk: {risk}"

    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)