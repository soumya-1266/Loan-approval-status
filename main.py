import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_and_train_model(data_path="sample_loan_data.csv"):
    columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
    df = pd.read_csv(data_path, sep='\t', names=columns, header=0)
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    
    # Drop rows with missing target
    df = df.dropna(subset=["Loan_Status"])

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_metrics = {}

    for name, clf in models.items():
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_metrics = {
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report
            }

    print(f"\nBest Model: {list(models.keys())[list(models.values()).index(best_model.named_steps['classifier'])]} with Accuracy: {best_accuracy:.4f}")

    return best_model, X_test, y_test, best_metrics


if __name__ == "__main__":
    model, X_test, y_test, metrics = load_and_train_model()
    print("\nBest Model Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))