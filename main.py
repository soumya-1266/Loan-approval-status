import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_and_train_model(data_path="sample_loan_data.csv"):
    columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
               'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

    df = pd.read_csv(data_path, sep='\t', names=columns, header=0)

    # Convert target
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = df.dropna(subset=["Loan_Status"])

    # Drop Loan_ID
    df = df.drop(columns=["Loan_ID"], errors="ignore")

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # 🔍 Check class distribution
    print("\nClass Distribution:")
    print(y.value_counts())

    # Separate features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
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

    # ✅ Stratified split (IMPORTANT FIX)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Balanced models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "SVM": SVC(probability=True, random_state=42, class_weight="balanced")
    }

    best_model = None
    best_score = 0
    best_metrics = {}

    for name, clf in models.items():
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # ✅ Select best model using F1-score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": cm
            }

        # 🔍 Feature Importance (RandomForest only)
        if name == "RandomForest":
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importances = model.named_steps['classifier'].feature_importances_

            print("\nTop Important Features:")
            for f, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]:
                print(f"{f}: {imp:.4f}")

    print("\n Best Model Selected")
    print(f"F1 Score: {best_score:.4f}")

    return best_model, X_test, y_test, best_metrics


# 🚀 Run
if __name__ == "__main__":
    model, X_test, y_test, metrics = load_and_train_model()

    print("\nFinal Model Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])