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

    # ✅ Clean data
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = df.dropna(subset=["Loan_Status"])

    # 🔥 Fix Dependents
    df["Dependents"] = df["Dependents"].replace("3+", "3")

    # Drop ID
    df = df.drop(columns=["Loan_ID"], errors="ignore")

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    print("\nClass Distribution:")
    print(y.value_counts())

    # Features
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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", solver='liblinear'),
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
        print(classification_report(y_test, y_pred))

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": cm
            }

    print("\nBest Model Selected")
    print(f"F1 Score: {best_score:.4f}")

    return best_model, X_test, y_test, best_metrics