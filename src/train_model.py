import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Define path
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "secom_features.csv"
)


# --------------------------------------------------
# 2. Load feature-engineered dataset
# --------------------------------------------------

print("\nLoading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------

feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[feature_columns].copy()

y = df["result"].map({
    "PASS": 0,
    "FAIL": 1
})

print("\nNumber of features:", X.shape[1])

print("\nTarget distribution:")
print(df["result"].value_counts())

print("\nTarget encoding:")
print("PASS = 0")
print("FAIL = 1")


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 5. Build Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# 8. Display model performance
# --------------------------------------------------

print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Meaning:")
print("[[True PASS,  False FAIL Alarm],")
print(" [Missed FAIL, Correctly Detected FAIL]]")


# --------------------------------------------------
# 10. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "PASS",
            "FAIL"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 11. Manufacturing interpretation
# --------------------------------------------------

print("========== MANUFACTURING MODEL SUMMARY ==========")

print(
    f"\nThe model correctly classified "
    f"{accuracy * 100:.2f}% of test production runs."
)

print(
    f"The model detected "
    f"{recall * 100:.2f}% of actual FAIL runs."
)

print(
    "\nFor semiconductor process monitoring, "
    "FAIL recall is especially important because "
    "missing a defective production run can be costly."
)

print("\nModel training and evaluation completed successfully.")