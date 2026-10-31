import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# 1. Load feature-engineered dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "secom_features.csv"
)

print("Loading feature-engineered dataset...")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Prepare X and y
# --------------------------------------------------

X = df.drop(columns=["result", "timestamp"], errors="ignore")

y = df["result"].map({
    "PASS": 0,
    "FAIL": 1
})

print("\nOriginal class distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining class distribution BEFORE SMOTE:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 4. Standardize sensor features
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. Apply SMOTE ONLY to training data
# --------------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nTraining class distribution AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())

print(
    "\nSMOTE was applied only to the training set."
    "\nThe test set remains untouched."
)


# --------------------------------------------------
# 6. Train Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

print("\nTraining Logistic Regression on SMOTE data...")

model.fit(
    X_train_smote,
    y_train_smote
)


# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

fail_recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

fail_precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

fail_f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["PASS", "FAIL"],
        zero_division=0
    )
)


print("\n========== SMOTE MODEL SUMMARY ==========")

print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"FAIL Recall:    {fail_recall * 100:.2f}%")
print(f"FAIL Precision: {fail_precision * 100:.2f}%")
print(f"FAIL F1 Score:  {fail_f1:.4f}")


# --------------------------------------------------
# 9. Manufacturing interpretation
# --------------------------------------------------

actual_failures = int((y_test == 1).sum())

detected_failures = int(
    ((y_test == 1) & (y_pred == 1)).sum()
)

missed_failures = actual_failures - detected_failures

print("\n========== MANUFACTURING INTERPRETATION ==========")

print(f"Actual FAIL runs in test set: {actual_failures}")

print(
    f"FAIL runs correctly detected: {detected_failures}"
)

print(
    f"FAIL runs missed by model: {missed_failures}"
)

print(
    "\nIn semiconductor process monitoring, reducing missed "
    "FAIL runs is important because an undetected failure "
    "may allow a defective production run to continue."
)

print("\nSMOTE model training completed successfully.")