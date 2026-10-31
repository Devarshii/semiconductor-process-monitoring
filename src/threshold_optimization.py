import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


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
# 2. Prepare features and target
# --------------------------------------------------

X = df.drop(
    columns=["result", "timestamp"],
    errors="ignore"
)

y = df["result"].map({
    "PASS": 0,
    "FAIL": 1
})

print("\nClass distribution:")
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


# --------------------------------------------------
# 4. Standardize features
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. Train Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

print("\nTraining Logistic Regression...")

model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 6. Get FAIL probabilities
# --------------------------------------------------

fail_probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]


# --------------------------------------------------
# 7. Test different decision thresholds
# --------------------------------------------------

thresholds = np.arange(
    0.10,
    0.71,
    0.05
)

results = []

print("\n========== THRESHOLD COMPARISON ==========\n")

for threshold in thresholds:

    y_pred = (
        fail_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "fail_precision": precision,
        "fail_recall": recall,
        "fail_f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    })


results_df = pd.DataFrame(results)

display_df = results_df.copy()

display_df["threshold"] = (
    display_df["threshold"].round(2)
)

display_df["accuracy"] = (
    display_df["accuracy"] * 100
).round(2)

display_df["fail_precision"] = (
    display_df["fail_precision"] * 100
).round(2)

display_df["fail_recall"] = (
    display_df["fail_recall"] * 100
).round(2)

display_df["fail_f1"] = (
    display_df["fail_f1"].round(4)
)

print(display_df.to_string(index=False))


# --------------------------------------------------
# 8. Select best threshold by FAIL F1 score
# --------------------------------------------------

best_index = results_df[
    "fail_f1"
].idxmax()

best_row = results_df.loc[
    best_index
]

best_threshold = float(
    best_row["threshold"]
)


print(
    "\n========== BEST THRESHOLD BY FAIL F1 =========="
)

print(
    f"Threshold:      {best_threshold:.2f}"
)

print(
    f"Accuracy:       "
    f"{best_row['accuracy'] * 100:.2f}%"
)

print(
    f"FAIL Recall:    "
    f"{best_row['fail_recall'] * 100:.2f}%"
)

print(
    f"FAIL Precision: "
    f"{best_row['fail_precision'] * 100:.2f}%"
)

print(
    f"FAIL F1 Score:  "
    f"{best_row['fail_f1']:.4f}"
)


# --------------------------------------------------
# 9. Detailed evaluation at selected threshold
# --------------------------------------------------

best_predictions = (
    fail_probabilities >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_test,
    best_predictions
)

tn, fp, fn, tp = cm.ravel()


print(
    "\n========== CONFUSION MATRIX =========="
)

print(cm)


print(
    "\n========== CLASSIFICATION REPORT =========="
)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["PASS", "FAIL"],
        zero_division=0
    )
)


# --------------------------------------------------
# 10. Manufacturing interpretation
# --------------------------------------------------

print(
    "\n========== MANUFACTURING INTERPRETATION =========="
)

print(
    f"Actual FAIL runs:            {tp + fn}"
)

print(
    f"FAIL runs detected:          {tp}"
)

print(
    f"FAIL runs missed:            {fn}"
)

print(
    f"PASS runs incorrectly flagged: {fp}"
)

print(
    "\nLowering the classification threshold can "
    "increase FAIL detection, but it may also increase "
    "false alarms."
)

print(
    "\nFor semiconductor process monitoring, the final "
    "threshold should balance the cost of missing a "
    "defective run against the cost of investigating "
    "false alarms."
)

print(
    "\nThreshold optimization completed successfully."
)