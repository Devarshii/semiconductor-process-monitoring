import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

# --------------------------------------------------
# 1. Locate project and load feature-engineered data
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

# Remove columns that should not be used as predictors
X = df.drop(columns=["result", "timestamp"], errors="ignore")

# Convert PASS / FAIL into 0 / 1
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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Create Random Forest
# --------------------------------------------------

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. Hyperparameter grid
# --------------------------------------------------

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [
        "balanced",
        {0: 1, 1: 3},
        {0: 1, 1: 5},
        {0: 1, 1: 10}
    ]
}


# --------------------------------------------------
# 6. Grid Search
# --------------------------------------------------

print("\nStarting Random Forest tuning...")
print("This may take a few minutes.\n")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# --------------------------------------------------
# 7. Best model
# --------------------------------------------------

best_model = grid_search.best_estimator_

print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

print("\nBest cross-validation FAIL recall:")
print(round(grid_search.best_score_, 4))


# --------------------------------------------------
# 8. Test predictions
# --------------------------------------------------

y_pred = best_model.predict(X_test)


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
fail_recall = recall_score(y_test, y_pred, pos_label=1)
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

print("\n========== TUNED RANDOM FOREST SUMMARY ==========")

print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"FAIL Recall:    {fail_recall * 100:.2f}%")
print(f"FAIL Precision: {fail_precision * 100:.2f}%")
print(f"FAIL F1 Score:  {fail_f1:.4f}")

print(
    "\nFor semiconductor monitoring, FAIL recall is especially "
    "important because undetected failures may represent "
    "defective production runs."
)

print("\nRandom Forest tuning completed successfully.")