"""
Final Evaluation
Semiconductor Process Monitoring Project

Final Random Forest evaluation for semiconductor
manufacturing failure detection.

Target:
    0 = PASS
    1 = FAIL
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "secom_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL SEMICONDUCTOR MODEL EVALUATION")
print("=" * 70)

print("\nLoading feature-engineered dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. CHECK TARGET
# ============================================================

if "result" not in df.columns:
    raise ValueError(
        "Target column 'result' was not found."
    )

print("\nOriginal result values:")
print(df["result"].value_counts(dropna=False))


# ============================================================
# 4. PREPARE TARGET
# ============================================================

raw_y = df["result"].copy()


def convert_target(value):
    """
    Convert different possible SECOM target formats to:

    0 = PASS
    1 = FAIL
    """

    if pd.isna(value):
        return np.nan

    # Handle numeric values first
    if isinstance(value, (int, float, np.integer, np.floating)):

        if value == -1:
            return 0

        if value == 0:
            return 0

        if value == 1:
            return 1

    # Handle strings
    value_string = str(value).strip().lower()

    pass_values = {
        "pass",
        "passed",
        "0",
        "0.0",
        "-1",
        "-1.0",
    }

    fail_values = {
        "fail",
        "failed",
        "failure",
        "1",
        "1.0",
    }

    if value_string in pass_values:
        return 0

    if value_string in fail_values:
        return 1

    return np.nan


y = raw_y.apply(convert_target)


print("\nTarget after conversion:")
print(y.value_counts(dropna=False))


# ============================================================
# 5. CHECK TARGET CONVERSION
# ============================================================

valid_target_mask = y.notna()

valid_count = int(valid_target_mask.sum())
invalid_count = int((~valid_target_mask).sum())

print(f"\nValid target rows:   {valid_count}")
print(f"Invalid target rows: {invalid_count}")

if valid_count == 0:
    raise ValueError(
        "\nTarget conversion failed.\n"
        "No valid PASS/FAIL values were found."
    )


# ============================================================
# 6. CREATE X AND y
# ============================================================

X = df.loc[
    valid_target_mask
].drop(
    columns=["result"]
).copy()

y = y.loc[
    valid_target_mask
].astype(int)


# ============================================================
# 7. REMOVE NON-NUMERIC FEATURES
# ============================================================

print(
    f"\nFeatures before numeric filtering: "
    f"{X.shape[1]}"
)

non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric_columns:

    print("\nRemoving non-numeric columns:")

    for column in non_numeric_columns:
        print(f"  - {column}")


# Keep only numeric columns
X = X.select_dtypes(
    include=[np.number]
).copy()

print(
    f"\nNumeric features retained: "
    f"{X.shape[1]}"
)


# ============================================================
# 8. REMOVE COMPLETELY EMPTY SENSOR COLUMNS
# ============================================================

empty_columns = X.columns[
    X.isna().all()
].tolist()

if empty_columns:

    print(
        f"\nRemoving {len(empty_columns)} "
        "completely empty sensor columns."
    )

    X = X.drop(
        columns=empty_columns
    )


print(
    f"Final number of sensor features: "
    f"{X.shape[1]}"
)


# ============================================================
# 9. SAFETY CHECKS
# ============================================================

if len(X) == 0:
    raise ValueError(
        "No samples remain after preprocessing."
    )

if X.shape[1] == 0:
    raise ValueError(
        "No numeric sensor features remain."
    )

if y.nunique() < 2:
    raise ValueError(
        "Target does not contain both PASS and FAIL classes."
    )

if len(X) != len(y):
    raise ValueError(
        "X and y contain different numbers of samples."
    )


# ============================================================
# 10. FINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"\nSamples available: {len(X)}")
print(f"Sensor features:   {X.shape[1]}")

pass_count = int((y == 0).sum())
fail_count = int((y == 1).sum())

pass_percentage = pass_count / len(y) * 100
fail_percentage = fail_count / len(y) * 100

print("\nClass distribution:")

print(
    f"PASS: {pass_count} "
    f"({pass_percentage:.2f}%)"
)

print(
    f"FAIL: {fail_count} "
    f"({fail_percentage:.2f}%)"
)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print(
    f"Training FAIL runs: "
    f"{int((y_train == 1).sum())}"
)

print(
    f"Testing FAIL runs:  "
    f"{int((y_test == 1).sum())}"
)


# ============================================================
# 12. FINAL RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight={
                    0: 1,
                    1: 5,
                },
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 70)

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)

print("Model training completed successfully.")


# ============================================================
# 14. GET FAILURE PROBABILITIES
# ============================================================

fail_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. FIND USEFUL THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

threshold_results = []

thresholds = np.arange(
    0.10,
    0.51,
    0.05,
)

for threshold in thresholds:

    predictions = (
        fail_probability >= threshold
    ).astype(int)

    threshold_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    threshold_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    threshold_precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    threshold_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    threshold_results.append(
        {
            "Threshold": round(
                float(threshold),
                2,
            ),
            "Accuracy": threshold_accuracy,
            "FAIL Recall": threshold_recall,
            "FAIL Precision": threshold_precision,
            "FAIL F1": threshold_f1,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

print("\nThreshold comparison:\n")

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 16. SELECT FINAL THRESHOLD
# ============================================================

# For semiconductor manufacturing, missing a true failure can
# be costly. We therefore prioritize FAIL recall while still
# considering precision.
#
# This threshold can be changed later after reviewing the
# threshold comparison.

FINAL_THRESHOLD = 0.15

print(
    f"\nSelected final threshold: "
    f"{FINAL_THRESHOLD:.2f}"
)


# ============================================================
# 17. FINAL PREDICTIONS
# ============================================================

y_pred = (
    fail_probability >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# 18. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

fail_recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

fail_precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

fail_f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

try:

    roc_auc = roc_auc_score(
        y_test,
        fail_probability,
    )

except ValueError:

    roc_auc = np.nan


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "PASS",
            "FAIL",
        ],
        zero_division=0,
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 21. FINAL MODEL SUMMARY
# ============================================================

print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy:       "
    f"{accuracy * 100:.2f}%"
)

print(
    f"FAIL Recall:    "
    f"{fail_recall * 100:.2f}%"
)

print(
    f"FAIL Precision: "
    f"{fail_precision * 100:.2f}%"
)

print(
    f"FAIL F1 Score:  "
    f"{fail_f1:.4f}"
)

if not np.isnan(roc_auc):

    print(
        f"ROC-AUC:        "
        f"{roc_auc:.4f}"
    )


# ============================================================
# 22. MANUFACTURING INTERPRETATION
# ============================================================

actual_failures = int(
    (y_test == 1).sum()
)

detected_failures = int(tp)

missed_failures = int(fn)

false_alarms = int(fp)

correct_passes = int(tn)


print("\n" + "=" * 70)
print("MANUFACTURING INTERPRETATION")
print("=" * 70)

print(
    f"\nActual FAIL runs:           "
    f"{actual_failures}"
)

print(
    f"FAIL runs detected:         "
    f"{detected_failures}"
)

print(
    f"FAIL runs missed:           "
    f"{missed_failures}"
)

print(
    f"PASS runs correctly passed: "
    f"{correct_passes}"
)

print(
    f"False alarms:               "
    f"{false_alarms}"
)


if actual_failures > 0:

    detection_rate = (
        detected_failures
        / actual_failures
        * 100
    )

    print(
        f"\nFailure detection rate: "
        f"{detection_rate:.2f}%"
    )


print(
    "\nIn semiconductor process monitoring, FAIL recall "
    "is particularly important because an undetected "
    "failure may allow a defective production run to "
    "continue."
)

print(
    "\nThe classification threshold creates a trade-off: "
    "lower thresholds generally detect more failures, "
    "but may also generate more false alarms."
)


# ============================================================
# 23. SAVE THRESHOLD RESULTS
# ============================================================

threshold_path = (
    OUTPUT_DIR
    / "threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False,
)

print(
    f"\nThreshold analysis saved to:\n"
    f"{threshold_path}"
)


# ============================================================
# 24. SAVE FINAL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "FAIL Recall",
            "FAIL Precision",
            "FAIL F1 Score",
            "ROC AUC",
            "Decision Threshold",
            "Actual FAIL Runs",
            "Detected FAIL Runs",
            "Missed FAIL Runs",
            "False Alarms",
        ],
        "Value": [
            accuracy,
            fail_recall,
            fail_precision,
            fail_f1,
            roc_auc,
            FINAL_THRESHOLD,
            actual_failures,
            detected_failures,
            missed_failures,
            false_alarms,
        ],
    }
)

metrics_path = (
    OUTPUT_DIR
    / "final_model_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

print(
    f"\nFinal metrics saved to:\n"
    f"{metrics_path}"
)


# ============================================================
# 25. CONFUSION MATRIX GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "PASS",
        "FAIL",
    ],
)

display.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title(
    "Final Semiconductor Failure Detection Model\n"
    f"Threshold = {FINAL_THRESHOLD:.2f}"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

plt.tight_layout()


confusion_matrix_path = (
    FIGURE_DIR
    / "final_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nConfusion matrix saved to:\n"
    f"{confusion_matrix_path}"
)


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

rf_classifier = model.named_steps[
    "classifier"
]

importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": (
            rf_classifier.feature_importances_
        ),
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 27. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    OUTPUT_DIR
    / "final_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)

print(
    f"\nFeature importance saved to:\n"
    f"{importance_path}"
)


# ============================================================
# 28. DISPLAY IMPORTANT SENSORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 15 IMPORTANT SENSORS")
print("=" * 70)

print()

print(
    importance_df
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 29. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (
    importance_df
    .head(15)
    .sort_values(
        by="Importance",
        ascending=True,
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Sensor"
)

plt.title(
    "Top 15 Sensors for Semiconductor Failure Detection"
)

plt.tight_layout()


importance_figure_path = (
    FIGURE_DIR
    / "final_feature_importance.png"
)

plt.savefig(
    importance_figure_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nFeature importance graph saved to:\n"
    f"{importance_figure_path}"
)


# ============================================================
# 30. THRESHOLD PERFORMANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["FAIL Recall"],
    marker="o",
    label="FAIL Recall",
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["FAIL Precision"],
    marker="o",
    label="FAIL Precision",
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["FAIL F1"],
    marker="o",
    label="FAIL F1",
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Failure Detection Performance by Decision Threshold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


threshold_figure_path = (
    FIGURE_DIR
    / "threshold_performance.png"
)

plt.savefig(
    threshold_figure_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"\nThreshold performance graph saved to:\n"
    f"{threshold_figure_path}"
)


# ============================================================
# 31. FINAL COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {metrics_path}"
)

print(
    f"2. {threshold_path}"
)

print(
    f"3. {importance_path}"
)

print(
    f"4. {confusion_matrix_path}"
)

print(
    f"5. {importance_figure_path}"
)

print(
    f"6. {threshold_figure_path}"
)

print(
    "\nSemiconductor process monitoring "
    "evaluation pipeline completed."
)


# ============================================================
# 32. SHOW GRAPHS
# ============================================================

plt.show()