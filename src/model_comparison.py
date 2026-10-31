import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL RESULTS
# ============================================================

# Results obtained from our completed model experiments.
#
# For semiconductor failure detection, FAIL recall is especially
# important because missing an actual failing production run can
# allow a defective process to continue.

results = {
    "Model": [
        "Baseline Model",
        "Random Forest",
        "Tuned Random Forest",
        "SMOTE Random Forest",
        "Threshold Optimized"
    ],

    "Accuracy": [
        74.52,
        92.04,
        91.08,
        76.43,
        87.00
    ],

    "FAIL Recall": [
        57.14,
        19.05,
        9.52,
        42.86,
        42.86
    ],

    "FAIL Precision": [
        14.00,
        33.33,
        18.18,
        12.68,
        24.00
    ],

    "FAIL F1": [
        23.00,
        24.24,
        12.50,
        19.57,
        31.00
    ]
}


df_results = pd.DataFrame(results)


# ============================================================
# 3. DISPLAY MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("SEMICONDUCTOR FAILURE DETECTION - MODEL COMPARISON")
print("=" * 75)

print(df_results.to_string(index=False))


# ============================================================
# 4. IDENTIFY BEST MODELS
# ============================================================

best_accuracy = df_results.loc[df_results["Accuracy"].idxmax()]
best_recall = df_results.loc[df_results["FAIL Recall"].idxmax()]
best_precision = df_results.loc[df_results["FAIL Precision"].idxmax()]
best_f1 = df_results.loc[df_results["FAIL F1"].idxmax()]


print("\n" + "=" * 75)
print("BEST MODEL BY METRIC")
print("=" * 75)

print(
    f"Highest Accuracy:       "
    f"{best_accuracy['Model']} ({best_accuracy['Accuracy']:.2f}%)"
)

print(
    f"Highest FAIL Recall:    "
    f"{best_recall['Model']} ({best_recall['FAIL Recall']:.2f}%)"
)

print(
    f"Highest FAIL Precision: "
    f"{best_precision['Model']} ({best_precision['FAIL Precision']:.2f}%)"
)

print(
    f"Highest FAIL F1 Score:  "
    f"{best_f1['Model']} ({best_f1['FAIL F1']:.2f}%)"
)


# ============================================================
# 5. MANUFACTURING INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("MANUFACTURING INTERPRETATION")
print("=" * 75)

print(
    "\nAccuracy alone should not determine the best model because "
    "the semiconductor dataset is highly imbalanced."
)

print(
    "\nA model can achieve high accuracy by predicting most production "
    "runs as PASS while still missing many actual FAIL runs."
)

print(
    "\nFor process monitoring, FAIL recall is particularly important "
    "because it measures how many actual failing production runs "
    "the model successfully detects."
)

print(
    "\nFAIL precision is also important because very low precision "
    "would generate excessive false alarms for manufacturing engineers."
)

print(
    "\nTherefore, model selection should consider the trade-off between "
    "FAIL recall, FAIL precision, F1 score, and overall accuracy."
)


# ============================================================
# 6. SAVE RESULTS AS CSV
# ============================================================

csv_path = OUTPUT_DIR / "model_comparison.csv"

df_results.to_csv(csv_path, index=False)

print(f"\nModel comparison table saved to:\n{csv_path}")


# ============================================================
# 7. VISUALIZE MODEL PERFORMANCE
# ============================================================

plot_df = df_results.set_index("Model")

ax = plot_df[
    ["Accuracy", "FAIL Recall", "FAIL Precision", "FAIL F1"]
].plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title(
    "Comparison of Semiconductor Failure Detection Models",
    fontsize=15
)

plt.xlabel("Model")
plt.ylabel("Performance (%)")

plt.xticks(rotation=25, ha="right")

plt.legend(
    title="Metric",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()


# ============================================================
# 8. SAVE VISUALIZATION
# ============================================================

figure_path = FIGURE_DIR / "model_comparison.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nModel comparison visualization saved to:\n{figure_path}")


# ============================================================
# 9. SHOW VISUALIZATION
# ============================================================

plt.show()


print("\nModel comparison completed successfully.")