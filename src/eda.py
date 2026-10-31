import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "secom_cleaned.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


# --------------------------------------------------
# 2. Create figures directory if needed
# --------------------------------------------------

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 3. Load cleaned semiconductor data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 4. Calculate PASS / FAIL counts
# --------------------------------------------------

result_counts = df["result"].value_counts()

print("\nPASS / FAIL counts:")
print(result_counts)


# --------------------------------------------------
# 5. Calculate percentages
# --------------------------------------------------

result_percentages = (
    df["result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPASS / FAIL percentages:")
print(result_percentages)


# --------------------------------------------------
# 6. Calculate manufacturing yield
# --------------------------------------------------

total_runs = len(df)

pass_runs = (df["result"] == "PASS").sum()
fail_runs = (df["result"] == "FAIL").sum()

yield_rate = (pass_runs / total_runs) * 100
failure_rate = (fail_runs / total_runs) * 100


print("\n========== MANUFACTURING SUMMARY ==========")

print(f"Total production runs: {total_runs}")
print(f"Passed runs: {pass_runs}")
print(f"Failed runs: {fail_runs}")
print(f"Yield rate: {yield_rate:.2f}%")
print(f"Failure rate: {failure_rate:.2f}%")


# --------------------------------------------------
# 7. Create PASS / FAIL bar chart
# --------------------------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    result_counts.index,
    result_counts.values
)

plt.title(
    "Semiconductor Manufacturing PASS vs FAIL Distribution"
)

plt.xlabel("Production Result")
plt.ylabel("Number of Production Runs")


# Add count labels above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


# --------------------------------------------------
# 8. Save visualization
# --------------------------------------------------

OUTPUT_PATH = FIGURES_DIR / "pass_fail_distribution.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)


print("\nVisualization saved to:")
print(OUTPUT_PATH)


# --------------------------------------------------
# 9. Display visualization
# --------------------------------------------------

plt.show()

# --------------------------------------------------
# 10. Identify sensor columns
# --------------------------------------------------

sensor_columns = [
    column for column in df.columns
    if column.startswith("sensor_")
]


# --------------------------------------------------
# 11. Calculate PASS and FAIL sensor averages
# --------------------------------------------------

pass_means = (
    df[df["result"] == "PASS"][sensor_columns]
    .mean()
)

fail_means = (
    df[df["result"] == "FAIL"][sensor_columns]
    .mean()
)


# --------------------------------------------------
# 12. Calculate standardized mean differences
# --------------------------------------------------

sensor_std = df[sensor_columns].std()

mean_difference = (
    (fail_means - pass_means).abs()
    / sensor_std.replace(0, float("nan"))
)

mean_difference = mean_difference.dropna()


# --------------------------------------------------
# 13. Select top 10 sensors
# --------------------------------------------------

top_sensors = (
    mean_difference
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 SENSORS ASSOCIATED WITH FAILURE ==========")

print(top_sensors)


# --------------------------------------------------
# 14. Create visualization
# --------------------------------------------------

plt.figure(figsize=(10, 6))

bars = plt.bar(
    top_sensors.index,
    top_sensors.values
)

plt.title(
    "Top 10 Sensors Differentiating PASS and FAIL Runs"
)

plt.xlabel("Sensor")
plt.ylabel("Standardized Mean Difference")

plt.xticks(
    rotation=45,
    ha="right"
)


# Add values above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


# --------------------------------------------------
# 15. Save visualization
# --------------------------------------------------

TOP_SENSOR_PATH = (
    FIGURES_DIR /
    "top_failure_sensors.png"
)

plt.savefig(
    TOP_SENSOR_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nTop sensor visualization saved to:")
print(TOP_SENSOR_PATH)


# --------------------------------------------------
# 16. Display visualization
# --------------------------------------------------

plt.show()

# --------------------------------------------------
# 17. Analyze strongest failure-associated sensor
# --------------------------------------------------

top_sensor = top_sensors.index[0]

print("\n========== STRONGEST SENSOR ANALYSIS ==========")
print(f"Selected sensor: {top_sensor}")

pass_sensor_mean = df.loc[
    df["result"] == "PASS",
    top_sensor
].mean()

fail_sensor_mean = df.loc[
    df["result"] == "FAIL",
    top_sensor
].mean()

print(f"PASS mean: {pass_sensor_mean:.4f}")
print(f"FAIL mean: {fail_sensor_mean:.4f}")


# --------------------------------------------------
# 18. Create PASS vs FAIL boxplot
# --------------------------------------------------

pass_values = df.loc[
    df["result"] == "PASS",
    top_sensor
].dropna()

fail_values = df.loc[
    df["result"] == "FAIL",
    top_sensor
].dropna()


plt.figure(figsize=(8, 6))

plt.boxplot(
    [pass_values, fail_values],
    tick_labels=["PASS", "FAIL"]
)

plt.title(
    f"{top_sensor}: PASS vs FAIL Distribution"
)

plt.xlabel("Production Result")
plt.ylabel("Sensor Reading")

plt.tight_layout()


# --------------------------------------------------
# 19. Save boxplot
# --------------------------------------------------

SENSOR_BOXPLOT_PATH = (
    FIGURES_DIR /
    "top_sensor_pass_fail_boxplot.png"
)

plt.savefig(
    SENSOR_BOXPLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nSensor boxplot saved to:")
print(SENSOR_BOXPLOT_PATH)


# --------------------------------------------------
# 20. Display boxplot
# --------------------------------------------------

plt.show()