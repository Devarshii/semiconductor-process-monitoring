import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "secom_cleaned.csv"
)

FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 3. Identify sensor columns
# --------------------------------------------------

sensor_columns = [
    column
    for column in df.columns
    if column.startswith("sensor_")
]

print("\nNumber of sensor columns:")
print(len(sensor_columns))


# --------------------------------------------------
# 4. Calculate PASS and FAIL sensor means
# --------------------------------------------------

pass_means = (
    df[df["result"] == "PASS"][sensor_columns]
    .mean()
)

fail_means = (
    df[df["result"] == "FAIL"][sensor_columns]
    .mean()
)

sensor_std = df[sensor_columns].std()


# --------------------------------------------------
# 5. Calculate standardized differences
# --------------------------------------------------

standardized_difference = (
    (fail_means - pass_means).abs()
    / sensor_std.replace(0, float("nan"))
)

standardized_difference = (
    standardized_difference
    .dropna()
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 6. Select top 10 sensors
# --------------------------------------------------

top_10_sensors = (
    standardized_difference
    .head(10)
    .index
    .tolist()
)

print("\nTop 10 sensors used for correlation analysis:")

for sensor in top_10_sensors:
    print(sensor)


# --------------------------------------------------
# 7. Calculate correlation matrix
# --------------------------------------------------

correlation_matrix = (
    df[top_10_sensors]
    .corr()
)

print("\n========== CORRELATION MATRIX ==========")

print(
    correlation_matrix
    .round(2)
)


# --------------------------------------------------
# 8. Create correlation heatmap
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 9)
)

heatmap = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)


# --------------------------------------------------
# 9. Configure axis labels
# --------------------------------------------------

ax.set_xticks(
    range(len(top_10_sensors))
)

ax.set_yticks(
    range(len(top_10_sensors))
)

ax.set_xticklabels(
    top_10_sensors,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    top_10_sensors
)


# --------------------------------------------------
# 10. Add correlation values to cells
# --------------------------------------------------

for i in range(len(top_10_sensors)):

    for j in range(len(top_10_sensors)):

        value = correlation_matrix.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )


# --------------------------------------------------
# 11. Add chart title and color bar
# --------------------------------------------------

ax.set_title(
    "Correlation Matrix of Top Failure-Associated Sensors",
    pad=20
)

colorbar = fig.colorbar(
    heatmap,
    ax=ax
)

colorbar.set_label(
    "Correlation Coefficient"
)

plt.tight_layout()


# --------------------------------------------------
# 12. Save correlation heatmap
# --------------------------------------------------

OUTPUT_PATH = (
    FIGURES_DIR
    / "top_sensor_correlation_heatmap.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("\nCorrelation heatmap saved to:")
print(OUTPUT_PATH)


# --------------------------------------------------
# 13. Display heatmap
# --------------------------------------------------

plt.show()