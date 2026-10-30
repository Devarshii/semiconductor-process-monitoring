import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "secom.data"
LABELS_PATH = BASE_DIR / "data" / "raw" / "secom_labels.data"


# --------------------------------------------------
# 2. Load sensor data
# --------------------------------------------------

sensor_data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

labels = pd.read_csv(
    LABELS_PATH,
    sep=r"\s+",
    header=None
)


# --------------------------------------------------
# 3. Rename columns
# --------------------------------------------------

sensor_data.columns = [
    f"sensor_{i+1}" for i in range(sensor_data.shape[1])
]

labels.columns = ["result", "timestamp"]


# --------------------------------------------------
# 4. Convert labels
# --------------------------------------------------

labels["result"] = labels["result"].map({
    -1: "PASS",
     1: "FAIL"
})


# --------------------------------------------------
# 5. Combine datasets
# --------------------------------------------------

df = pd.concat(
    [sensor_data, labels],
    axis=1
)


# --------------------------------------------------
# 6. Dataset dimensions
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("\nNumber of rows:")
print(df.shape[0])

print("\nNumber of columns:")
print(df.shape[1])


# --------------------------------------------------
# 7. PASS / FAIL distribution
# --------------------------------------------------

print("\n========== PASS / FAIL DISTRIBUTION ==========")

print(df["result"].value_counts())

print("\nPercentage distribution:")

print(
    df["result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# 8. Missing values
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = sensor_data.isnull().sum()

print("\nTotal missing values:")
print(missing_values.sum())

print("\nTop 20 sensors with most missing values:")

print(
    missing_values
    .sort_values(ascending=False)
    .head(20)
)


# --------------------------------------------------
# 9. Missing-value percentages
# --------------------------------------------------

missing_percentage = (
    sensor_data.isnull().mean() * 100
)

print("\nTop 20 sensors by missing percentage:")

print(
    missing_percentage
    .sort_values(ascending=False)
    .head(20)
    .round(2)
)


# --------------------------------------------------
# 10. Duplicate rows
# --------------------------------------------------

print("\n========== DUPLICATE ROWS ==========")

duplicate_rows = df.duplicated().sum()

print("Number of duplicate rows:")
print(duplicate_rows)


# --------------------------------------------------
# 11. Constant sensors
# --------------------------------------------------

print("\n========== CONSTANT SENSORS ==========")

unique_values = sensor_data.nunique()

constant_sensors = unique_values[
    unique_values <= 1
]

print("Number of constant sensors:")
print(len(constant_sensors))

print("\nConstant sensor names:")

print(constant_sensors.index.tolist())


# --------------------------------------------------
# 12. Basic sensor statistics
# --------------------------------------------------

print("\n========== SENSOR STATISTICS ==========")

print(
    sensor_data.describe()
)