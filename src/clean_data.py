import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "secom.data"
LABELS_PATH = BASE_DIR / "data" / "raw" / "secom_labels.data"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "secom_cleaned.csv"


# --------------------------------------------------
# 2. Load raw data
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

print("Original sensor data shape:")
print(sensor_data.shape)


# --------------------------------------------------
# 3. Rename columns
# --------------------------------------------------

sensor_data.columns = [
    f"sensor_{i+1}" for i in range(sensor_data.shape[1])
]

labels.columns = ["result", "timestamp"]


# --------------------------------------------------
# 4. Convert labels to readable values
# --------------------------------------------------

labels["result"] = labels["result"].map({
    -1: "PASS",
     1: "FAIL"
})


# --------------------------------------------------
# 5. Convert timestamp to datetime
# --------------------------------------------------

labels["timestamp"] = pd.to_datetime(
    labels["timestamp"],
    format="%d/%m/%Y %H:%M:%S"
)


# --------------------------------------------------
# 6. Remove sensors with more than 50% missing data
# --------------------------------------------------

missing_percentage = sensor_data.isnull().mean() * 100

high_missing_sensors = missing_percentage[
    missing_percentage > 50
].index.tolist()

print("\nSensors with more than 50% missing data:")
print(len(high_missing_sensors))

sensor_data = sensor_data.drop(
    columns=high_missing_sensors
)

print("\nShape after removing high-missing sensors:")
print(sensor_data.shape)


# --------------------------------------------------
# 7. Remove constant sensors
# --------------------------------------------------

constant_sensors = [
    column
    for column in sensor_data.columns
    if sensor_data[column].nunique(dropna=True) <= 1
]

print("\nConstant sensors found:")
print(len(constant_sensors))

sensor_data = sensor_data.drop(
    columns=constant_sensors
)

print("\nShape after removing constant sensors:")
print(sensor_data.shape)


# --------------------------------------------------
# 8. Fill remaining missing values with median
# --------------------------------------------------

missing_before = sensor_data.isnull().sum().sum()

print("\nRemaining missing values before imputation:")
print(missing_before)

sensor_data = sensor_data.fillna(
    sensor_data.median()
)

missing_after = sensor_data.isnull().sum().sum()

print("\nRemaining missing values after imputation:")
print(missing_after)


# --------------------------------------------------
# 9. Combine sensor data with labels
# --------------------------------------------------

df = pd.concat(
    [
        sensor_data.reset_index(drop=True),
        labels.reset_index(drop=True)
    ],
    axis=1
)


# --------------------------------------------------
# 10. Remove duplicate rows
# --------------------------------------------------

duplicates_before = df.duplicated().sum()

print("\nDuplicate rows found:")
print(duplicates_before)

df = df.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------
# 11. Final quality checks
# --------------------------------------------------

print("\n========== CLEANED DATA SUMMARY ==========")

print("\nFinal dataset shape:")
print(df.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nPASS / FAIL distribution:")
print(df["result"].value_counts())

print("\nPASS / FAIL percentage:")
print(
    df["result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# 12. Save cleaned dataset
# --------------------------------------------------

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nCleaned dataset successfully saved to:")
print(OUTPUT_PATH)