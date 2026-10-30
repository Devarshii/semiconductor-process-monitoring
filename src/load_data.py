import pandas as pd
from pathlib import Path


# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define paths to the raw data files
DATA_PATH = BASE_DIR / "data" / "raw" / "secom.data"
LABELS_PATH = BASE_DIR / "data" / "raw" / "secom_labels.data"


# Load the semiconductor sensor data
sensor_data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None
)

# Load the labels and timestamps
labels = pd.read_csv(
    LABELS_PATH,
    sep=r"\s+",
    header=None
)


# Rename sensor columns
sensor_data.columns = [
    f"sensor_{i+1}" for i in range(sensor_data.shape[1])
]

# Rename label columns
labels.columns = ["result", "timestamp"]


# Convert numeric labels into readable categories
labels["result"] = labels["result"].map({
    -1: "PASS",
     1: "FAIL"
})


# Combine sensor measurements with labels
df = pd.concat([sensor_data, labels], axis=1)


# Display basic information
print("Sensor data shape:")
print(sensor_data.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nFirst 5 rows of sensor data:")
print(sensor_data.head())

print("\nFirst 5 rows of labels:")
print(labels.head())

print("\nCombined dataset shape:")
print(df.shape)

print("\nResult distribution:")
print(df["result"].value_counts())

print("\nFirst 5 rows of combined dataset:")
print(df.head())