import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "secom_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "secom_features.csv"
)


# --------------------------------------------------
# 2. Load cleaned dataset
# --------------------------------------------------

print("\nLoading cleaned semiconductor dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Separate sensor features and target
# --------------------------------------------------

sensor_columns = [
    col for col in df.columns
    if col.startswith("sensor_")
]

X = df[sensor_columns].copy()
y = df["result"].copy()

print("\nNumber of sensor features:", len(sensor_columns))

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. Convert target to numeric
# --------------------------------------------------

y_numeric = y.map({
    "PASS": 0,
    "FAIL": 1
})

print("\nTarget encoding:")
print("PASS = 0")
print("FAIL = 1")


# --------------------------------------------------
# 5. Remove constant features
# --------------------------------------------------

unique_counts = X.nunique()

constant_features = unique_counts[
    unique_counts <= 1
].index.tolist()

print("\nConstant features found:")
print(len(constant_features))

X = X.drop(
    columns=constant_features
)

print(
    "Features remaining after constant-feature removal:",
    X.shape[1]
)


# --------------------------------------------------
# 6. Remove duplicate feature columns
# --------------------------------------------------

X_transposed = X.T

duplicate_mask = X_transposed.duplicated()

duplicate_features = X_transposed[
    duplicate_mask
].index.tolist()

print("\nDuplicate features found:")
print(len(duplicate_features))

X = X.drop(
    columns=duplicate_features
)

print(
    "Features remaining after duplicate removal:",
    X.shape[1]
)


# --------------------------------------------------
# 7. Standardize sensor features
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

print("\nSensor features standardized successfully.")


# --------------------------------------------------
# 8. Select the most informative sensors
# --------------------------------------------------

NUMBER_OF_FEATURES = min(
    50,
    X_scaled.shape[1]
)

selector = SelectKBest(
    score_func=f_classif,
    k=NUMBER_OF_FEATURES
)

X_selected = selector.fit_transform(
    X_scaled,
    y_numeric
)

selected_mask = selector.get_support()

selected_features = X_scaled.columns[
    selected_mask
].tolist()


# --------------------------------------------------
# 9. Display selected sensors
# --------------------------------------------------

feature_scores = pd.DataFrame({
    "sensor": X_scaled.columns,
    "f_score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    "f_score",
    ascending=False
)

print("\nTop 20 sensors by ANOVA F-score:")
print(feature_scores.head(20).to_string(index=False))

print(
    "\nTotal selected features:",
    len(selected_features)
)


# --------------------------------------------------
# 10. Create final feature dataset
# --------------------------------------------------

features_df = pd.DataFrame(
    X_selected,
    columns=selected_features,
    index=df.index
)

features_df["result"] = y.values

if "timestamp" in df.columns:
    features_df["timestamp"] = df["timestamp"].values


# --------------------------------------------------
# 11. Save feature-engineered dataset
# --------------------------------------------------

features_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nFeature-engineered dataset shape:")
print(features_df.shape)

print("\nFeature-engineered dataset saved to:")
print(OUTPUT_PATH)

print("\nFeature engineering completed successfully.")