import os
import pandas as pd
import numpy as np

# ---------------------------------------------
# Paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_final_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_preprocessed.csv"
)

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

# ---------------------------------------------
# Convert timedelta columns to seconds
# ---------------------------------------------
time_columns = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "LapStartTime",
    "WeatherTime",
    "PitInTime",
    "PitOutTime",
    "Sector1SessionTime",
    "Sector2SessionTime",
    "Sector3SessionTime"
]

for col in time_columns:
    if col in df.columns:
        df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()

# ---------------------------------------------
# Convert Date column
# ---------------------------------------------
if "LapStartDate" in df.columns:
    df["LapStartDate"] = pd.to_datetime(
        df["LapStartDate"],
        errors="coerce"
    )

print("\nMissing values before preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ---------------------------------------------
# Boolean → Integer
# ---------------------------------------------
bool_columns = [
    "FreshTyre",
    "Deleted",
    "FastF1Generated",
    "IsPersonalBest",
    "IsAccurate"
]

for col in bool_columns:
    if col in df.columns:
        # Replace missing values with False
        df[col] = df[col].fillna(False)

        # Convert True/False to 1/0
        df[col] = df[col].astype(bool).astype(int)

if "Rainfall" in df.columns:
    df["Rainfall"] = df["Rainfall"].astype(int)
# ---------------------------------------------
# Missing values
# ---------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# ---------------------------------------------
# Remove unnecessary columns
# ---------------------------------------------
drop_cols = [
    "DeletedReason"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ---------------------------------------------
# Save
# ---------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nPreprocessing Complete")
print("New Shape:", df.shape)

print("\nMissing Values Remaining")
print(df.isnull().sum().sum())

print("\nSaved to:")
print(OUTPUT_FILE)