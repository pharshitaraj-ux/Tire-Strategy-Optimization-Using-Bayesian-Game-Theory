import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_weather_merged.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_preprocessed.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("F1 2025 DATASET PREPROCESSING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nInput file:")
print(INPUT_FILE)

print("\nOriginal Shape:", df.shape)


# ============================================================
# VERIFY WEATHER MERGE
# ============================================================

required_weather_columns = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
    "WeatherMatched",
    "RaceDateValid"
]

print("\nChecking weather columns...")

missing_weather_columns = [
    col
    for col in required_weather_columns
    if col not in df.columns
]

if missing_weather_columns:
    raise ValueError(
        "Missing required weather columns: "
        + str(missing_weather_columns)
    )

print("Weather columns validated successfully.")


# ============================================================
# WEATHER MATCH VALIDATION
# ============================================================

weather_match_rate = (
    df["WeatherMatched"]
    .astype(bool)
    .mean()
    * 100
)

race_date_valid_rate = (
    df["RaceDateValid"]
    .astype(bool)
    .mean()
    * 100
)

print("\nWeather validation:")
print(
    f"Weather matched: "
    f"{weather_match_rate:.2f}%"
)

print(
    f"Race date valid: "
    f"{race_date_valid_rate:.2f}%"
)

if weather_match_rate < 100:
    print(
        "\nWARNING: Some rows do not have a valid "
        "weather match."
    )

if race_date_valid_rate < 100:
    print(
        "\nWARNING: Some rows failed race-date validation."
    )


# ============================================================
# CONVERT TIME COLUMNS TO SECONDS
# ============================================================

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
    "Sector3SessionTime",
    "WeatherTimeDifference"
]

print("\nConverting time columns to seconds...")

for col in time_columns:

    if col in df.columns:

        df[col] = pd.to_timedelta(
            df[col],
            errors="coerce"
        ).dt.total_seconds()


# ============================================================
# CONVERT DATE COLUMNS
# ============================================================

date_columns = [
    "LapStartDate",
    "EventDate",
    "WeatherEventDate"
]

print("Converting date columns...")

for col in date_columns:

    if col in df.columns:

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )


# ============================================================
# BOOLEAN COLUMNS → INTEGER
# ============================================================

bool_columns = [
    "FreshTyre",
    "Deleted",
    "FastF1Generated",
    "IsPersonalBest",
    "IsAccurate",
    "Rainfall",
    "WeatherMatched",
    "RaceDateValid"
]

print("\nConverting boolean columns...")

for col in bool_columns:

    if col in df.columns:

        # Handle missing values
        df[col] = df[col].fillna(False)

        # Convert True / False → 1 / 0
        df[col] = (
            df[col]
            .astype(bool)
            .astype(int)
        )


# ============================================================
# MISSING VALUES BEFORE IMPUTATION
# ============================================================

print("\nMissing values before preprocessing:")

missing_before = (
    df.isnull()
    .sum()
)

missing_before = (
    missing_before[
        missing_before > 0
    ]
)

if len(missing_before) == 0:

    print("No missing values.")

else:

    print(missing_before)


# ============================================================
# NUMERIC MISSING VALUES
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

print("\nHandling numeric missing values...")

for col in numeric_cols:

    if df[col].isnull().any():

        median_value = df[col].median()

        df[col] = df[col].fillna(
            median_value
        )


# ============================================================
# CATEGORICAL MISSING VALUES
# ============================================================

categorical_cols = df.select_dtypes(
    include="object"
).columns

print("Handling categorical missing values...")

for col in categorical_cols:

    if df[col].isnull().any():

        df[col] = df[col].fillna(
            "Unknown"
        )


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

drop_cols = [
    "DeletedReason"
]

print("\nRemoving unnecessary columns...")

for col in drop_cols:

    if col in df.columns:

        df.drop(
            columns=col,
            inplace=True
        )

        print(
            f"Removed: {col}"
        )


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    "\nFinal dataset shape:",
    df.shape
)

remaining_missing = (
    df.isnull()
    .sum()
    .sum()
)

print(
    "Total missing values:",
    remaining_missing
)

print(
    "Weather matched rows:",
    df["WeatherMatched"].sum(),
    "/",
    len(df)
)

print(
    "Race date valid rows:",
    df["RaceDateValid"].sum(),
    "/",
    len(df)
)


# ============================================================
# CHECK WEATHER VALUES
# ============================================================

weather_columns = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed"
]

print("\nWeather missing values after preprocessing:")

for col in weather_columns:

    if col in df.columns:

        print(
            f"{col:15s}: "
            f"{df[col].isnull().sum()}"
        )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nSaved to:"
)

print(OUTPUT_FILE)

print(
    "\nFinal Shape:",
    df.shape
)

print(
    "\nTotal Missing Values:",
    df.isnull().sum().sum()
)

print("\nDone.")