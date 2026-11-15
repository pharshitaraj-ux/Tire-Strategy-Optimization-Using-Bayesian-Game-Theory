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

DATA = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_preprocessed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "f1_2025_feature_engineered.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("F1 2025 FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(DATA)

print("\nInput file:")
print(DATA)

print("\nInput dataset shape:")
print(df.shape)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Race",
    "Driver",
    "Stint",
    "LapTime",
    "TyreLife",
    "Compound",
    "Rainfall",
    "TrackTemp",
    "PitInTime"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )

print("\nRequired columns validated successfully.")


# ============================================================
# 1. WEATHER STATE
# ============================================================

print("\n1. Creating WeatherState...")

df["WeatherState"] = np.where(
    df["Rainfall"] == 1,
    "Wet",
    "Dry"
)

print(
    df["WeatherState"]
    .value_counts()
)


# ============================================================
# 2. TRACK TEMPERATURE LEVEL
# ============================================================

print("\n2. Creating TemperatureLevel...")

df["TemperatureLevel"] = pd.cut(
    df["TrackTemp"],
    bins=[0, 30, 40, 100],
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    include_lowest=True
)


# ============================================================
# 3. TYRE WEAR INDEX
# ============================================================

print("\n3. Creating TyreWear...")

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"],
    errors="coerce"
)

# Maximum observed tyre life within each driver/stint
stint_max_life = (
    df.groupby(
        [
            "Race",
            "Driver",
            "Stint"
        ]
    )["TyreLife"]
    .transform("max")
)

# Tyre wear index:
#
# TyreWear =
# ((Current TyreLife - 1) /
#  (Maximum Stint TyreLife - 1)) * 100
#
# This represents the relative position within
# the observed stint, not physical rubber wear.

df["TyreWear"] = np.where(
    stint_max_life > 1,
    (
        (df["TyreLife"] - 1)
        /
        (stint_max_life - 1)
    ) * 100,
    0
)

df["TyreWear"] = df[
    "TyreWear"
].clip(
    lower=0,
    upper=100
)

print(
    "TyreWear range:",
    round(df["TyreWear"].min(), 2),
    "-",
    round(df["TyreWear"].max(), 2)
)


# ============================================================
# 4. PIT STOP INDICATOR
# ============================================================

print("\n4. Creating PitStop...")

df["PitStop"] = np.where(
    df["PitInTime"].notna(),
    1,
    0
)


# ============================================================
# 5. STINT LENGTH
# ============================================================

print("\n5. Creating StintLength...")

df["StintLength"] = pd.cut(
    df["TyreLife"],
    bins=[
        0,
        10,
        25,
        100
    ],
    labels=[
        "Short",
        "Medium",
        "Long"
    ],
    include_lowest=True
)

print(
    df["StintLength"]
    .value_counts()
)


# ============================================================
# 6. PACE CATEGORY
# ============================================================

print("\n6. Creating PaceCategory...")

q1 = df["LapTime"].quantile(
    0.33
)

q2 = df["LapTime"].quantile(
    0.66
)

df["PaceCategory"] = pd.cut(
    df["LapTime"],
    bins=[
        0,
        q1,
        q2,
        df["LapTime"].max() + 1
    ],
    labels=[
        "Fast",
        "Normal",
        "Slow"
    ],
    include_lowest=True
)

print(
    "Pace thresholds:"
)

print(
    f"Fast    : LapTime <= {q1:.3f}"
)

print(
    f"Normal  : {q1:.3f} < LapTime <= {q2:.3f}"
)

print(
    f"Slow    : LapTime > {q2:.3f}"
)


# ============================================================
# 7. COMPOUND ENCODING
# ============================================================

print("\n7. Creating CompoundCode...")

compound_map = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "Unknown": 4
}

df["Compound"] = (
    df["Compound"]
    .fillna("Unknown")
)

df["CompoundCode"] = (
    df["Compound"]
    .map(compound_map)
    .fillna(4)
    .astype(int)
)

print(
    df["Compound"]
    .value_counts()
)


# ============================================================
# 8. STRATEGY LABEL
# ============================================================

print("\n8. Creating Strategy...")

df["Strategy"] = (
    df["Compound"].astype(str)
    + "_"
    + df["StintLength"].astype(str)
)

print(
    "Unique strategies:",
    df["Strategy"].nunique()
)


# ============================================================
# 9. EVENT-LEVEL WEATHER PROBABILITY
# ============================================================

print("\n9. Calculating weather probabilities...")

# Determine the dominant weather state for each race.
#
# A race is classified as Wet if at least one official
# race-weather observation reports rainfall.
#
# Otherwise it is classified as Dry.

race_weather = (
    df.groupby("Race")["WeatherState"]
    .apply(
        lambda x:
        "Wet"
        if (x == "Wet").any()
        else "Dry"
    )
    .reset_index()
)

race_weather_counts = (
    race_weather["WeatherState"]
    .value_counts()
)

total_races = len(
    race_weather
)

dry_races = race_weather_counts.get(
    "Dry",
    0
)

wet_races = race_weather_counts.get(
    "Wet",
    0
)

p_dry = (
    dry_races / total_races
)

p_wet = (
    wet_races / total_races
)

print(
    f"\nTotal races : {total_races}"
)

print(
    f"Dry races   : {dry_races}"
)

print(
    f"Wet races   : {wet_races}"
)

print(
    f"P(Dry)      : {p_dry:.4f}"
)

print(
    f"P(Wet)      : {p_wet:.4f}"
)

# Assign race-level probability to every lap
df["WeatherProbability"] = np.where(
    df["WeatherState"] == "Wet",
    p_wet,
    p_dry
)


# ============================================================
# 10. NORMALIZED LAP TIME
# ============================================================

print("\n10. Normalizing LapTime...")

lap_min = df["LapTime"].min()

lap_max = df["LapTime"].max()

if lap_max > lap_min:

    df["LapTimeNormalized"] = (
        (
            df["LapTime"]
            - lap_min
        )
        /
        (
            lap_max
            - lap_min
        )
    )

else:

    df["LapTimeNormalized"] = 0


# ============================================================
# 11. NORMALIZED TYRE WEAR
# ============================================================

print("\n11. Normalizing TyreWear...")

wear_min = df["TyreWear"].min()

wear_max = df["TyreWear"].max()

if wear_max > wear_min:

    df["TyreWearNormalized"] = (
        (
            df["TyreWear"]
            - wear_min
        )
        /
        (
            wear_max
            - wear_min
        )
    )

else:

    df["TyreWearNormalized"] = 0


# ============================================================
# 12. PAYOFF SCORE
# ============================================================

print("\n12. Creating PayoffScore...")

# Payoff rewards:
#
# 1. Faster lap time
# 2. Lower tyre wear
#
# Weight:
# 70% = race pace
# 30% = tyre condition
#
# Higher score = better payoff.

df["PayoffScore"] = (
    0.7
    * (
        1
        - df["LapTimeNormalized"]
    )
    +
    0.3
    * (
        1
        - df["TyreWearNormalized"]
    )
)


# ============================================================
# 13. PAYOFF VALIDATION
# ============================================================

print("\nPayoffScore statistics:")

print(
    df["PayoffScore"]
    .describe()
)

print(
    "\nPayoffScore range:",
    round(
        df["PayoffScore"].min(),
        4
    ),
    "-",
    round(
        df["PayoffScore"].max(),
        4
    )
)


# ============================================================
# 14. FINAL FEATURE VALIDATION
# ============================================================

new_features = [
    "WeatherState",
    "TemperatureLevel",
    "TyreWear",
    "PitStop",
    "StintLength",
    "PaceCategory",
    "CompoundCode",
    "Strategy",
    "WeatherProbability",
    "LapTimeNormalized",
    "TyreWearNormalized",
    "PayoffScore"
]

print("\n" + "=" * 70)
print("NEW FEATURES")
print("=" * 70)

for feature in new_features:

    print(
        f"{feature:25s} : "
        f"{'Created' if feature in df.columns else 'MISSING'}"
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values in new features:")

for feature in new_features:

    if feature in df.columns:

        print(
            f"{feature:25s}: "
            f"{df[feature].isna().sum()}"
        )


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    "\nFinal dataset shape:"
)

print(
    df.shape
)

print(
    "\nSaved to:"
)

print(
    OUTPUT
)

print("\nDone.")