import os
import pandas as pd
import numpy as np

# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------
# Load Data
# ---------------------------------------

df = pd.read_csv(DATA)

# ---------------------------------------
# 1. Weather State
# ---------------------------------------

df["WeatherState"] = np.where(
    df["Rainfall"] == 1,
    "Wet",
    "Dry"
)

# ---------------------------------------
# 2. Track Temperature Level
# ---------------------------------------

df["TemperatureLevel"] = pd.cut(
    df["TrackTemp"],
    bins=[0,30,40,100],
    labels=["Low","Medium","High"]
)

# ---------------------------------------
# 3. Tyre Wear Level
# ---------------------------------------

df["TyreWear"] = pd.cut(
    df["TyreLife"],
    bins=[0,10,25,100],
    labels=["Low","Medium","High"]
)

# ---------------------------------------
# 4. Pit Stop Indicator
# ---------------------------------------

df["PitStop"] = np.where(
    df["PitInTime"].notna(),
    1,
    0
)

# ---------------------------------------
# 5. Stint Length
# ---------------------------------------

df["StintLength"] = pd.cut(
    df["TyreLife"],
    bins=[0,10,25,100],
    labels=["Short","Medium","Long"]
)

# ---------------------------------------
# 6. Pace Category
# ---------------------------------------

q1 = df["LapTime"].quantile(0.33)
q2 = df["LapTime"].quantile(0.66)

df["PaceCategory"] = pd.cut(
    df["LapTime"],
    bins=[0,q1,q2,df["LapTime"].max()+1],
    labels=["Fast","Normal","Slow"]
)

# ---------------------------------------
# 7. Compound Encoding
# ---------------------------------------

compound_map = {
    "SOFT":0,
    "MEDIUM":1,
    "HARD":2,
    "INTERMEDIATE":3,
    "Unknown":4
}

df["CompoundCode"] = df["Compound"].map(compound_map)

# ---------------------------------------
# 8. Strategy Label
# ---------------------------------------

df["Strategy"] = (
    df["Compound"] +
    "_" +
    df["StintLength"].astype(str)
)

# ---------------------------------------
# 9. Weather Probability
# ---------------------------------------

weather_prob = (
    df["WeatherState"]
    .value_counts(normalize=True)
)

df["WeatherProbability"] = (
    df["WeatherState"]
    .map(weather_prob)
)

# ---------------------------------------
# 10. Payoff Score
# ---------------------------------------

lap_norm = (
    df["LapTime"] -
    df["LapTime"].min()
) / (
    df["LapTime"].max() -
    df["LapTime"].min()
)

wear_norm = (
    df["TyreLife"] -
    df["TyreLife"].min()
) / (
    df["TyreLife"].max() -
    df["TyreLife"].min()
)

df["PayoffScore"] = (
    0.7*(1-lap_norm) +
    0.3*(1-wear_norm)
)

# ---------------------------------------
# Save Dataset
# ---------------------------------------

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "f1_2025_feature_engineered.csv"
)

df.to_csv(
    OUTPUT,
    index=False
)

print("="*60)
print("FEATURE ENGINEERING COMPLETED")
print("="*60)

print("\nNew Dataset Shape")
print(df.shape)

print("\nNew Features Added")

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
    "PayoffScore"
]

print(new_features)

print("\nSaved to:")
print(OUTPUT)