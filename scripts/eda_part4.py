import os
import pandas as pd
import matplotlib.pyplot as plt

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
    "eda",
    "04_circuit_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)

# ---------------------------------------
# 1. Average Lap Time by Circuit
# ---------------------------------------

plt.figure(figsize=(12,6))

avg_lap = (
    df.groupby("Circuit")["LapTime"]
      .mean()
      .sort_values()
)

avg_lap.plot(kind="bar")

plt.title("Average Lap Time by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Average Lap Time (sec)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "average_laptime_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 2. Average Tyre Life by Circuit
# ---------------------------------------

plt.figure(figsize=(12,6))

avg_tyre = (
    df.groupby("Circuit")["TyreLife"]
      .mean()
      .sort_values()
)

avg_tyre.plot(kind="bar", color="orange")

plt.title("Average Tyre Life by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Average Tyre Life (laps)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "average_tyrelife_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 3. Tyre Compound Usage by Circuit
# ---------------------------------------

compound = pd.crosstab(
    df["Circuit"],
    df["Compound"]
)

compound.plot(
    kind="bar",
    stacked=True,
    figsize=(14,6)
)

plt.title("Tyre Compound Usage by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Number of Laps")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "compound_usage_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 4. Average Air Temperature by Circuit
# ---------------------------------------

plt.figure(figsize=(12,6))

air = (
    df.groupby("Circuit")["AirTemp"]
      .mean()
      .sort_values()
)

air.plot(kind="bar", color="green")

plt.title("Average Air Temperature by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Air Temperature (°C)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "airtemp_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 5. Average Track Temperature by Circuit
# ---------------------------------------

plt.figure(figsize=(12,6))

track = (
    df.groupby("Circuit")["TrackTemp"]
      .mean()
      .sort_values()
)

track.plot(kind="bar", color="red")

plt.title("Average Track Temperature by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Track Temperature (°C)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "tracktemp_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 6. Rainfall Occurrence by Circuit
# ---------------------------------------

rain = pd.crosstab(
    df["Circuit"],
    df["Rainfall"]
)

rain.plot(
    kind="bar",
    stacked=True,
    figsize=(14,6)
)

plt.title("Rainfall Occurrence by Circuit")
plt.xlabel("Circuit")
plt.ylabel("Number of Laps")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "rainfall_by_circuit.png"
))
plt.close()

# ---------------------------------------
# 7. Average Lap Time by Team
# ---------------------------------------

plt.figure(figsize=(12,6))

team = (
    df.groupby("Team")["LapTime"]
      .mean()
      .sort_values()
)

team.plot(kind="bar")

plt.title("Average Lap Time by Team")
plt.xlabel("Team")
plt.ylabel("Average Lap Time (sec)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(os.path.join(
    OUTPUT_DIR,
    "average_laptime_by_team.png"
))
plt.close()

print("="*60)
print("EDA PART 4 COMPLETED")
print("="*60)

print("\nAverage Lap Time by Circuit")
print(avg_lap)

print("\nAverage Tyre Life by Circuit")
print(avg_tyre)