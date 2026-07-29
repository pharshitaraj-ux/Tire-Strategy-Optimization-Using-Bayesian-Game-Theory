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

OUT= os.path.join(
    BASE_DIR,
    "outputs",
    "eda",
    "02_tyre_analysis"
)
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# ---------------------------------------
# 1. Tyre Compound Distribution
# ---------------------------------------

plt.figure(figsize=(8,5))

df["Compound"].value_counts().plot(kind="bar")

plt.title("Tyre Compound Distribution")
plt.xlabel("Compound")
plt.ylabel("Number of Laps")

plt.tight_layout()

plt.savefig(os.path.join(OUT,"compound_distribution.png"))
plt.close()

# ---------------------------------------
# 2. Tyre Life Distribution
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.hist(df["TyreLife"],bins=30)

plt.title("Tyre Life Distribution")
plt.xlabel("Tyre Life (laps)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(os.path.join(OUT,"tyre_life_distribution.png"))
plt.close()

# ---------------------------------------
# 3. Average Lap Time by Compound
# ---------------------------------------

order = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "Unknown"]

avg = (
    df.groupby("Compound")["LapTime"]
      .mean()
      .reindex(order)
)

plt.figure(figsize=(8,5))

avg.plot(kind="bar")

plt.title("Average Lap Time by Compound")

plt.ylabel("Average Lap Time (seconds)")

plt.tight_layout()

plt.savefig(os.path.join(OUT,"avg_laptime_compound.png"))
plt.close()

# ---------------------------------------
# 4. Tyre Life vs Lap Time
# ---------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["TyreLife"],
    df["LapTime"],
    alpha=0.3,
    s=10
)

plt.xlabel("Tyre Life")

plt.ylabel("Lap Time (sec)")

plt.title("Tyre Life vs Lap Time")

plt.tight_layout()

plt.savefig(os.path.join(OUT,"tyrelife_vs_laptime.png"))
plt.close()

# ---------------------------------------
# 5. Team-wise Tyre Usage
# ---------------------------------------

team_compound=pd.crosstab(
    df["Team"],
    df["Compound"]
)

team_compound.plot(
    kind="bar",
    stacked=True,
    figsize=(12,6)
)

plt.title("Team-wise Tyre Compound Usage")

plt.ylabel("Number of Laps")

plt.tight_layout()

plt.savefig(os.path.join(OUT,"team_tyre_usage.png"))
plt.close()

print("="*60)
print("EDA PART 2 COMPLETED")
print("="*60)

print("\nAverage Lap Time by Compound")

print(avg)