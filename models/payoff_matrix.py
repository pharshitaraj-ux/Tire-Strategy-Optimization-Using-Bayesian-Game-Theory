import os
import pandas as pd

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv(DATA)

# -------------------------------------------------
# Selected Players
# -------------------------------------------------

players = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

df = df[df["Team"].isin(players)]

# -------------------------------------------------
# Selected Strategies
# -------------------------------------------------

strategies = [
    "SOFT_Short",
    "HARD_Short",
    "MEDIUM_Short",
    "SOFT_Medium"
]

df = df[df["Strategy"].isin(strategies)]

# -------------------------------------------------
# Dry Weather Payoff Matrix
# -------------------------------------------------

dry = df[df["WeatherState"] == "Dry"]

payoff_dry = (
    dry.pivot_table(
        values="PayoffScore",
        index="Team",
        columns="Strategy",
        aggfunc="mean"
    )
)

payoff_dry = payoff_dry.round(4)

# -------------------------------------------------
# Wet Weather Payoff Matrix
# -------------------------------------------------

wet = df[df["WeatherState"] == "Wet"]

payoff_wet = (
    wet.pivot_table(
        values="PayoffScore",
        index="Team",
        columns="Strategy",
        aggfunc="mean"
    )
)

payoff_wet = payoff_wet.round(4)

# -------------------------------------------------
# Save CSV Files
# -------------------------------------------------

payoff_dry.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "payoff_matrix_dry.csv"
    )
)

payoff_wet.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "payoff_matrix_wet.csv"
    )
)

# -------------------------------------------------
# Save Text Summary
# -------------------------------------------------

summary = os.path.join(
    OUTPUT_DIR,
    "payoff_summary.txt"
)

with open(summary, "w") as f:

    f.write("PAYOFF MATRIX SUMMARY\n\n")

    f.write("Players\n")
    f.write("-----------------\n")
    for p in players:
        f.write(p + "\n")

    f.write("\nStrategies\n")
    f.write("-----------------\n")
    for s in strategies:
        f.write(s + "\n")

    f.write("\nDry Weather Payoff Matrix\n")
    f.write("-----------------\n")
    f.write(payoff_dry.to_string())

    f.write("\n\nWet Weather Payoff Matrix\n")
    f.write("-----------------\n")
    f.write(payoff_wet.to_string())

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("=" * 60)
print("PAYOFF MATRIX CREATED")
print("=" * 60)

print("\nPlayers")
print(players)

print("\nStrategies")
print(strategies)

print("\nDry Weather Payoff Matrix")
print(payoff_dry)

print("\nWet Weather Payoff Matrix")
print(payoff_wet)

print("\nFiles Saved To")
print(OUTPUT_DIR)