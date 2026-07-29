import os
import pandas as pd

# ------------------------------------
# Paths
# ------------------------------------

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

# ------------------------------------
# Load Data
# ------------------------------------

df = pd.read_csv(DATA)

# ------------------------------------
# Players
# ------------------------------------

players = sorted(df["Team"].dropna().unique())

# ------------------------------------
# Types
# ------------------------------------

types = sorted(df["WeatherState"].dropna().unique())

# ------------------------------------
# Strategies
# ------------------------------------

strategies = sorted(df["Strategy"].dropna().unique())

# ------------------------------------
# Prior Probabilities
# ------------------------------------

priors = (
    df["WeatherState"]
      .value_counts(normalize=True)
)

# ------------------------------------
# Average Payoff per Strategy
# ------------------------------------

payoff = (
    df.groupby("Strategy")["PayoffScore"]
      .mean()
      .sort_values(ascending=False)
)

# ------------------------------------
# Save Summary
# ------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "game_definition.txt"
)

with open(output_file, "w") as f:

    f.write("BAYESIAN GAME FORMULATION\n\n")

    f.write("PLAYERS\n")
    f.write(str(players))
    f.write("\n\n")

    f.write("TYPES\n")
    f.write(str(types))
    f.write("\n\n")

    f.write("STRATEGIES\n")
    f.write(strategies.__str__())
    f.write("\n\n")

    f.write("PRIOR PROBABILITIES\n")
    f.write(priors.to_string())
    f.write("\n\n")

    f.write("AVERAGE PAYOFF BY STRATEGY\n")
    f.write(payoff.to_string())

print("="*60)
print("BAYESIAN GAME MODEL CREATED")
print("="*60)

print("\nPlayers")
print(players)

print("\nTypes")
print(types)

print("\nStrategies")
print(strategies)

print("\nPrior Probabilities")
print(priors)

print("\nTop Strategies")
print(payoff.head())

print("\nSaved to:")
print(output_file)