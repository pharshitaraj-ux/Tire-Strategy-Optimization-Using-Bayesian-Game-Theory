import os
import pandas as pd

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DRY = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game",
    "payoff_matrix_dry.csv"
)

WET = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game",
    "payoff_matrix_wet.csv"
)

FEATURE_DATA = os.path.join(
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

# ---------------------------------------------------
# Load Files
# ---------------------------------------------------

dry = pd.read_csv(DRY, index_col=0)
wet = pd.read_csv(WET, index_col=0)
df = pd.read_csv(FEATURE_DATA)

# ---------------------------------------------------
# Prior Probabilities
# ---------------------------------------------------

priors = (
    df["WeatherState"]
      .value_counts(normalize=True)
)

P_DRY = priors.get("Dry", 0)
P_WET = priors.get("Wet", 0)

print("\nPrior Probabilities")
print(priors)

# ---------------------------------------------------
# Common Strategies
# ---------------------------------------------------

common_strategies = sorted(
    list(
        set(dry.columns).intersection(
            set(wet.columns)
        )
    )
)

expected = pd.DataFrame(index=dry.index)

# ---------------------------------------------------
# Expected Payoff
# ---------------------------------------------------

for strategy in common_strategies:

    expected[strategy] = (
        P_DRY * dry[strategy] +
        P_WET * wet[strategy]
    )

expected = expected.round(4)

# ---------------------------------------------------
# Optimal Strategy
# ---------------------------------------------------

optimal = pd.DataFrame({
    "OptimalStrategy":
        expected.idxmax(axis=1),
    "ExpectedPayoff":
        expected.max(axis=1)
})

optimal = optimal.round(4)

# ---------------------------------------------------
# Save Outputs
# ---------------------------------------------------

expected.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "expected_payoffs.csv"
    )
)

optimal.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "optimal_strategy.csv"
    )
)

summary = os.path.join(
    OUTPUT_DIR,
    "bayesian_nash_summary.txt"
)

with open(summary, "w") as f:

    f.write("BAYESIAN NASH EQUILIBRIUM\n\n")

    f.write("Prior Probabilities\n")
    f.write("---------------------\n")
    f.write(priors.to_string())

    f.write("\n\nExpected Payoffs\n")
    f.write("---------------------\n")
    f.write(expected.to_string())

    f.write("\n\nOptimal Strategies\n")
    f.write("---------------------\n")
    f.write(optimal.to_string())

# ---------------------------------------------------
# Display
# ---------------------------------------------------

print("\n" + "="*60)
print("BAYESIAN NASH EQUILIBRIUM COMPUTED")
print("="*60)

print("\nExpected Payoffs")
print(expected)

print("\nOptimal Strategy")
print(optimal)

print("\nSaved to")
print(OUTPUT_DIR)