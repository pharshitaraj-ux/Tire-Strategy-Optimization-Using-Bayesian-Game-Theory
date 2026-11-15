import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game",
    "weather_probability_sensitivity.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# ============================================================
# FERRARI vs MERCEDES
# ============================================================

plot_df = df[
    (df["Player1"] == "Ferrari") &
    (df["Player2"] == "Mercedes")
].copy()

# ============================================================
# CALCULATE EXPECTED PAYOFF FOR BOTH STRATEGIES
# ============================================================

strategies_to_plot = [
    "HARD_Short",
    "MEDIUM_Short"
]

# The CSV contains only the best strategy at each probability,
# so recover the two strategy payoff curves from the original
# weather-specific payoff matrices.

pair_dir = os.path.join(
    OUTPUT_DIR,
    "Ferrari_vs_Mercedes"
)

dry_file = os.path.join(
    pair_dir,
    "U1_dry.csv"
)

wet_file = os.path.join(
    pair_dir,
    "U1_wet.csv"
)

dry = pd.read_csv(
    dry_file,
    index_col=0
)

wet = pd.read_csv(
    wet_file,
    index_col=0
)

dry = dry.apply(
    pd.to_numeric,
    errors="coerce"
)

wet = wet.apply(
    pd.to_numeric,
    errors="coerce"
)

opponent = "HARD_Short"

probabilities = [
    i / 100
    for i in range(0, 101, 5)
]

curves = {}

for strategy in strategies_to_plot:

    values = []

    for p_wet in probabilities:

        dry_payoff = dry.loc[
            strategy,
            opponent
        ]

        wet_payoff = wet.loc[
            strategy,
            opponent
        ]

        if pd.isna(dry_payoff) or pd.isna(wet_payoff):

            values.append(float("nan"))

        else:

            expected = (
                (1 - p_wet) * dry_payoff
                +
                p_wet * wet_payoff
            )

            values.append(expected)

    curves[strategy] = values


# ============================================================
# FIND CROSSOVER
# ============================================================

crossover_probability = None

for i in range(len(probabilities)):

    hard_value = curves["HARD_Short"][i]
    medium_value = curves["MEDIUM_Short"][i]

    if (
        not pd.isna(hard_value)
        and not pd.isna(medium_value)
        and medium_value >= hard_value
    ):

        crossover_probability = probabilities[i] * 100
        break


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(11, 7))

for strategy in strategies_to_plot:

    plt.plot(
        [p * 100 for p in probabilities],
        curves[strategy],
        marker="o",
        label=strategy
    )

# Actual 2025 Bayesian probability
actual_probability = 12.5

plt.axvline(
    actual_probability,
    linestyle="--",
    label="2025 observed wet probability (12.5%)"
)

# Strategy crossover
if crossover_probability is not None:

    plt.axvline(
        crossover_probability,
        linestyle=":",
        label=f"Strategy switch ≈ {crossover_probability:.0f}%"
    )

plt.xlabel(
    "Wet Weather Probability (%)"
)

plt.ylabel(
    "Expected Payoff"
)

plt.title(
    "Bayesian Strategy Sensitivity: Ferrari vs Mercedes"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "final_weather_probability_sensitivity.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 70)
print("FINAL WEATHER SENSITIVITY GRAPH CREATED")
print("=" * 70)

print(
    f"\nStrategy switch approximately at: "
    f"{crossover_probability:.0f}% wet probability"
)

print(
    "\n2025 observed wet probability: 12.5%"
)

print(
    "\nOutput:"
)

print(output_file)

print("\nDone.")