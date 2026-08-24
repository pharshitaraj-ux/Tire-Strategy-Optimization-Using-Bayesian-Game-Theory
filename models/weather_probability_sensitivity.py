import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# WEATHER PROBABILITY SENSITIVITY ANALYSIS
# ============================================================

print("=" * 70)
print("WEATHER PROBABILITY SENSITIVITY ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAME_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

OUTPUT_DIR = os.path.join(
    GAME_DIR,
    "weather_sensitivity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Teams and common strategies
# ------------------------------------------------------------

TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

STRATEGIES = [
    "SOFT_Short",
    "HARD_Short",
    "MEDIUM_Short",
    "SOFT_Medium"
]

# ------------------------------------------------------------
# Load payoff matrices
# ------------------------------------------------------------

dry_file = os.path.join(
    GAME_DIR,
    "payoff_matrix_dry.csv"
)

wet_file = os.path.join(
    GAME_DIR,
    "payoff_matrix_wet.csv"
)

dry = pd.read_csv(dry_file, index_col=0)
wet = pd.read_csv(wet_file, index_col=0)

# ------------------------------------------------------------
# Standardize strategy names
# ------------------------------------------------------------

dry.index = dry.index.astype(str)
dry.columns = dry.columns.astype(str)

wet.index = wet.index.astype(str)
wet.columns = wet.columns.astype(str)

# ------------------------------------------------------------
# IMPORTANT:
# Some strategies may not exist in the wet matrix.
#
# We need to construct missing values using the team's
# strategy-specific payoff information from the feature data.
# ------------------------------------------------------------

feature_file = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

df = pd.read_csv(feature_file)

# ------------------------------------------------------------
# Calculate strategy payoffs directly from feature dataset
# ------------------------------------------------------------

strategy_payoff = (
    df[
        df["Team"].isin(TEAMS) &
        df["Strategy"].isin(STRATEGIES)
    ]
    .groupby(["Team", "WeatherState", "Strategy"])["PayoffScore"]
    .mean()
)

# ------------------------------------------------------------
# Build complete Dry/Wet payoff dictionaries
# ------------------------------------------------------------

dry_payoff = {}
wet_payoff = {}

for team in TEAMS:

    dry_payoff[team] = {}
    wet_payoff[team] = {}

    for strategy in STRATEGIES:

        # ----------------------------------------------------
        # Dry payoff
        # ----------------------------------------------------

        try:
            value = strategy_payoff.loc[
                (team, "Dry", strategy)
            ]

            dry_payoff[team][strategy] = float(value)

        except KeyError:

            # fallback to payoff matrix
            try:
                dry_payoff[team][strategy] = float(
                    dry.loc[team, strategy]
                )
            except KeyError:
                dry_payoff[team][strategy] = None

        # ----------------------------------------------------
        # Wet payoff
        # ----------------------------------------------------

        try:
            value = strategy_payoff.loc[
                (team, "Wet", strategy)
            ]

            wet_payoff[team][strategy] = float(value)

        except KeyError:

            # fallback to existing wet matrix
            try:
                wet_payoff[team][strategy] = float(
                    wet.loc[team, strategy]
                )
            except KeyError:
                wet_payoff[team][strategy] = None


# ------------------------------------------------------------
# Display available strategies
# ------------------------------------------------------------

print("\nTeams:")
print(TEAMS)

print("\nStrategies:")
print(STRATEGIES)

print("\nChecking payoff availability...")

for team in TEAMS:

    print(f"\n{team}")

    for strategy in STRATEGIES:

        print(
            f"{strategy:15s} "
            f"Dry={dry_payoff[team][strategy]} "
            f"Wet={wet_payoff[team][strategy]}"
        )


# ============================================================
# BAYESIAN EXPECTED PAYOFF
# ============================================================

def expected_payoff(
    dry_value,
    wet_value,
    wet_probability
):

    dry_probability = 1 - wet_probability

    return (
        dry_probability * dry_value
        +
        wet_probability * wet_value
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

results = []

probabilities = [
    i / 100
    for i in range(0, 101, 5)
]

for team in TEAMS:

    for wet_probability in probabilities:

        for strategy in STRATEGIES:

            dry_value = dry_payoff[team][strategy]
            wet_value = wet_payoff[team][strategy]

            # Skip strategies for which a payoff cannot be
            # reliably obtained
            if dry_value is None or wet_value is None:
                continue

            payoff = expected_payoff(
                dry_value,
                wet_value,
                wet_probability
            )

            results.append({
                "Team": team,
                "WetProbability": wet_probability,
                "DryProbability": 1 - wet_probability,
                "Strategy": strategy,
                "DryPayoff": dry_value,
                "WetPayoff": wet_value,
                "ExpectedPayoff": payoff
            })


sensitivity_df = pd.DataFrame(results)

# ------------------------------------------------------------
# Save complete sensitivity results
# ------------------------------------------------------------

sensitivity_file = os.path.join(
    OUTPUT_DIR,
    "weather_probability_sensitivity.csv"
)

sensitivity_df.to_csv(
    sensitivity_file,
    index=False
)


# ============================================================
# FIND OPTIMAL STRATEGY AT EACH WEATHER PROBABILITY
# ============================================================

optimal_rows = []

for team in TEAMS:

    team_data = sensitivity_df[
        sensitivity_df["Team"] == team
    ]

    for probability in probabilities:

        subset = team_data[
            team_data["WetProbability"] == probability
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["ExpectedPayoff"].idxmax()
        ]

        optimal_rows.append({
            "Team": team,
            "WetProbability": probability,
            "OptimalStrategy": best["Strategy"],
            "ExpectedPayoff": best["ExpectedPayoff"]
        })


optimal_df = pd.DataFrame(optimal_rows)

optimal_file = os.path.join(
    OUTPUT_DIR,
    "optimal_strategy_by_weather_probability.csv"
)

optimal_df.to_csv(
    optimal_file,
    index=False
)


# ============================================================
# DETECT STRATEGY SWITCH POINTS
# ============================================================

switch_rows = []

for team in TEAMS:

    team_optimal = optimal_df[
        optimal_df["Team"] == team
    ].sort_values("WetProbability")

    previous_strategy = None

    for _, row in team_optimal.iterrows():

        current_strategy = row["OptimalStrategy"]

        if (
            previous_strategy is not None
            and current_strategy != previous_strategy
        ):

            switch_rows.append({
                "Team": team,
                "WetProbability": row["WetProbability"],
                "PreviousStrategy": previous_strategy,
                "NewStrategy": current_strategy
            })

        previous_strategy = current_strategy


switch_df = pd.DataFrame(switch_rows)

switch_file = os.path.join(
    OUTPUT_DIR,
    "strategy_switch_points.csv"
)

switch_df.to_csv(
    switch_file,
    index=False
)


# ============================================================
# PLOT 1:
# OPTIMAL STRATEGY BY WEATHER PROBABILITY
# ============================================================

for team in TEAMS:

    team_data = sensitivity_df[
        sensitivity_df["Team"] == team
    ]

    plt.figure(figsize=(10, 6))

    for strategy in STRATEGIES:

        strategy_data = team_data[
            team_data["Strategy"] == strategy
        ]

        if strategy_data.empty:
            continue

        plt.plot(
            strategy_data["WetProbability"] * 100,
            strategy_data["ExpectedPayoff"],
            marker="o",
            label=strategy
        )

    plt.title(
        f"{team}: Strategy Payoff vs Wet Weather Probability"
    )

    plt.xlabel("Probability of Wet Weather (%)")
    plt.ylabel("Bayesian Expected Payoff")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    filename = (
        team.lower()
        .replace(" ", "_")
        + "_weather_sensitivity.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 2:
# OPTIMAL STRATEGY SWITCHES
# ============================================================

plt.figure(figsize=(10, 6))

for team in TEAMS:

    team_optimal = optimal_df[
        optimal_df["Team"] == team
    ]

    plt.plot(
        team_optimal["WetProbability"] * 100,
        team_optimal["ExpectedPayoff"],
        marker="o",
        label=team
    )

plt.title(
    "Optimal Bayesian Strategy Under Weather Uncertainty"
)

plt.xlabel("Probability of Wet Weather (%)")
plt.ylabel("Expected Payoff")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "optimal_strategy_weather_probability.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "weather_sensitivity_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write(
        "WEATHER PROBABILITY SENSITIVITY ANALYSIS\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Teams:\n"
    )

    for team in TEAMS:
        f.write(f"- {team}\n")

    f.write("\nStrategies:\n")

    for strategy in STRATEGIES:
        f.write(f"- {strategy}\n")

    f.write("\nStrategy Switch Points:\n\n")

    if switch_df.empty:

        f.write(
            "No strategy switches detected.\n"
        )

    else:

        f.write(
            switch_df.to_string(index=False)
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("WEATHER PROBABILITY SENSITIVITY ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print(" - weather_probability_sensitivity.csv")
print(" - optimal_strategy_by_weather_probability.csv")
print(" - strategy_switch_points.csv")
print(" - optimal_strategy_weather_probability.png")
print(" - *_weather_sensitivity.png")
print(" - weather_sensitivity_summary.txt")

print("\nStrategy Switch Points:")

if switch_df.empty:
    print("No strategy switches detected.")
else:
    print(switch_df.to_string(index=False))