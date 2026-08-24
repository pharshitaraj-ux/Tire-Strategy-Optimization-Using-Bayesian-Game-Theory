import os
import pandas as pd
import numpy as np

# ============================================================
# REAL-TIME BAYESIAN STRATEGY
# STEP 3 - EXPECTED PAYOFF CALCULATION
# ============================================================

print("=" * 70)
print("REAL-TIME BAYESIAN STRATEGY")
print("STEP 3 - EXPECTED PAYOFF CALCULATION")
print("=" * 70)

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# ------------------------------------------------------------
# INPUT PATHS
# ------------------------------------------------------------

PAYOFF_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "01_intermediate_payoff",
    "extended_payoff_data.csv"
)

WEATHER_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "02_weather_probability",
    "weather_probability_scenarios.csv"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "03_expected_payoff"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD INPUT DATA
# ------------------------------------------------------------

payoff_df = pd.read_csv(PAYOFF_PATH)
weather_df = pd.read_csv(WEATHER_PATH)

print("\nPayoff Data Shape:")
print(payoff_df.shape)

print("\nWeather Scenario Shape:")
print(weather_df.shape)

# ------------------------------------------------------------
# SELECTED TEAMS
# ------------------------------------------------------------

TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

# ------------------------------------------------------------
# STRATEGIES
# ------------------------------------------------------------

STRATEGIES = [
    "SOFT_Short",
    "MEDIUM_Short",
    "HARD_Short",
    "INTERMEDIATE_Medium"
]

# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_payoff_columns = [
    "Team",
    "Strategy",
    "WeatherState",
    "AveragePayoff"
]

required_weather_columns = [
    "WetProbability",
    "DryProbability"
]

missing_payoff = [
    col for col in required_payoff_columns
    if col not in payoff_df.columns
]

missing_weather = [
    col for col in required_weather_columns
    if col not in weather_df.columns
]

if missing_payoff:
    raise ValueError(
        f"Missing payoff columns: {missing_payoff}"
    )

if missing_weather:
    raise ValueError(
        f"Missing weather columns: {missing_weather}"
    )

# ------------------------------------------------------------
# CREATE PAYOFF LOOKUP
# ------------------------------------------------------------

payoff_lookup = {}

for _, row in payoff_df.iterrows():

    key = (
        row["Team"],
        row["Strategy"],
        row["WeatherState"]
    )

    payoff_lookup[key] = row["AveragePayoff"]

# ------------------------------------------------------------
# EXPECTED PAYOFF CALCULATION
# ------------------------------------------------------------

records = []

for _, weather_row in weather_df.iterrows():

    wet_probability = float(
        weather_row["WetProbability"]
    )

    dry_probability = float(
        weather_row["DryProbability"]
    )

    for team in TEAMS:

        for strategy in STRATEGIES:

            dry_payoff = payoff_lookup.get(
                (team, strategy, "Dry"),
                np.nan
            )

            wet_payoff = payoff_lookup.get(
                (team, strategy, "Wet"),
                np.nan
            )

            # ------------------------------------------------
            # BOTH WEATHER PAYOFFS AVAILABLE
            # ------------------------------------------------

            if (
                not pd.isna(dry_payoff)
                and not pd.isna(wet_payoff)
            ):

                expected_payoff = (
                    dry_probability * dry_payoff
                    +
                    wet_probability * wet_payoff
                )

                status = "Available"

            else:

                expected_payoff = np.nan
                status = "Insufficient weather payoff data"

            records.append({
                "Team": team,
                "Strategy": strategy,
                "WetProbability": wet_probability,
                "DryProbability": dry_probability,
                "DryPayoff": dry_payoff,
                "WetPayoff": wet_payoff,
                "ExpectedPayoff": expected_payoff,
                "Status": status
            })

expected_df = pd.DataFrame(records)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPECTED PAYOFF RESULTS")
print("=" * 70)

print(
    expected_df.to_string(index=False)
)

# ------------------------------------------------------------
# FIND BEST STRATEGY FOR EACH TEAM
# ------------------------------------------------------------

available_df = expected_df[
    expected_df["ExpectedPayoff"].notna()
].copy()

best_records = []

for (
    team,
    wet_probability
), group in available_df.groupby(
    ["Team", "WetProbability"]
):

    best_row = group.loc[
        group["ExpectedPayoff"].idxmax()
    ]

    best_records.append({
        "Team": team,
        "WetProbability": wet_probability,
        "DryProbability": best_row["DryProbability"],
        "OptimalStrategy": best_row["Strategy"],
        "ExpectedPayoff": best_row["ExpectedPayoff"]
    })

optimal_df = pd.DataFrame(best_records)

# ------------------------------------------------------------
# SAVE COMPLETE EXPECTED PAYOFF DATA
# ------------------------------------------------------------

expected_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "expected_payoff_scenarios.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SAVE OPTIMAL STRATEGIES
# ------------------------------------------------------------

optimal_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "optimal_strategy_by_probability.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# HISTORICAL BASELINE
# ------------------------------------------------------------

historical_rows = weather_df[
    weather_df["HistoricalBaseline"] == True
].copy()

if len(historical_rows) > 0:

    baseline_wet = float(
        historical_rows.iloc[0]["WetProbability"]
    )

    baseline_dry = float(
        historical_rows.iloc[0]["DryProbability"]
    )

else:

    baseline_wet = np.nan
    baseline_dry = np.nan

baseline_records = []

for team in TEAMS:

    team_data = expected_df[
        (expected_df["Team"] == team) &
        (expected_df["WetProbability"] == baseline_wet)
    ]

    if len(team_data) > 0:

        valid_data = team_data[
            team_data["ExpectedPayoff"].notna()
        ]

        if len(valid_data) > 0:

            best = valid_data.loc[
                valid_data["ExpectedPayoff"].idxmax()
            ]

            baseline_records.append({
                "Team": team,
                "WetProbability": baseline_wet,
                "DryProbability": baseline_dry,
                "OptimalStrategy": best["Strategy"],
                "ExpectedPayoff": best["ExpectedPayoff"]
            })

baseline_df = pd.DataFrame(
    baseline_records
)

baseline_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_baseline_strategy.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# STRATEGY SWITCH DETECTION
# ------------------------------------------------------------

switch_records = []

for team in TEAMS:

    team_data = optimal_df[
        optimal_df["Team"] == team
    ].sort_values(
        "WetProbability"
    )

    previous_strategy = None

    for _, row in team_data.iterrows():

        current_strategy = row["OptimalStrategy"]

        if (
            previous_strategy is not None
            and current_strategy != previous_strategy
        ):

            switch_records.append({
                "Team": team,
                "WetProbability": row["WetProbability"],
                "PreviousStrategy": previous_strategy,
                "NewStrategy": current_strategy
            })

        previous_strategy = current_strategy

switch_df = pd.DataFrame(
    switch_records
)

switch_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_switch_points.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "expected_payoff_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "REAL-TIME BAYESIAN STRATEGY\n"
    )

    file.write(
        "STEP 3 - EXPECTED PAYOFF CALCULATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Expected payoff equation:\n\n"
    )

    file.write(
        "EU(s) = P(Dry) * U(s,Dry) + "
        "P(Wet) * U(s,Wet)\n\n"
    )

    file.write(
        "where:\n"
    )

    file.write(
        "- EU(s) = Expected payoff of strategy s\n"
    )

    file.write(
        "- P(Dry) = Probability of dry weather\n"
    )

    file.write(
        "- P(Wet) = Probability of wet weather\n"
    )

    file.write(
        "- U(s,Dry) = Payoff under dry weather\n"
    )

    file.write(
        "- U(s,Wet) = Payoff under wet weather\n\n"
    )

    file.write(
        f"Historical Wet Probability: "
        f"{baseline_wet:.6f}\n"
    )

    file.write(
        f"Historical Dry Probability: "
        f"{baseline_dry:.6f}\n\n"
    )

    file.write(
        "HISTORICAL BASELINE STRATEGIES\n"
    )

    file.write("=" * 70 + "\n\n")

    if len(baseline_df) > 0:
        file.write(
            baseline_df.to_string(index=False)
        )

    file.write(
        "\n\nSTRATEGY SWITCH POINTS\n"
    )

    file.write("=" * 70 + "\n\n")

    if len(switch_df) > 0:
        file.write(
            switch_df.to_string(index=False)
        )
    else:
        file.write(
            "No strategy switching points detected "
            "among available payoff combinations.\n"
        )

    file.write(
        "\n\nNOTE:\n"
    )

    file.write(
        "Expected payoffs are calculated using observed "
        "2025 F1 payoff values and weather probability "
        "scenarios. Missing weather-strategy combinations "
        "are not assigned synthetic payoff values.\n"
    )

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETED")
print("=" * 70)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print(" - expected_payoff_scenarios.csv")
print(" - optimal_strategy_by_probability.csv")
print(" - historical_baseline_strategy.csv")
print(" - strategy_switch_points.csv")
print(" - expected_payoff_summary.txt")

print("\n" + "=" * 70)
print("HISTORICAL BASELINE STRATEGIES")
print("=" * 70)

if len(baseline_df) > 0:
    print(baseline_df.to_string(index=False))
else:
    print("No valid historical baseline strategies available.")