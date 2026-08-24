import os
import pandas as pd
import numpy as np

# ============================================================
# REAL-TIME BAYESIAN STRATEGY
# STEP 1 - INTERMEDIATE PAYOFF EXTENSION
# ============================================================

print("=" * 70)
print("REAL-TIME BAYESIAN STRATEGY")
print("STEP 1 - INTERMEDIATE PAYOFF EXTENSION")
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
# INPUT DATA
# ------------------------------------------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

# ------------------------------------------------------------
# REAL-TIME OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "01_intermediate_payoff"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

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
# MODELLED STRATEGIES
# ------------------------------------------------------------

STRATEGIES = [
    "SOFT_Short",
    "MEDIUM_Short",
    "HARD_Short",
    "INTERMEDIATE_Medium"
]

# ------------------------------------------------------------
# WEATHER STATES
# ------------------------------------------------------------

WEATHER_STATES = [
    "Dry",
    "Wet"
]

# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Team",
    "Strategy",
    "WeatherState",
    "PayoffScore"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ------------------------------------------------------------
# FILTER TEAMS
# ------------------------------------------------------------

game_df = df[
    df["Team"].isin(TEAMS)
].copy()

print("\nFiltered Dataset Shape:")
print(game_df.shape)

# ------------------------------------------------------------
# DATA AVAILABILITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY / WEATHER DATA AVAILABILITY")
print("=" * 70)

availability_records = []

for team in TEAMS:

    print(f"\n{team}")

    team_df = game_df[
        game_df["Team"] == team
    ]

    for strategy in STRATEGIES:

        strategy_df = team_df[
            team_df["Strategy"] == strategy
        ]

        dry_count = len(
            strategy_df[
                strategy_df["WeatherState"] == "Dry"
            ]
        )

        wet_count = len(
            strategy_df[
                strategy_df["WeatherState"] == "Wet"
            ]
        )

        print(
            f"{strategy:25s}"
            f"Dry={dry_count:5d} "
            f"Wet={wet_count:5d}"
        )

        availability_records.append({
            "Team": team,
            "Strategy": strategy,
            "DryObservations": dry_count,
            "WetObservations": wet_count
        })

availability_df = pd.DataFrame(
    availability_records
)

availability_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_weather_availability.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# CALCULATE PAYOFFS
# ------------------------------------------------------------

records = []

for team in TEAMS:

    for strategy in STRATEGIES:

        for weather in WEATHER_STATES:

            subset = game_df[
                (game_df["Team"] == team) &
                (game_df["Strategy"] == strategy) &
                (game_df["WeatherState"] == weather)
            ]

            observations = len(subset)

            if observations > 0:

                average_payoff = (
                    subset["PayoffScore"].mean()
                )

            else:

                average_payoff = np.nan

            records.append({
                "Team": team,
                "Strategy": strategy,
                "WeatherState": weather,
                "AveragePayoff": average_payoff,
                "Observations": observations
            })

payoff_df = pd.DataFrame(records)

# ------------------------------------------------------------
# DISPLAY PAYOFF DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXTENDED PAYOFF DATA")
print("=" * 70)

print(
    payoff_df.to_string(index=False)
)

# ------------------------------------------------------------
# DRY PAYOFF MATRIX
# ------------------------------------------------------------

dry_matrix = payoff_df[
    payoff_df["WeatherState"] == "Dry"
].pivot(
    index="Team",
    columns="Strategy",
    values="AveragePayoff"
)

# ------------------------------------------------------------
# WET PAYOFF MATRIX
# ------------------------------------------------------------

wet_matrix = payoff_df[
    payoff_df["WeatherState"] == "Wet"
].pivot(
    index="Team",
    columns="Strategy",
    values="AveragePayoff"
)

# ------------------------------------------------------------
# SAVE COMPLETE PAYOFF DATA
# ------------------------------------------------------------

payoff_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extended_payoff_data.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SAVE DRY MATRIX
# ------------------------------------------------------------

dry_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extended_payoff_matrix_dry.csv"
    )
)

# ------------------------------------------------------------
# SAVE WET MATRIX
# ------------------------------------------------------------

wet_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extended_payoff_matrix_wet.csv"
    )
)

# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "intermediate_payoff_summary.txt"
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
        "STEP 1 - INTERMEDIATE PAYOFF EXTENSION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Selected Teams:\n"
    )

    for team in TEAMS:
        file.write(f"- {team}\n")

    file.write("\nModelled Strategies:\n")

    for strategy in STRATEGIES:
        file.write(f"- {strategy}\n")

    file.write("\nWeather States:\n")

    for weather in WEATHER_STATES:
        file.write(f"- {weather}\n")

    file.write(
        "\n\nPayoff Data\n"
    )

    file.write("=" * 70 + "\n")

    file.write(
        payoff_df.to_string(index=False)
    )

    file.write(
        "\n\nNOTE:\n"
    )

    file.write(
        "Payoffs are calculated from observed 2025 F1 data "
        "using the existing PayoffScore feature. "
        "No synthetic payoff values are introduced.\n"
    )

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 COMPLETED")
print("=" * 70)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print(" - strategy_weather_availability.csv")
print(" - extended_payoff_data.csv")
print(" - extended_payoff_matrix_dry.csv")
print(" - extended_payoff_matrix_wet.csv")
print(" - intermediate_payoff_summary.txt")

print("\n" + "=" * 70)
print("DRY WEATHER PAYOFF MATRIX")
print("=" * 70)

print(dry_matrix)

print("\n" + "=" * 70)
print("WET WEATHER PAYOFF MATRIX")
print("=" * 70)

print(wet_matrix)