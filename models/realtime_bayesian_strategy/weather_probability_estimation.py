import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# REAL-TIME BAYESIAN STRATEGY
# STEP 2 - WEATHER PROBABILITY ESTIMATION
# ============================================================

print("=" * 70)
print("REAL-TIME BAYESIAN STRATEGY")
print("STEP 2 - WEATHER PROBABILITY ESTIMATION")
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
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "02_weather_probability"
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
# REQUIRED COLUMN CHECK
# ------------------------------------------------------------

required_columns = [
    "Team",
    "WeatherState"
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
# FILTER SELECTED TEAMS
# ------------------------------------------------------------

game_df = df[
    df["Team"].isin(TEAMS)
].copy()

print("\nFiltered Dataset Shape:")
print(game_df.shape)

# ------------------------------------------------------------
# CLEAN WEATHER STATES
# ------------------------------------------------------------

game_df["WeatherState"] = (
    game_df["WeatherState"]
    .astype(str)
    .str.strip()
)

# Keep only modelled weather states
game_df = game_df[
    game_df["WeatherState"].isin(
        ["Dry", "Wet"]
    )
].copy()

# ------------------------------------------------------------
# HISTORICAL WEATHER COUNTS
# ------------------------------------------------------------

weather_counts = (
    game_df["WeatherState"]
    .value_counts()
    .reindex(
        ["Dry", "Wet"],
        fill_value=0
    )
)

total_observations = weather_counts.sum()

if total_observations == 0:
    raise ValueError(
        "No Dry/Wet observations available."
    )

# ------------------------------------------------------------
# HISTORICAL PRIOR PROBABILITIES
# ------------------------------------------------------------

dry_probability = (
    weather_counts["Dry"] /
    total_observations
)

wet_probability = (
    weather_counts["Wet"] /
    total_observations
)

print("\n" + "=" * 70)
print("HISTORICAL WEATHER PROBABILITY")
print("=" * 70)

print(
    f"Dry observations : {weather_counts['Dry']}"
)

print(
    f"Wet observations : {weather_counts['Wet']}"
)

print(
    f"Total observations: {total_observations}"
)

print(
    f"\nP(Dry) = {dry_probability:.6f}"
)

print(
    f"P(Wet) = {wet_probability:.6f}"
)

# ------------------------------------------------------------
# SAVE CURRENT / BASELINE WEATHER PROBABILITY
# ------------------------------------------------------------

current_probability_df = pd.DataFrame({
    "WeatherState": [
        "Dry",
        "Wet"
    ],
    "Observations": [
        weather_counts["Dry"],
        weather_counts["Wet"]
    ],
    "Probability": [
        dry_probability,
        wet_probability
    ]
})

current_probability_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "current_weather_probability.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SCENARIO PROBABILITIES
# ------------------------------------------------------------
#
# These scenarios allow the Bayesian decision model
# to test how strategy recommendations change as
# the probability of wet weather increases.
#
# P(Dry) + P(Wet) = 1
#
# ------------------------------------------------------------

wet_probabilities = np.arange(
    0.0,
    1.01,
    0.05
)

scenario_records = []

for wet_probability_scenario in wet_probabilities:

    dry_probability_scenario = (
        1.0 - wet_probability_scenario
    )

    scenario_records.append({
        "WetProbability": round(
            wet_probability_scenario,
            2
        ),
        "DryProbability": round(
            dry_probability_scenario,
            2
        )
    })

scenario_df = pd.DataFrame(
    scenario_records
)

# ------------------------------------------------------------
# MARK HISTORICAL BASELINE
# ------------------------------------------------------------

scenario_df["HistoricalBaseline"] = (
    np.isclose(
        scenario_df["WetProbability"],
        wet_probability,
        atol=0.025
    )
)

# ------------------------------------------------------------
# DISPLAY SCENARIOS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WEATHER PROBABILITY SCENARIOS")
print("=" * 70)

print(
    scenario_df.to_string(index=False)
)

# ------------------------------------------------------------
# SAVE SCENARIO DATA
# ------------------------------------------------------------

scenario_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weather_probability_scenarios.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# TEAM-WISE HISTORICAL WEATHER PROBABILITY
# ------------------------------------------------------------

team_records = []

for team in TEAMS:

    team_df = game_df[
        game_df["Team"] == team
    ]

    team_counts = (
        team_df["WeatherState"]
        .value_counts()
        .reindex(
            ["Dry", "Wet"],
            fill_value=0
        )
    )

    team_total = team_counts.sum()

    if team_total > 0:

        team_dry_probability = (
            team_counts["Dry"] /
            team_total
        )

        team_wet_probability = (
            team_counts["Wet"] /
            team_total
        )

    else:

        team_dry_probability = np.nan
        team_wet_probability = np.nan

    team_records.append({
        "Team": team,
        "DryObservations": team_counts["Dry"],
        "WetObservations": team_counts["Wet"],
        "TotalObservations": team_total,
        "DryProbability": team_dry_probability,
        "WetProbability": team_wet_probability
    })

team_probability_df = pd.DataFrame(
    team_records
)

# ------------------------------------------------------------
# SAVE TEAM PROBABILITIES
# ------------------------------------------------------------

team_probability_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "team_weather_probability.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# PLOT WEATHER PROBABILITY SCENARIOS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    scenario_df["WetProbability"],
    scenario_df["DryProbability"],
    marker="o",
    label="P(Dry)"
)

plt.plot(
    scenario_df["WetProbability"],
    scenario_df["WetProbability"],
    marker="o",
    label="P(Wet)"
)

plt.axvline(
    wet_probability,
    linestyle="--",
    label="Historical Wet Probability"
)

plt.xlabel(
    "Assumed Wet Weather Probability"
)

plt.ylabel(
    "Weather Probability"
)

plt.title(
    "Weather Probability Scenarios"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "weather_probability_distribution.png"
    ),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# SUMMARY FILE
# ------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "weather_probability_summary.txt"
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
        "STEP 2 - WEATHER PROBABILITY ESTIMATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Purpose:\n"
    )

    file.write(
        "Estimate the historical Dry/Wet probability from "
        "the 2025 dataset and create probability scenarios "
        "for Bayesian strategy decision-making.\n\n"
    )

    file.write(
        "Selected Teams:\n"
    )

    for team in TEAMS:
        file.write(
            f"- {team}\n"
        )

    file.write(
        "\nHistorical Weather Probability:\n"
    )

    file.write(
        f"P(Dry) = {dry_probability:.6f}\n"
    )

    file.write(
        f"P(Wet) = {wet_probability:.6f}\n"
    )

    file.write(
        "\nProbability Model:\n"
    )

    file.write(
        "P(Dry) + P(Wet) = 1\n"
    )

    file.write(
        "\nScenario Range:\n"
    )

    file.write(
        "Wet probability is varied from 0% to 100% "
        "in 5% increments.\n"
    )

    file.write(
        "\nImportant Interpretation:\n"
    )

    file.write(
        "The historical probability represents the empirical "
        "weather distribution in the selected 2025 data. "
        "The scenario probabilities are used for decision "
        "sensitivity analysis and do not represent a live "
        "weather forecast.\n"
    )

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 COMPLETED")
print("=" * 70)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print(" - current_weather_probability.csv")
print(" - weather_probability_scenarios.csv")
print(" - team_weather_probability.csv")
print(" - weather_probability_distribution.png")
print(" - weather_probability_summary.txt")

print("\n" + "=" * 70)
print("HISTORICAL BASELINE")
print("=" * 70)

print(
    f"P(Dry) = {dry_probability:.6f}"
)

print(
    f"P(Wet) = {wet_probability:.6f}"
)