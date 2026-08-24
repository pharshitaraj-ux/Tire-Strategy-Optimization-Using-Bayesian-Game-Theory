import os
import pandas as pd
import numpy as np

# ============================================================
# STRATEGIC & OPPONENT INTERACTION ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game",
    "opponent_interaction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SELECT FOUR TEAMS
# ============================================================

TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

# Strategies used in our payoff-game analysis
STRATEGIES = [
    "SOFT_Short",
    "HARD_Short",
    "MEDIUM_Short",
    "SOFT_Medium"
]

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("STRATEGIC & OPPONENT INTERACTION ANALYSIS")
print("=" * 65)

print("\nOriginal Dataset Shape:")
print(df.shape)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "Race",
    "Team",
    "Strategy",
    "PayoffScore",
    "WeatherState"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# FILTER FOUR TEAMS
# ============================================================

df = df[df["Team"].isin(TEAMS)].copy()

df = df[
    df["Strategy"].isin(STRATEGIES)
].copy()

df = df.dropna(
    subset=[
        "Race",
        "Team",
        "Strategy",
        "PayoffScore"
    ]
)

print("\nFiltered Dataset Shape:")
print(df.shape)

# ============================================================
# STEP 1
# DETERMINE PRIMARY STRATEGY PER TEAM PER RACE
# ============================================================
#
# A team can have many lap-level strategy records.
# We therefore identify the strategy appearing most frequently
# for that team in each race.
# ============================================================

strategy_counts = (
    df.groupby(
        ["Race", "Team", "Strategy"]
    )
    .size()
    .reset_index(name="LapCount")
)

strategy_counts = strategy_counts.sort_values(
    ["Race", "Team", "LapCount"],
    ascending=[True, True, False]
)

primary_strategy = (
    strategy_counts
    .drop_duplicates(
        subset=["Race", "Team"]
    )
    .rename(
        columns={
            "Strategy": "PrimaryStrategy"
        }
    )
)

# ============================================================
# STEP 2
# CALCULATE TEAM-LEVEL RACE PAYOFF
# ============================================================

race_payoff = (
    df.groupby(
        ["Race", "Team"]
    )
    .agg(
        AveragePayoff=("PayoffScore", "mean"),
        AverageLapTime=("LapTime", "mean"),
        Observations=("PayoffScore", "count")
    )
    .reset_index()
)

race_strategy = primary_strategy.merge(
    race_payoff,
    on=["Race", "Team"],
    how="inner"
)

# ============================================================
# STEP 3
# ADD WEATHER STATE
# ============================================================

weather = (
    df.groupby(
        ["Race", "Team"]
    )["WeatherState"]
    .agg(
        lambda x: x.mode().iloc[0]
        if not x.mode().empty
        else x.iloc[0]
    )
    .reset_index()
)

race_strategy = race_strategy.merge(
    weather,
    on=["Race", "Team"],
    how="left"
)

# ============================================================
# SAVE TEAM RACE STRATEGIES
# ============================================================

race_strategy.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "race_team_strategies.csv"
    ),
    index=False
)

# ============================================================
# STEP 4
# CREATE OPPONENT STRATEGY INFORMATION
# ============================================================

records = []

for race in race_strategy["Race"].unique():

    race_data = race_strategy[
        race_strategy["Race"] == race
    ]

    for focal_team in TEAMS:

        focal = race_data[
            race_data["Team"] == focal_team
        ]

        if focal.empty:
            continue

        focal = focal.iloc[0]

        for opponent in TEAMS:

            if opponent == focal_team:
                continue

            opp = race_data[
                race_data["Team"] == opponent
            ]

            if opp.empty:
                continue

            opp = opp.iloc[0]

            records.append({
                "Race": race,
                "WeatherState": focal["WeatherState"],
                "FocalTeam": focal_team,
                "FocalStrategy": focal["PrimaryStrategy"],
                "OpponentTeam": opponent,
                "OpponentStrategy": opp["PrimaryStrategy"],
                "FocalPayoff": focal["AveragePayoff"],
                "FocalLapTime": focal["AverageLapTime"]
            })

interaction_df = pd.DataFrame(records)

# ============================================================
# SAVE RAW INTERACTION DATA
# ============================================================

interaction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "team_opponent_interactions.csv"
    ),
    index=False
)

# ============================================================
# STEP 5
# OPPONENT-CONDITIONAL PAYOFF
# ============================================================

opponent_payoff = (
    interaction_df
    .groupby(
        [
            "FocalTeam",
            "OpponentTeam",
            "FocalStrategy",
            "OpponentStrategy"
        ]
    )
    .agg(
        AveragePayoff=("FocalPayoff", "mean"),
        AverageLapTime=("FocalLapTime", "mean"),
        Observations=("FocalPayoff", "count")
    )
    .reset_index()
)

opponent_payoff = opponent_payoff.sort_values(
    [
        "FocalTeam",
        "OpponentTeam",
        "AveragePayoff"
    ],
    ascending=[True, True, False]
)

opponent_payoff.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "opponent_conditional_payoffs.csv"
    ),
    index=False
)

# ============================================================
# STEP 6
# BEST RESPONSE TO EACH OPPONENT STRATEGY
# ============================================================

best_response_records = []

for focal_team in TEAMS:

    focal_data = opponent_payoff[
        opponent_payoff["FocalTeam"] == focal_team
    ]

    for opponent in TEAMS:

        if opponent == focal_team:
            continue

        opponent_data = focal_data[
            focal_data["OpponentTeam"] == opponent
        ]

        if opponent_data.empty:
            continue

        for opponent_strategy in (
            opponent_data["OpponentStrategy"].unique()
        ):

            condition = opponent_data[
                opponent_data["OpponentStrategy"]
                == opponent_strategy
            ]

            if condition.empty:
                continue

            best = condition.loc[
                condition["AveragePayoff"].idxmax()
            ]

            best_response_records.append({
                "FocalTeam": focal_team,
                "OpponentTeam": opponent,
                "OpponentStrategy": opponent_strategy,
                "BestResponse": best["FocalStrategy"],
                "ExpectedPayoff": best["AveragePayoff"],
                "Observations": best["Observations"]
            })

best_response = pd.DataFrame(
    best_response_records
)

best_response.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "best_response_analysis.csv"
    ),
    index=False
)

# ============================================================
# STEP 7
# WEATHER-SPECIFIC INTERACTION
# ============================================================

weather_interaction = (
    interaction_df
    .groupby(
        [
            "WeatherState",
            "FocalTeam",
            "OpponentTeam",
            "FocalStrategy",
            "OpponentStrategy"
        ]
    )
    .agg(
        AveragePayoff=("FocalPayoff", "mean"),
        AverageLapTime=("FocalLapTime", "mean"),
        Observations=("FocalPayoff", "count")
    )
    .reset_index()
)

weather_interaction.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weather_opponent_interaction.csv"
    ),
    index=False
)

# ============================================================
# STEP 8
# STRATEGY SWITCHING ANALYSIS
# ============================================================

switching_records = []

for focal_team in TEAMS:

    team_data = best_response[
        best_response["FocalTeam"] == focal_team
    ]

    if team_data.empty:
        continue

    strategy_counts = (
        team_data["BestResponse"]
        .value_counts()
    )

    for strategy, count in strategy_counts.items():

        switching_records.append({
            "FocalTeam": focal_team,
            "BestResponseStrategy": strategy,
            "NumberOfOpponentConditions": count
        })

switching = pd.DataFrame(
    switching_records
)

switching.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_switching_summary.csv"
    ),
    index=False
)

# ============================================================
# STEP 9
# SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "opponent_interaction_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 65 + "\n")
    f.write("STRATEGIC & OPPONENT INTERACTION SUMMARY\n")
    f.write("=" * 65 + "\n\n")

    f.write(
        "Teams analysed:\n"
    )

    for team in TEAMS:
        f.write(f"- {team}\n")

    f.write("\nStrategies analysed:\n")

    for strategy in STRATEGIES:
        f.write(f"- {strategy}\n")

    f.write(
        "\n\nBest Response Analysis:\n\n"
    )

    f.write(
        best_response.to_string(index=False)
    )

    f.write(
        "\n\n\nStrategy Switching Summary:\n\n"
    )

    f.write(
        switching.to_string(index=False)
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("STRATEGIC INTERACTION ANALYSIS COMPLETED")
print("=" * 65)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(" -", file)

print("\nBest Response Analysis:")

if not best_response.empty:
    print(
        best_response.to_string(
            index=False
        )
    )
else:
    print(
        "No sufficient opponent combinations were found."
    )