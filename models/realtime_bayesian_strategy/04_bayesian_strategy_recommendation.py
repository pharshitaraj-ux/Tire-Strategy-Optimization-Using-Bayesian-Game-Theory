import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# REAL-TIME BAYESIAN STRATEGY
# STEP 4 - BAYESIAN STRATEGY RECOMMENDATION
# ============================================================

print("=" * 70)
print("REAL-TIME BAYESIAN STRATEGY")
print("STEP 4 - BAYESIAN STRATEGY RECOMMENDATION")
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
# INPUT
# ------------------------------------------------------------

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "03_expected_payoff",
    "expected_payoff_scenarios.csv"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "04_bayesian_strategy_recommendation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nInput Dataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# TEAMS
# ------------------------------------------------------------

TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Team",
    "Strategy",
    "WetProbability",
    "DryProbability",
    "ExpectedPayoff"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ------------------------------------------------------------
# FILTER TEAMS
# ------------------------------------------------------------

game_df = df[
    df["Team"].isin(TEAMS)
].copy()

# ------------------------------------------------------------
# REMOVE UNAVAILABLE PAYOFFS
# ------------------------------------------------------------

game_df = game_df.dropna(
    subset=["ExpectedPayoff"]
)

# ------------------------------------------------------------
# BEST STRATEGY FOR EVERY WEATHER PROBABILITY
# ------------------------------------------------------------

recommendation_records = []

probabilities = sorted(
    game_df["WetProbability"].unique()
)

for wet_probability in probabilities:

    scenario = game_df[
        game_df["WetProbability"] == wet_probability
    ]

    for team in TEAMS:

        team_data = scenario[
            scenario["Team"] == team
        ]

        if team_data.empty:
            continue

        best_row = team_data.loc[
            team_data["ExpectedPayoff"].idxmax()
        ]

        recommendation_records.append({
            "Team": team,
            "WetProbability": wet_probability,
            "DryProbability": best_row["DryProbability"],
            "RecommendedStrategy": best_row["Strategy"],
            "ExpectedPayoff": best_row["ExpectedPayoff"]
        })

recommendation_df = pd.DataFrame(
    recommendation_records
)

# ------------------------------------------------------------
# DISPLAY RECOMMENDATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRATEGY RECOMMENDATIONS BY WEATHER PROBABILITY")
print("=" * 70)

print(
    recommendation_df.to_string(index=False)
)

# ------------------------------------------------------------
# FIND STRATEGY SWITCH POINTS
# ------------------------------------------------------------

switch_records = []

for team in TEAMS:

    team_data = recommendation_df[
        recommendation_df["Team"] == team
    ].sort_values(
        "WetProbability"
    )

    previous_strategy = None

    for _, row in team_data.iterrows():

        current_strategy = row[
            "RecommendedStrategy"
        ]

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

# ------------------------------------------------------------
# CURRENT REAL-TIME SCENARIOS
# ------------------------------------------------------------

scenario_probabilities = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60
]

scenario_records = []

for probability in scenario_probabilities:

    available = recommendation_df[
        recommendation_df["WetProbability"] == probability
    ]

    if available.empty:
        continue

    for _, row in available.iterrows():

        scenario_records.append({
            "WetProbability": probability,
            "Team": row["Team"],
            "RecommendedStrategy": row[
                "RecommendedStrategy"
            ],
            "ExpectedPayoff": row[
                "ExpectedPayoff"
            ]
        })

scenario_df = pd.DataFrame(
    scenario_records
)

# ------------------------------------------------------------
# SAVE COMPLETE RECOMMENDATIONS
# ------------------------------------------------------------

recommendation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_recommendations_by_weather_probability.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SAVE SWITCH POINTS
# ------------------------------------------------------------

switch_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "bayesian_strategy_switch_points.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SAVE SCENARIO RECOMMENDATIONS
# ------------------------------------------------------------

scenario_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "real_time_strategy_scenarios.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# STRATEGY PROFILE AT 20% RAIN
# ------------------------------------------------------------

example_probability = 0.20

example = recommendation_df[
    recommendation_df["WetProbability"] ==
    example_probability
]

print("\n" + "=" * 70)
print("EXAMPLE REAL-TIME SCENARIO")
print("=" * 70)

print(
    f"\nIf Wet Probability = {example_probability:.0%}"
)

if not example.empty:

    print(
        example[
            [
                "Team",
                "RecommendedStrategy",
                "ExpectedPayoff"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "20% scenario not available."
    )

# ------------------------------------------------------------
# PLOT STRATEGY CHANGES
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

for team in TEAMS:

    team_data = recommendation_df[
        recommendation_df["Team"] == team
    ].sort_values(
        "WetProbability"
    )

    plt.plot(
        team_data["WetProbability"],
        team_data["ExpectedPayoff"],
        marker="o",
        label=team
    )

plt.xlabel(
    "Probability of Wet Weather"
)

plt.ylabel(
    "Expected Payoff"
)

plt.title(
    "Expected Payoff Under Changing Weather Probability"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "real_time_expected_payoff.png"
    )
)

plt.close()

# ------------------------------------------------------------
# STRATEGY SWITCH SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "bayesian_strategy_recommendation_summary.txt"
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
        "STEP 4 - BAYESIAN STRATEGY RECOMMENDATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Decision Rule:\n"
    )

    file.write(
        "For each team and weather-probability scenario, "
        "select the available strategy with the highest "
        "expected payoff.\n\n"
    )

    file.write(
        "Recommended Strategy:\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        recommendation_df.to_string(index=False)
    )

    file.write(
        "\n\nStrategy Switch Points:\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    if switch_df.empty:

        file.write(
            "No strategy switches detected.\n"
        )

    else:

        file.write(
            switch_df.to_string(index=False)
        )

    file.write(
        "\n\nInterpretation:\n"
    )

    file.write(
        "The model evaluates tyre strategies under different "
        "beliefs about future wet weather. As the probability "
        "of wet weather changes, the expected payoff of each "
        "strategy changes. The strategy with the highest "
        "expected payoff is selected as the Bayesian strategy "
        "recommendation for that scenario.\n"
    )

    file.write(
        "\nThis stage represents a Bayesian best-response "
        "recommendation based on the current payoff model. "
        "A complete multi-player Bayesian Nash equilibrium "
        "requires payoff functions explicitly dependent on "
        "the strategies selected by all players.\n"
    )

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")

print(
    " - strategy_recommendations_by_weather_probability.csv"
)

print(
    " - bayesian_strategy_switch_points.csv"
)

print(
    " - real_time_strategy_scenarios.csv"
)

print(
    " - real_time_expected_payoff.png"
)

print(
    " - bayesian_strategy_recommendation_summary.txt"
)

print("\n" + "=" * 70)
print("STRATEGY SWITCH POINTS")
print("=" * 70)

if switch_df.empty:

    print(
        "No strategy switches detected."
    )

else:

    print(
        switch_df.to_string(index=False)
    )

print("\n" + "=" * 70)