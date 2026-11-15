import os
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

DRY_PATH = os.path.join(
    "outputs",
    "bayesian_game",
    "payoff_matrix_dry.csv"
)

WET_PATH = os.path.join(
    "outputs",
    "bayesian_game",
    "payoff_matrix_wet.csv"
)

OUTPUT_DIR = os.path.join(
    "outputs",
    "advanced_analytics",
    "dynamic_strategy_recommendation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PAYOFF MATRICES
# ============================================================

print("Loading Bayesian payoff matrices...")

dry_df = pd.read_csv(DRY_PATH)
wet_df = pd.read_csv(WET_PATH)

dry_df = dry_df.set_index("Team")
wet_df = wet_df.set_index("Team")

print("\nDry-weather strategies:")
print(list(dry_df.columns))

print("\nWet-weather strategies:")
print(list(wet_df.columns))


# ============================================================
# 3. TEAM SELECTION
# ============================================================

teams = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

print("\nAvailable teams:")
for team in teams:
    print(f" - {team}")


# ============================================================
# 4. FUNCTION TO CALCULATE EXPECTED PAYOFF
# ============================================================

def calculate_expected_payoffs(team, wet_probability):
    """
    Calculate expected payoff for each strategy that has
    valid payoff values in both weather states.

    E[U(s)] = (1-p)U(s,Dry) + pU(s,Wet)
    """

    dry_probability = 1 - wet_probability

    results = []

    # Use strategies common to BOTH weather matrices.
    common_strategies = [
        strategy
        for strategy in dry_df.columns
        if strategy in wet_df.columns
    ]

    for strategy in common_strategies:

        dry_payoff = dry_df.loc[team, strategy]
        wet_payoff = wet_df.loc[team, strategy]

        expected_payoff = (
            dry_probability * dry_payoff
            + wet_probability * wet_payoff
        )

        results.append({
            "Team": team,
            "Strategy": strategy,
            "DryPayoff": dry_payoff,
            "WetPayoff": wet_payoff,
            "DryProbability": dry_probability,
            "WetProbability": wet_probability,
            "ExpectedPayoff": expected_payoff
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="ExpectedPayoff",
        ascending=False
    ).reset_index(drop=True)

    return result_df


# ============================================================
# 5. GET USER INPUT
# ============================================================

print("\n" + "=" * 60)
print("DYNAMIC FORMULA 1 STRATEGY RECOMMENDATION")
print("=" * 60)

print("\nSelect a team:")
print("1 - Red Bull Racing")
print("2 - McLaren")
print("3 - Ferrari")
print("4 - Mercedes")

team_choice = input("\nEnter team number: ").strip()

team_map = {
    "1": "Red Bull Racing",
    "2": "McLaren",
    "3": "Ferrari",
    "4": "Mercedes"
}

if team_choice not in team_map:
    raise ValueError("Invalid team selection.")

team = team_map[team_choice]


# ============================================================
# 6. WEATHER PROBABILITY
# ============================================================

wet_input = input(
    "\nEnter estimated wet-weather probability (%): "
).strip()

try:
    wet_probability = float(wet_input) / 100
except ValueError:
    raise ValueError(
        "Wet-weather probability must be a number."
    )

if not 0 <= wet_probability <= 1:
    raise ValueError(
        "Wet-weather probability must be between 0 and 100."
    )


# ============================================================
# 7. CALCULATE EXPECTED PAYOFFS
# ============================================================

results = calculate_expected_payoffs(
    team,
    wet_probability
)


# ============================================================
# 8. RECOMMENDATION
# ============================================================

best_row = results.iloc[0]

best_strategy = best_row["Strategy"]
best_payoff = best_row["ExpectedPayoff"]

if len(results) > 1:
    second_payoff = results.iloc[1]["ExpectedPayoff"]
    robustness_margin = best_payoff - second_payoff
else:
    second_payoff = None
    robustness_margin = None


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "-" * 60)
print(f"Team: {team}")
print(f"Wet-weather probability: {wet_probability * 100:.1f}%")
print(f"Dry-weather probability: {(1 - wet_probability) * 100:.1f}%")
print("-" * 60)

print("\nExpected Payoffs:")

print(
    results[
        [
            "Strategy",
            "DryPayoff",
            "WetPayoff",
            "ExpectedPayoff"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)

print(f"RECOMMENDED STRATEGY: {best_strategy}")
print(f"EXPECTED PAYOFF: {best_payoff:.4f}")

if robustness_margin is not None:
    print(
        f"ROBUSTNESS MARGIN: {robustness_margin:.4f}"
    )

print("=" * 60)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dynamic_expected_payoffs.csv"
    ),
    index=False
)

recommendation = pd.DataFrame({
    "Team": [team],
    "WetProbability": [wet_probability],
    "DryProbability": [1 - wet_probability],
    "RecommendedStrategy": [best_strategy],
    "ExpectedPayoff": [best_payoff],
    "SecondBestPayoff": [second_payoff],
    "RobustnessMargin": [robustness_margin]
})

recommendation.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_recommendation.csv"
    ),
    index=False
)


# ============================================================
# 11. SAVE TEXT SUMMARY
# ============================================================

summary = f"""
Dynamic Formula 1 Strategy Recommendation
==========================================

Team:
{team}

Wet-weather probability:
{wet_probability:.4f}

Dry-weather probability:
{1 - wet_probability:.4f}

Recommended strategy:
{best_strategy}

Expected payoff:
{best_payoff:.4f}

Second-best payoff:
{second_payoff if second_payoff is not None else "N/A"}

Robustness margin:
{robustness_margin if robustness_margin is not None else "N/A"}

The recommendation is based on:

E[U(s)] = (1-p)U(s,Dry) + pU(s,Wet)

Only strategies with valid payoff values in both
weather conditions were included.
"""

with open(
    os.path.join(
        OUTPUT_DIR,
        "dynamic_strategy_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print(" - dynamic_expected_payoffs.csv")
print(" - strategy_recommendation.csv")
print(" - dynamic_strategy_summary.txt")