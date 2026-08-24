import os
import pandas as pd
import numpy as np

# ============================================================
# REAL-TIME BAYESIAN STRATEGY
# STEP 5 - INTERACTIVE REAL-TIME STRATEGY INPUT
# ============================================================

print("REAL-TIME BAYESIAN STRATEGY")
print("INTERACTIVE REAL-TIME STRATEGY INPUT")


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

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "01_intermediate_payoff",
    "extended_payoff_data.csv"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "realtime_bayesian_strategy",
    "05_realtime_strategy"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD PAYOFF DATA
# ------------------------------------------------------------

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"Required input file not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print("\nHistorical payoff data loaded.")
print(f"Records: {len(df)}")

# ------------------------------------------------------------
# AVAILABLE TEAMS
# ------------------------------------------------------------

TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

# ------------------------------------------------------------
# AVAILABLE STRATEGIES
# ------------------------------------------------------------

STRATEGIES = [
    "SOFT_Short",
    "MEDIUM_Short",
    "HARD_Short",
    "INTERMEDIATE_Medium"
]

# ------------------------------------------------------------
# INPUT VALIDATION FUNCTION
# ------------------------------------------------------------

def get_team(prompt):

    while True:

        print("\nAvailable Teams:")
        for i, team in enumerate(TEAMS, start=1):
            print(f"{i}. {team}")

        value = input(f"\n{prompt}: ").strip()

        if value in TEAMS:
            return value

        try:
            index = int(value)

            if 1 <= index <= len(TEAMS):
                return TEAMS[index - 1]

        except ValueError:
            pass

        print("\nInvalid team. Please enter the team name or number.")


def get_probability():

    while True:

        value = input(
            "\nEnter current probability of wet weather (%): "
        ).strip()

        try:

            probability = float(value)

            if 0 <= probability <= 100:
                return probability / 100

        except ValueError:
            pass

        print(
            "Invalid probability. Enter a value between 0 and 100."
        )


# ============================================================
# INTERACTIVE INPUT
# ============================================================

print("CURRENT RACE SCENARIO INPUT")

team_a = get_team("Select Team A")

team_b = get_team("Select Opponent Team")

while team_b == team_a:

    print(
        "\nTeam A and Opponent Team must be different."
    )

    team_b = get_team("Select Opponent Team")

wet_probability = get_probability()

dry_probability = 1 - wet_probability

# ============================================================
# DISPLAY SCENARIO
# ============================================================

print("\n" + "=" * 70)
print("CURRENT REAL-TIME SCENARIO")
print("=" * 70)

print(f"Team A              : {team_a}")
print(f"Opponent Team       : {team_b}")
print(f"Wet Weather         : {wet_probability:.2%}")
print(f"Dry Weather         : {dry_probability:.2%}")

# ============================================================
# EXPECTED PAYOFF FUNCTION
# ============================================================

def calculate_team_strategy(team):

    team_results = []

    for strategy in STRATEGIES:

        dry_rows = df[
            (df["Team"] == team) &
            (df["Strategy"] == strategy) &
            (df["WeatherState"] == "Dry")
        ]

        wet_rows = df[
            (df["Team"] == team) &
            (df["Strategy"] == strategy) &
            (df["WeatherState"] == "Wet")
        ]

        # ----------------------------------------------------
        # Historical payoff
        # ----------------------------------------------------

        dry_payoff = (
            dry_rows["AveragePayoff"].iloc[0]
            if len(dry_rows) > 0
            and pd.notna(dry_rows["AveragePayoff"].iloc[0])
            else np.nan
        )

        wet_payoff = (
            wet_rows["AveragePayoff"].iloc[0]
            if len(wet_rows) > 0
            and pd.notna(wet_rows["AveragePayoff"].iloc[0])
            else np.nan
        )

        # ----------------------------------------------------
        # Expected payoff
        # ----------------------------------------------------

        if pd.notna(dry_payoff) and pd.notna(wet_payoff):

            expected_payoff = (
                dry_probability * dry_payoff
                +
                wet_probability * wet_payoff
            )

        elif pd.notna(dry_payoff):

            # No observed wet payoff.
            # Do NOT invent a value.
            #
            # Strategy cannot be evaluated under both states.

            expected_payoff = np.nan

        elif pd.notna(wet_payoff):

            expected_payoff = np.nan

        else:

            expected_payoff = np.nan

        team_results.append({
            "Team": team,
            "Strategy": strategy,
            "DryProbability": dry_probability,
            "WetProbability": wet_probability,
            "DryPayoff": dry_payoff,
            "WetPayoff": wet_payoff,
            "ExpectedPayoff": expected_payoff
        })

    return pd.DataFrame(team_results)


# ============================================================
# CALCULATE TEAM A
# ============================================================

team_a_results = calculate_team_strategy(team_a)

# ============================================================
# CALCULATE TEAM B
# ============================================================

team_b_results = calculate_team_strategy(team_b)

# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [team_a_results, team_b_results],
    ignore_index=True
)

# ============================================================
# GET RECOMMENDATIONS
# ============================================================

def get_recommendation(results):

    valid = results[
        results["ExpectedPayoff"].notna()
    ].copy()

    if valid.empty:

        return None

    best_row = valid.loc[
        valid["ExpectedPayoff"].idxmax()
    ]

    return best_row


team_a_best = get_recommendation(team_a_results)
team_b_best = get_recommendation(team_b_results)

# ============================================================
# DISPLAY TEAM A ANALYSIS
# ============================================================


print(f"TEAM A - {team_a}")

display_a = team_a_results[
    [
        "Strategy",
        "DryPayoff",
        "WetPayoff",
        "ExpectedPayoff"
    ]
].copy()

print(
    display_a.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# DISPLAY TEAM B ANALYSIS
# ============================================================

print(f"OPPONENT - {team_b}")

display_b = team_b_results[
    [
        "Strategy",
        "DryPayoff",
        "WetPayoff",
        "ExpectedPayoff"
    ]
].copy()

print(
    display_b.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("REAL-TIME STRATEGY RECOMMENDATION")
print("=" * 70)

if team_a_best is not None:

    print(
        f"\n{team_a} → "
        f"{team_a_best['Strategy']}"
    )

    print(
        f"Expected Payoff = "
        f"{team_a_best['ExpectedPayoff']:.6f}"
    )

else:

    print(
        f"\n{team_a}: "
        "No strategy can currently be evaluated."
    )


if team_b_best is not None:

    print(
        f"\n{team_b} → "
        f"{team_b_best['Strategy']}"
    )

    print(
        f"Expected Payoff = "
        f"{team_b_best['ExpectedPayoff']:.6f}"
    )

else:

    print(
        f"\n{team_b}: "
        "No strategy can currently be evaluated."
    )

# ============================================================
# SAVE RESULTS
# ============================================================

all_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "realtime_strategy_analysis.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# Save scenario
# ------------------------------------------------------------

scenario = pd.DataFrame([{
    "TeamA": team_a,
    "TeamB": team_b,
    "WetProbability": wet_probability,
    "DryProbability": dry_probability,
    "TeamARecommendation":
        team_a_best["Strategy"]
        if team_a_best is not None
        else "Unavailable",
    "TeamAExpectedPayoff":
        team_a_best["ExpectedPayoff"]
        if team_a_best is not None
        else np.nan,
    "TeamBRecommendation":
        team_b_best["Strategy"]
        if team_b_best is not None
        else "Unavailable",
    "TeamBExpectedPayoff":
        team_b_best["ExpectedPayoff"]
        if team_b_best is not None
        else np.nan
}])

scenario.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "realtime_scenario_result.csv"
    ),
    index=False
)

# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "realtime_strategy_summary.txt"
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
        "STEP 5 - INTERACTIVE REAL-TIME STRATEGY INPUT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Team A: {team_a}\n"
    )

    file.write(
        f"Opponent Team: {team_b}\n"
    )

    file.write(
        f"Wet Probability: {wet_probability:.6f}\n"
    )

    file.write(
        f"Dry Probability: {dry_probability:.6f}\n\n"
    )

    file.write(
        "TEAM A RECOMMENDATION\n"
    )

    file.write("-" * 70 + "\n")

    if team_a_best is not None:

        file.write(
            f"Strategy: {team_a_best['Strategy']}\n"
        )

        file.write(
            f"Expected Payoff: "
            f"{team_a_best['ExpectedPayoff']:.6f}\n"
        )

    else:

        file.write(
            "No strategy available.\n"
        )

    file.write(
        "\nOPPONENT RECOMMENDATION\n"
    )

    file.write("-" * 70 + "\n")

    if team_b_best is not None:

        file.write(
            f"Strategy: {team_b_best['Strategy']}\n"
        )

        file.write(
            f"Expected Payoff: "
            f"{team_b_best['ExpectedPayoff']:.6f}\n"
        )

    else:

        file.write(
            "No strategy available.\n"
        )

    file.write(
        "\nExpected Payoff Equation:\n"
    )

    file.write(
        "EU(s) = P(Dry)*U(s,Dry) + P(Wet)*U(s,Wet)\n"
    )

    file.write(
        "\nNOTE:\n"
    )

    file.write(
        "The system uses observed historical 2025 payoff "
        "values. Missing weather-strategy combinations "
        "are not assigned synthetic payoff values.\n"
    )


print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
print(" - realtime_strategy_analysis.csv")
print(" - realtime_scenario_result.csv")
print(" - realtime_strategy_summary.txt")

