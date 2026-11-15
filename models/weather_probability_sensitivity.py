import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

# ============================================================
# SETTINGS
# ============================================================

players = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]

strategies = [
    "SOFT_Short",
    "HARD_Short",
    "MEDIUM_Short",
    "SOFT_Medium"
]

probabilities = np.arange(
    0.00,
    1.01,
    0.05
)


# ============================================================
# LOAD WEATHER-SPECIFIC PAYOFFS
# ============================================================

def load_payoffs(player1, player2):

    folder = (
        f"{player1}_vs_{player2}"
        .replace(" ", "_")
    )

    pair_dir = os.path.join(
        OUTPUT_DIR,
        folder
    )

    u1_dry = pd.read_csv(
        os.path.join(pair_dir, "U1_dry.csv"),
        index_col=0
    )

    u2_dry = pd.read_csv(
        os.path.join(pair_dir, "U2_dry.csv"),
        index_col=0
    )

    u1_wet = pd.read_csv(
        os.path.join(pair_dir, "U1_wet.csv"),
        index_col=0
    )

    u2_wet = pd.read_csv(
        os.path.join(pair_dir, "U2_wet.csv"),
        index_col=0
    )

    for matrix in [
        u1_dry,
        u2_dry,
        u1_wet,
        u2_wet
    ]:
        matrix[:] = matrix.apply(
            pd.to_numeric,
            errors="coerce"
        )

    return (
        u1_dry,
        u2_dry,
        u1_wet,
        u2_wet
    )


# ============================================================
# FIND BEST RESPONSE
# ============================================================

def best_strategy(
    dry_matrix,
    wet_matrix,
    opponent_strategy,
    probability_wet
):

    expected_payoffs = {}

    for strategy in strategies:

        dry = dry_matrix.loc[
            strategy,
            opponent_strategy
        ]

        wet = wet_matrix.loc[
            strategy,
            opponent_strategy
        ]

        if pd.isna(dry) or pd.isna(wet):
            continue

        expected = (
            (1 - probability_wet) * dry
            +
            probability_wet * wet
        )

        expected_payoffs[strategy] = expected

    if not expected_payoffs:
        return None, np.nan

    best = max(
        expected_payoffs,
        key=expected_payoffs.get
    )

    return (
        best,
        expected_payoffs[best]
    )


# ============================================================
# ANALYSIS
# ============================================================

results = []

print("=" * 75)
print("WEATHER PROBABILITY SENSITIVITY ANALYSIS")
print("=" * 75)

for player1, player2 in [
    ("Red Bull Racing", "McLaren"),
    ("Red Bull Racing", "Ferrari"),
    ("Red Bull Racing", "Mercedes"),
    ("McLaren", "Ferrari"),
    ("McLaren", "Mercedes"),
    ("Ferrari", "Mercedes")
]:

    (
        u1_dry,
        u2_dry,
        u1_wet,
        u2_wet
    ) = load_payoffs(
        player1,
        player2
    )

    print("\n" + "=" * 75)
    print(
        f"{player1} vs {player2}"
    )
    print("=" * 75)

    # Use HARD_Short as the opponent reference strategy
    opponent_strategy = "HARD_Short"

    print(
        f"\nOpponent strategy fixed at: "
        f"{opponent_strategy}"
    )

    print(
        "\nWet Probability | "
        f"{player1} Best Strategy | "
        "Expected Payoff"
    )

    print("-" * 75)

    for p_wet in probabilities:

        strategy, payoff = best_strategy(
            u1_dry,
            u1_wet,
            opponent_strategy,
            p_wet
        )

        print(
            f"{p_wet:14.2f} | "
            f"{str(strategy):25s} | "
            f"{payoff:.4f}"
            if strategy is not None
            else
            f"{p_wet:14.2f} | N/A"
        )

        results.append({
            "Player1": player1,
            "Player2": player2,
            "WetProbability": p_wet,
            "DryProbability": 1 - p_wet,
            "OpponentStrategy": opponent_strategy,
            "BestStrategy": strategy,
            "ExpectedPayoff": payoff
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "weather_probability_sensitivity.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("WEATHER PROBABILITY SENSITIVITY COMPLETED")
print("=" * 75)

print(
    "\nOutput:"
)

print(
    output_file
)

print("\nDone.")