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


# ============================================================
# FIND PURE NASH EQUILIBRIA
# ============================================================

def find_nash_equilibria(u1, u2):

    equilibria = []

    for s1 in strategies:
        for s2 in strategies:

            p1_payoff = u1.loc[s1, s2]
            p2_payoff = u2.loc[s1, s2]

            if pd.isna(p1_payoff) or pd.isna(p2_payoff):
                continue

            # Player 1 best response to Player 2 strategy
            column = u1[s2].dropna()

            if len(column) == 0:
                continue

            p1_best = p1_payoff >= column.max()

            # Player 2 best response to Player 1 strategy
            row = u2.loc[s1].dropna()

            if len(row) == 0:
                continue

            p2_best = p2_payoff >= row.max()

            if p1_best and p2_best:

                equilibria.append(
                    (
                        s1,
                        s2,
                        p1_payoff,
                        p2_payoff
                    )
                )

    return equilibria


# ============================================================
# PROCESS ALL PAIRS
# ============================================================

pairs = []

for i in range(len(players)):
    for j in range(i + 1, len(players)):
        pairs.append((players[i], players[j]))

print("\n")
print("=" * 75)
print("BAYESIAN NASH EQUILIBRIUM ANALYSIS")


for player1, player2 in pairs:

    folder = (
        f"{player1}_vs_{player2}"
        .replace(" ", "_")
    )

    pair_dir = os.path.join(
        OUTPUT_DIR,
        folder
    )

    u1_file = os.path.join(
        pair_dir,
        "U1_bayesian_expected.csv"
    )

    u2_file = os.path.join(
        pair_dir,
        "U2_bayesian_expected.csv"
    )

    if not os.path.exists(u1_file):
        print(f"\nMissing: {u1_file}")
        continue

    u1 = pd.read_csv(
        u1_file,
        index_col=0
    )

    u2 = pd.read_csv(
        u2_file,
        index_col=0
    )

    u1 = u1.apply(
        pd.to_numeric,
        errors="coerce"
    )

    u2 = u2.apply(
        pd.to_numeric,
        errors="coerce"
    )

    equilibria = find_nash_equilibria(
        u1,
        u2
    )

    print("\n" + "=" * 75)
    print(f"PLAYER 1: {player1}")
    print(f"PLAYER 2: {player2}")
    #print("=" * 75)

    if len(equilibria) == 0:

        print("\nNo pure-strategy Nash equilibrium found.")

    else:

        print("\nPURE-STRATEGY NASH EQUILIBRIA")
        print("-" * 75)

        for eq in equilibria:

            s1, s2, payoff1, payoff2 = eq

            print(
                f"\n{player1} → {s1}"
            )

            print(
                f"{player2} → {s2}"
            )

            print(
                f"Payoffs = ({payoff1:.4f}, {payoff2:.4f})"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_file = os.path.join(
        pair_dir,
        "bayesian_nash_equilibrium.txt"
    )

    with open(result_file, "w", encoding="utf-8") as f:

        f.write(
            "BAYESIAN NASH EQUILIBRIUM ANALYSIS\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Player 1: {player1}\n"
        )

        f.write(
            f"Player 2: {player2}\n\n"
        )

        if len(equilibria) == 0:

            f.write(
                "No pure-strategy Nash equilibrium found.\n"
            )

        else:

            f.write(
                "PURE-STRATEGY NASH EQUILIBRIA\n\n"
            )

            for eq in equilibria:

                s1, s2, payoff1, payoff2 = eq

                f.write(
                    f"{player1} → {s1}\n"
                )

                f.write(
                    f"{player2} → {s2}\n"
                )

                f.write(
                    f"Payoffs = "
                    f"({payoff1:.4f}, {payoff2:.4f})\n\n"
                )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 75)
print("BAYESIAN NASH EQUILIBRIUM ANALYSIS COMPLETED")
print("=" * 75)