import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game",
    "optimal_strategy.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load Results
# --------------------------------------------------

df = pd.read_csv(DATA)

# --------------------------------------------------
# Team Ranking
# --------------------------------------------------

ranking = df.sort_values(
    by="ExpectedPayoff",
    ascending=False
)

ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_ranking.csv"
    ),
    index=False
)

# --------------------------------------------------
# Team Performance
# --------------------------------------------------

performance = df.copy()

performance["Rank"] = (
    performance["ExpectedPayoff"]
    .rank(
        ascending=False,
        method="dense"
    )
    .astype(int)
)

performance = performance.sort_values("Rank")

performance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "team_performance.csv"
    ),
    index=False
)

# --------------------------------------------------
# Plot 1
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    ranking["Team"],
    ranking["ExpectedPayoff"]
)

plt.xticks(rotation=20)

plt.ylabel("Expected Payoff")

plt.title("Average Expected Payoff by Team")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_expected_payoff.png"
    )
)

plt.close()

# --------------------------------------------------
# Plot 2
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    performance["Team"],
    performance["Rank"]
)

plt.xticks(rotation=20)

plt.ylabel("Rank")

plt.title("Bayesian Strategy Ranking")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "team_strategy_comparison.png"
    )
)

plt.close()

# --------------------------------------------------
# Summary Report
# --------------------------------------------------

summary = os.path.join(
    OUTPUT_DIR,
    "strategy_evaluation.txt"
)

best_team = ranking.iloc[0]

with open(summary, "w") as f:

    f.write("BAYESIAN STRATEGY EVALUATION\n\n")

    f.write("Overall Best Team\n")
    f.write("-----------------------\n")
    f.write(best_team["Team"] + "\n\n")

    f.write("Recommended Strategy\n")
    f.write("-----------------------\n")
    f.write(best_team["OptimalStrategy"] + "\n\n")

    f.write("Expected Payoff\n")
    f.write("-----------------------\n")
    f.write(str(round(best_team["ExpectedPayoff"],4)))

    f.write("\n\n")

    f.write("Complete Ranking\n")
    f.write("-----------------------\n")
    f.write(ranking.to_string(index=False))

print("="*60)
print("STRATEGY EVALUATION COMPLETED")
print("="*60)

print("\nRanking")
print(ranking)

print("\nResults saved to")
print(OUTPUT_DIR)