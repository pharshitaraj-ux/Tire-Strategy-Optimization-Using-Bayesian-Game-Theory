import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

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
    "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = pd.read_csv(DATA)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

metrics = pd.DataFrame({
    "Metric":[
        "Average Expected Payoff",
        "Maximum Expected Payoff",
        "Minimum Expected Payoff",
        "Standard Deviation"
    ],
    "Value":[
        round(df["ExpectedPayoff"].mean(),4),
        round(df["ExpectedPayoff"].max(),4),
        round(df["ExpectedPayoff"].min(),4),
        round(df["ExpectedPayoff"].std(),4)
    ]
})

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.csv"
    ),
    index=False
)

# ---------------------------------------------------
# Strategy Frequency
# ---------------------------------------------------

strategy_freq = (
    df["OptimalStrategy"]
      .value_counts()
)

# ---------------------------------------------------
# Plot 1
# ---------------------------------------------------

plt.figure(figsize=(7,5))

plt.bar(
    df["Team"],
    df["ExpectedPayoff"]
)

plt.xticks(rotation=20)

plt.ylabel("Expected Payoff")

plt.title("Expected Payoff by Team")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "payoff_distribution.png"
    )
)

plt.close()

# ---------------------------------------------------
# Plot 2
# ---------------------------------------------------

plt.figure(figsize=(6,5))

plt.bar(
    strategy_freq.index,
    strategy_freq.values
)

plt.ylabel("Frequency")

plt.title("Recommended Strategy Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "strategy_frequency.png"
    )
)

plt.close()

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

summary = os.path.join(
    OUTPUT_DIR,
    "performance_summary.txt"
)

best = df.loc[
    df["ExpectedPayoff"].idxmax()
]

with open(summary, "w") as f:

    f.write("PERFORMANCE EVALUATION\n\n")

    f.write("Average Expected Payoff : ")
    f.write(str(round(df["ExpectedPayoff"].mean(),4)))
    f.write("\n")

    f.write("Maximum Expected Payoff : ")
    f.write(str(round(df["ExpectedPayoff"].max(),4)))
    f.write("\n")

    f.write("Minimum Expected Payoff : ")
    f.write(str(round(df["ExpectedPayoff"].min(),4)))
    f.write("\n")

    f.write("Standard Deviation : ")
    f.write(str(round(df["ExpectedPayoff"].std(),4)))
    f.write("\n\n")

    f.write("Best Performing Team\n")
    f.write("----------------------\n")
    f.write(best["Team"])
    f.write("\n")

    f.write(best["OptimalStrategy"])

print("="*60)
print("PERFORMANCE EVALUATION COMPLETED")
print("="*60)

print(metrics)

print("\nBest Team")
print(best["Team"])
print(best["OptimalStrategy"])