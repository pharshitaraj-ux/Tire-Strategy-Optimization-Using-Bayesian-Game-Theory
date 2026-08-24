import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# TYRE DEGRADATION & STRATEGY ANALYSIS
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
    "tyre_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("TYRE DEGRADATION & STRATEGY ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# CLEAN REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Team",
    "Compound",
    "TyreLife",
    "LapTime",
    "TyreWear",
    "Strategy",
    "WeatherState",
    "PayoffScore"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"],
    errors="coerce"
)

df["LapTime"] = pd.to_numeric(
    df["LapTime"],
    errors="coerce"
)

df["TyreWear"] = pd.to_numeric(
    df["TyreWear"],
    errors="coerce"
)

df["PayoffScore"] = pd.to_numeric(
    df["PayoffScore"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Compound",
        "TyreLife",
        "LapTime"
    ]
)

# Focus on meaningful race compounds
compounds = [
    "SOFT",
    "MEDIUM",
    "HARD",
    "INTERMEDIATE"
]

df_compound = df[
    df["Compound"].isin(compounds)
].copy()

# ============================================================
# 1. TYRE LIFE vs LAP TIME
# ============================================================

plt.figure(figsize=(10, 6))

for compound in compounds:

    subset = df_compound[
        df_compound["Compound"] == compound
    ]

    if len(subset) == 0:
        continue

    # Reduce noise using tyre-life groups
    grouped = (
        subset.groupby("TyreLife")["LapTime"]
        .mean()
        .reset_index()
    )

    plt.plot(
        grouped["TyreLife"],
        grouped["LapTime"],
        marker="o",
        markersize=3,
        label=compound
    )

plt.title(
    "Tyre Degradation: Tyre Life vs Average Lap Time"
)

plt.xlabel("Tyre Life (laps)")
plt.ylabel("Average Lap Time (seconds)")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tyre_degradation_by_compound.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 2. AVERAGE LAP TIME BY COMPOUND AND STINT LENGTH
# ============================================================

df_compound["StintCategory"] = pd.cut(
    df_compound["TyreLife"],
    bins=[0, 10, 20, 40, np.inf],
    labels=[
        "Short",
        "Medium",
        "Long",
        "Very Long"
    ]
)

stint_analysis = (
    df_compound
    .groupby(
        ["Compound", "StintCategory"],
        observed=True
    )["LapTime"]
    .agg(
        AverageLapTime="mean",
        MedianLapTime="median",
        LapCount="count"
    )
    .reset_index()
)

stint_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "compound_stint_performance.csv"
    ),
    index=False
)

# ============================================================
# 3. TYRE WEAR BY COMPOUND
# ============================================================

wear_analysis = (
    df_compound
    .groupby("Compound")
    .agg(
        AverageTyreWear=("TyreWear", "mean"),
        MaximumTyreWear=("TyreWear", "max"),
        AverageTyreLife=("TyreLife", "mean"),
        AverageLapTime=("LapTime", "mean"),
        LapCount=("LapTime", "count")
    )
    .reset_index()
)

wear_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tyre_wear_summary.csv"
    ),
    index=False
)

# ============================================================
# 4. TYRE WEAR COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    wear_analysis["Compound"],
    wear_analysis["AverageTyreWear"]
)

plt.title(
    "Average Tyre Wear by Compound"
)

plt.xlabel("Tyre Compound")
plt.ylabel("Average Tyre Wear")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_tyre_wear.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 5. STRATEGY PERFORMANCE
# ============================================================

strategy_analysis = (
    df.groupby(
        ["Team", "Strategy"]
    )
    .agg(
        AveragePayoff=("PayoffScore", "mean"),
        AverageLapTime=("LapTime", "mean"),
        AverageTyreLife=("TyreLife", "mean"),
        AverageTyreWear=("TyreWear", "mean"),
        Observations=("Strategy", "count")
    )
    .reset_index()
)

strategy_analysis = strategy_analysis.sort_values(
    "AveragePayoff",
    ascending=False
)

strategy_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_degradation_performance.csv"
    ),
    index=False
)

# ============================================================
# 6. TOP STRATEGIES
# ============================================================

top_strategies = (
    strategy_analysis
    .groupby("Strategy")
    .agg(
        AveragePayoff=("AveragePayoff", "mean"),
        AverageLapTime=("AverageLapTime", "mean"),
        AverageTyreLife=("AverageTyreLife", "mean"),
        AverageTyreWear=("AverageTyreWear", "mean")
    )
    .sort_values(
        "AveragePayoff",
        ascending=False
    )
    .reset_index()
)

top_strategies.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "strategy_degradation_ranking.csv"
    ),
    index=False
)

# ============================================================
# 7. PAYOFF vs TYRE WEAR
# ============================================================

plt.figure(figsize=(10, 6))

for compound in compounds:

    subset = df_compound[
        df_compound["Compound"] == compound
    ]

    if len(subset) == 0:
        continue

    grouped = (
        subset.groupby("TyreWear")["PayoffScore"]
        .mean()
        .reset_index()
        .sort_values("TyreWear")
    )

    plt.plot(
        grouped["TyreWear"],
        grouped["PayoffScore"],
        marker="o",
        markersize=3,
        label=compound
    )

plt.title(
    "Strategy Payoff vs Tyre Wear"
)

plt.xlabel("Tyre Wear")
plt.ylabel("Average Payoff Score")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "payoff_vs_tyre_wear.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 8. WEATHER-SPECIFIC STRATEGY PERFORMANCE
# ============================================================

weather_strategy = (
    df.groupby(
        ["WeatherState", "Strategy"]
    )
    .agg(
        AveragePayoff=("PayoffScore", "mean"),
        AverageLapTime=("LapTime", "mean"),
        AverageTyreWear=("TyreWear", "mean"),
        Observations=("Strategy", "count")
    )
    .reset_index()
)

weather_strategy.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "weather_strategy_performance.csv"
    ),
    index=False
)

# ============================================================
# 9. SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "tyre_analysis_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("=" * 60 + "\n")
    f.write("TYRE DEGRADATION & STRATEGY ANALYSIS SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        "Purpose:\n"
        "Analyse the relationship between tyre life, tyre wear, "
        "lap time and strategy payoff.\n\n"
    )

    f.write(
        "Compound Performance:\n"
    )

    f.write(
        wear_analysis.to_string(index=False)
    )

    f.write("\n\nTop Strategies by Average Payoff:\n")

    f.write(
        top_strategies.head(10).to_string(index=False)
    )

    f.write("\n\nWeather-Specific Strategy Performance:\n")

    f.write(
        weather_strategy.to_string(index=False)
    )

print("\n" + "=" * 60)
print("TYRE DEGRADATION ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutput Directory:")
print(OUTPUT_DIR)

print("\nGenerated Files:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", file)

print("\nTop Strategies:")
print(
    top_strategies.head(10).to_string(index=False)
)