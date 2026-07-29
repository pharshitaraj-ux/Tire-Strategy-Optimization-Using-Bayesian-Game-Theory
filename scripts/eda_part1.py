import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_preprocessed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "eda",
    "01_dataset_overview"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# -------------------------------------------------
# Save Summary Statistics
# -------------------------------------------------
summary = df.describe(include="all")

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "summary_statistics.csv"
    )
)

print("\nSummary statistics saved.")

# -------------------------------------------------
# Missing Values Plot
# -------------------------------------------------
missing = df.isnull().sum()

plt.figure(figsize=(12,5))
missing.plot(kind="bar")

plt.title("Missing Values Per Column")
plt.xlabel("Columns")
plt.ylabel("Missing Values")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "missing_values.png"
    )
)

plt.close()

print("Missing value plot saved.")

# -------------------------------------------------
# Numeric Distribution
# -------------------------------------------------
numeric = df.select_dtypes(include="number")

numeric.hist(
    figsize=(18,18),
    bins=30
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "numeric_distributions.png"
    )
)

plt.close()

print("Numeric distribution plots saved.")

print("\nEDA Part 1 Completed Successfully")