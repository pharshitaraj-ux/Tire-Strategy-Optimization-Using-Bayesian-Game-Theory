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
    "03_weather_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# -------------------------------------------------
# Weather Distributions
# -------------------------------------------------

weather_cols = [
    ("AirTemp", "Air Temperature (°C)"),
    ("TrackTemp", "Track Temperature (°C)"),
    ("Humidity", "Humidity (%)"),
    ("Pressure", "Pressure (hPa)"),
    ("WindSpeed", "Wind Speed (m/s)")
]

for col, title in weather_cols:

    plt.figure(figsize=(8,5))

    plt.hist(df[col], bins=30)

    plt.title(title + " Distribution")

    plt.xlabel(title)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{col.lower()}_distribution.png"
        )
    )

    plt.close()

# -------------------------------------------------
# Rainfall Distribution
# -------------------------------------------------

plt.figure(figsize=(6,4))

df["Rainfall"].value_counts().sort_index().plot(kind="bar")

plt.title("Rainfall Distribution")

plt.xlabel("Rainfall")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rainfall_distribution.png"
    )
)

plt.close()

# -------------------------------------------------
# Air Temperature vs Lap Time
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["AirTemp"],
    df["LapTime"],
    alpha=0.3,
    s=10
)

plt.xlabel("Air Temperature (°C)")

plt.ylabel("Lap Time (sec)")

plt.title("Air Temperature vs Lap Time")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "airtemp_vs_laptime.png"
    )
)

plt.close()

# -------------------------------------------------
# Track Temperature vs Lap Time
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    df["TrackTemp"],
    df["LapTime"],
    alpha=0.3,
    s=10
)

plt.xlabel("Track Temperature (°C)")

plt.ylabel("Lap Time (sec)")

plt.title("Track Temperature vs Lap Time")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tracktemp_vs_laptime.png"
    )
)

plt.close()

# -------------------------------------------------
# Rainfall vs Lap Time
# -------------------------------------------------

plt.figure(figsize=(6,5))

df.boxplot(
    column="LapTime",
    by="Rainfall"
)

plt.title("Lap Time under Rainfall")

plt.suptitle("")

plt.xlabel("Rainfall")

plt.ylabel("Lap Time (sec)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rainfall_vs_laptime.png"
    )
)

plt.close()

# -------------------------------------------------
# Weather Correlation
# -------------------------------------------------

weather = df[
    [
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "LapTime"
    ]
]

corr = weather.corr()

plt.figure(figsize=(8,6))

plt.imshow(corr, cmap="coolwarm", aspect="auto")

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Weather Correlation Matrix")

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "weather_correlation.png"
    )
)

plt.close()

print("="*60)
print("EDA PART 3 COMPLETED")
print("="*60)

print("\nWeather Correlation Matrix")

print(corr)