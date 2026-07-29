import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAPS_FILE = os.path.join(BASE_DIR, "data", "raw", "f1_2025_final_dataset.csv")
CSV_OUTPUT = os.path.join(BASE_DIR, "csv_output")

laps = pd.read_csv(LAPS_FILE)

weather_columns = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Rainfall",
    "WindSpeed",
    "Pressure"
]

# Add empty columns first
for col in weather_columns:
    laps[col] = None

race_folders = sorted(
    [d for d in os.listdir(CSV_OUTPUT)
     if os.path.isdir(os.path.join(CSV_OUTPUT, d))]
)

for folder in race_folders:

    weather_file = os.path.join(
        CSV_OUTPUT,
        folder,
        "weather_data.csv"
    )

    if not os.path.exists(weather_file):
        continue

    weather = pd.read_csv(weather_file)

    race_name = folder.split("_", 3)[-1].replace("_", " ")

    race_rows = laps["Race"] == race_name

    # Use average weather for that race
    for col in weather_columns:

        if col in weather.columns:

            laps.loc[race_rows, col] = weather[col].mean()

print("Weather merged successfully")

output = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_weather_dataset.csv"
)

laps.to_csv(output, index=False)

print("Saved:", output)
print(laps.shape)