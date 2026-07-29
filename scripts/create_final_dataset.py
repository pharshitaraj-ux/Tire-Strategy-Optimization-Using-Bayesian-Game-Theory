import os
import pandas as pd
import fastf1

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "f1_2025_dataset.csv")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "weather")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "f1_2025_final_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

fastf1.Cache.enable_cache(CACHE_DIR)

# -----------------------------------------------------
# Load lap dataset
# -----------------------------------------------------
laps = pd.read_csv(RAW_FILE)

laps["LapStartTime"] = pd.to_timedelta(laps["LapStartTime"])

# -----------------------------------------------------
# Add Circuit & Country
# -----------------------------------------------------
schedule = fastf1.get_event_schedule(2025)

location = {}

for _, event in schedule.iterrows():

    if "Testing" in event["EventName"]:
        continue

    location[event["EventName"]] = {
        "Circuit": event["Location"],
        "Country": event["Country"]
    }

laps["Circuit"] = laps["Race"].map(lambda x: location[x]["Circuit"])
laps["Country"] = laps["Race"].map(lambda x: location[x]["Country"])

# -----------------------------------------------------
# Merge weather race by race
# -----------------------------------------------------
final_data = []

for race in laps["Race"].unique():

    print(f"Merging {race}")

    weather_file = os.path.join(
        WEATHER_DIR,
        race.replace(" ", "_") + "_weather.csv"
    )

    if not os.path.exists(weather_file):
        print("Weather file missing:", race)
        continue

    weather = pd.read_csv(weather_file)

    weather["Time"] = pd.to_timedelta(weather["Time"])

    weather = weather.sort_values("Time")

    race_laps = laps[laps["Race"] == race].copy()

    race_laps = race_laps.sort_values("LapStartTime")

    merged = pd.merge_asof(
        race_laps,
        weather[
            [
                "Time",
                "AirTemp",
                "TrackTemp",
                "Humidity",
                "Pressure",
                "Rainfall",
                "WindSpeed",
            ]
        ],
        left_on="LapStartTime",
        right_on="Time",
        direction="nearest",
    )

    if "Time_y" in merged.columns:
        merged.drop(columns=["Time_y"], inplace=True)

    if "Time_x" in merged.columns:
        merged.rename(columns={"Time_x": "WeatherTime"}, inplace=True)
    elif "Time" in merged.columns:
        merged.rename(columns={"Time": "WeatherTime"}, inplace=True)

    final_data.append(merged)
# -----------------------------------------------------
# Combine all races
# -----------------------------------------------------
print("Number of merged races:", len(final_data))
final_df = pd.concat(final_data, ignore_index=True)

# -----------------------------------------------------
# Save
# -----------------------------------------------------
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("Final Dataset Created Successfully")
print(final_df.shape)
print("Saved to:")
print(OUTPUT_FILE)
print("===================================")