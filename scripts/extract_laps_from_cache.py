import os
import fastf1
import pandas as pd

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Enable Cache
# -----------------------------
fastf1.Cache.enable_cache(CACHE_DIR)

# -----------------------------
# Load 2025 Schedule
# -----------------------------
schedule = fastf1.get_event_schedule(2025)

all_laps = []

for _, event in schedule.iterrows():

    # Skip testing events
    if "Testing" in event["EventName"]:
        continue

    race = event["EventName"]

    print(f"\nLoading {race}")

    try:

        session = fastf1.get_session(2025, race, "R")

        # Uses existing cache instead of downloading again
        session.load()

        laps = session.laps.copy()

        laps["Race"] = race
        laps["Year"] = 2025

        all_laps.append(laps)

        print(f"Collected {len(laps)} laps")

    except Exception as e:

        print(f"Skipped {race}")
        print(e)

# -----------------------------
# Merge
# -----------------------------
dataset = pd.concat(all_laps, ignore_index=True)

dataset.to_csv(
    os.path.join(OUTPUT_DIR, "f1_2025_dataset.csv"),
    index=False
)

print("\n===================================")
print("Dataset Created Successfully")
print(dataset.shape)
print("===================================")