import os
import fastf1
import pandas as pd
# -------------------------------
# Create folders
# -------------------------------
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
# Enable FastF1 cache
fastf1.Cache.enable_cache("data/cache")
# Seasons to download
SEASONS = [2025, 2026]
all_data = []
for year in SEASONS:
    print(f"\nDownloading {year} schedule...")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Cannot load season {year}: {e}")
        continue
    for _, event in schedule.iterrows():
        race = event["EventName"]
        print(f"\nLoading {race} ({year})")
        try:
            session = fastf1.get_session(year, race, "R")
            session.load()

            laps = session.laps.copy()

            laps["Year"] = year
            laps["Race"] = race

            all_data.append(laps)

            print(f"Collected {len(laps)} laps")

        except Exception as e:
            print(f"Skipped {race}: {e}")

# -------------------------------
# Merge all races
# -------------------------------

dataset = pd.concat(all_data, ignore_index=True)

dataset.to_csv(
    "data/raw/f1_2025_2026_raw.csv",
    index=False
)

print("\n===================================")
print("Dataset Saved Successfully!")
print(dataset.shape)
print("===================================")