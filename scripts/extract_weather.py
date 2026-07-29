import os
import fastf1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "weather")

os.makedirs(OUTPUT_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

schedule = fastf1.get_event_schedule(2025)

for _, event in schedule.iterrows():

    if "Testing" in event["EventName"]:
        continue

    print(f"Loading {event['EventName']}")

    session = fastf1.get_session(
        2025,
        event["EventName"],
        "R"
    )

    session.load(weather=True, laps=False, telemetry=False)

    weather = session.weather_data

    filename = event["EventName"].replace(" ", "_") + "_weather.csv"

    weather.to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False
    )

print("Done")