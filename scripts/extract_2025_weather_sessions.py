import os
import pandas as pd
import fastf1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEDULE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "weather",
    "f1_2025_event_schedule.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "weather",
    "f1_2025_weather_official.csv"
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

schedule = pd.read_csv(SCHEDULE_FILE)

all_weather = []

for _, event in schedule.iterrows():

    round_number = int(event["RoundNumber"])
    event_name = event["EventName"]
    country = event["Country"]
    location = event["Location"]
    event_date = event["EventDate"]

    print(
        f"Loading Round {round_number}: "
        f"{event_name} ({location})"
    )

    try:
        session = fastf1.get_session(
            2025,
            round_number,
            "R"
        )

        session.load(
            laps=False,
            telemetry=False,
            weather=True,
            messages=False
        )

        weather = session.weather_data.copy()

        if weather.empty:
            print("  No weather data found")
            continue

        weather["Year"] = 2025
        weather["RoundNumber"] = round_number
        weather["EventName"] = event_name
        weather["Country"] = country
        weather["Location"] = location
        weather["Session"] = "Race"
        weather["EventDate"] = event_date

        all_weather.append(weather)

        print(
            f"  Weather observations: {len(weather)}"
        )

    except Exception as e:
        print(
            f"  ERROR: {event_name}: {e}"
        )

if all_weather:

    final_weather = pd.concat(
        all_weather,
        ignore_index=True
    )

    output_columns = [
        "Year",
        "RoundNumber",
        "EventName",
        "Country",
        "Location",
        "Session",
        "EventDate",
        "Time",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed"
    ]

    existing_columns = [
        col for col in output_columns
        if col in final_weather.columns
    ]

    final_weather = final_weather[
        existing_columns
    ]

    final_weather.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nWeather extraction completed.")
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        f"\nDataset shape: "
        f"{final_weather.shape}"
    )

    print("\nFirst rows:")
    print(final_weather.head())

else:

    print(
        "\nNo weather data was extracted."
    )