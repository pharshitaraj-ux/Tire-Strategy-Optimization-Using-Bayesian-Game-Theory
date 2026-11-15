import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

F1_DATA = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_dataset.csv"
)

WEATHER_DATA = os.path.join(
    BASE_DIR,
    "data",
    "weather",
    "f1_2025_weather_official.csv"
)

SCHEDULE_DATA = os.path.join(
    BASE_DIR,
    "data",
    "weather",
    "f1_2025_event_schedule.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "f1_2025_weather_merged.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("F1 2025 WEATHER MERGE")
print("=" * 70)

f1 = pd.read_csv(F1_DATA)
weather = pd.read_csv(WEATHER_DATA)
schedule = pd.read_csv(SCHEDULE_DATA)

print(f"\nF1 dataset shape: {f1.shape}")
print(f"Weather dataset shape: {weather.shape}")
print(f"Schedule shape: {schedule.shape}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_f1 = [
    "Race",
    "Year",
    "LapStartTime",
    "LapStartDate"
]

required_weather = [
    "Year",
    "RoundNumber",
    "EventName",
    "EventDate",
    "Time",
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindSpeed"
]

required_schedule = [
    "RoundNumber",
    "EventName",
    "Country",
    "Location",
    "EventDate"
]

for col in required_f1:
    if col not in f1.columns:
        raise ValueError(f"Missing F1 column: {col}")

for col in required_weather:
    if col not in weather.columns:
        raise ValueError(f"Missing weather column: {col}")

for col in required_schedule:
    if col not in schedule.columns:
        raise ValueError(f"Missing schedule column: {col}")

print("\nRequired columns validated successfully.")


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

f1["Year"] = pd.to_numeric(
    f1["Year"],
    errors="coerce"
)

weather["Year"] = pd.to_numeric(
    weather["Year"],
    errors="coerce"
)

f1["LapStartTime"] = pd.to_timedelta(
    f1["LapStartTime"],
    errors="coerce"
)

weather["Time"] = pd.to_timedelta(
    weather["Time"],
    errors="coerce"
)

f1["LapStartDate"] = pd.to_datetime(
    f1["LapStartDate"],
    errors="coerce"
)

schedule["EventDate"] = pd.to_datetime(
    schedule["EventDate"],
    errors="coerce"
)

weather["EventDate"] = pd.to_datetime(
    weather["EventDate"],
    errors="coerce"
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print("\nMissing F1 LapStartTime:",
      f1["LapStartTime"].isna().sum())

print("Missing F1 LapStartDate:",
      f1["LapStartDate"].isna().sum())

print("Missing weather Time:",
      weather["Time"].isna().sum())


# ============================================================
# CHECK RACE COVERAGE
# ============================================================

f1_races = set(f1["Race"].dropna().unique())
schedule_races = set(schedule["EventName"].dropna().unique())

print("\nUnique races in F1 dataset:",
      len(f1_races))

print("Unique races in official schedule:",
      len(schedule_races))

f1_not_schedule = sorted(
    f1_races - schedule_races
)

schedule_not_f1 = sorted(
    schedule_races - f1_races
)

print(
    "\nF1 races not found in schedule:",
    f1_not_schedule if f1_not_schedule else "None"
)

print(
    "Schedule races not found in F1 dataset:",
    schedule_not_f1 if schedule_not_f1 else "None"
)


# ============================================================
# PREPARE SCHEDULE MAPPING
# ============================================================

schedule_map = schedule[
    [
        "RoundNumber",
        "EventName",
        "Country",
        "Location",
        "EventDate"
    ]
].copy()

schedule_map = schedule_map.rename(
    columns={
        "EventName": "Race"
    }
)


# ============================================================
# MERGE F1 WITH OFFICIAL EVENT SCHEDULE
# ============================================================

f1 = f1.merge(
    schedule_map,
    on="Race",
    how="left"
)

print(
    "\nRows without official event match:",
    f1["EventDate"].isna().sum()
)


# ============================================================
# PREPARE WEATHER DATA
# ============================================================

weather_race = weather.rename(
    columns={
        "RoundNumber": "WeatherRoundNumber",
        "EventName": "WeatherEventName",
        "Location": "WeatherLocation",
        "Session": "WeatherSession",
        "EventDate": "WeatherEventDate",
        "Time": "WeatherTime"
    }
).copy()


weather_columns = [
    "WeatherRoundNumber",
    "WeatherEventName",
    "WeatherLocation",
    "WeatherSession",
    "WeatherEventDate",
    "WeatherTime",
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindSpeed"
]

weather_race = weather_race[weather_columns].copy()


# ============================================================
# RACE-BY-RACE WEATHER MATCHING
# ============================================================

merged_races = []

round_numbers = sorted(
    f1["RoundNumber"].dropna().unique()
)

print("\n" + "=" * 70)
print("RACE-WISE WEATHER MATCHING")
print("=" * 70)

for round_number in round_numbers:

    f1_race = f1[
        f1["RoundNumber"] == round_number
    ].copy()

    weather_race_data = weather_race[
        weather_race["WeatherRoundNumber"] == round_number
    ].copy()

    if f1_race.empty:
        continue

    if weather_race_data.empty:
        print(
            f"Round {round_number}: "
            f"No weather data found"
        )

        f1_race["WeatherMatched"] = False
        merged_races.append(f1_race)

        continue

    # Sort before merge_asof
    f1_race = f1_race.sort_values(
        "LapStartTime"
    )

    weather_race_data = weather_race_data.sort_values(
        "WeatherTime"
    )

    # --------------------------------------------------------
    # Match each F1 lap to nearest weather observation
    # within 60 seconds.
    # --------------------------------------------------------

    matched = pd.merge_asof(
        f1_race,
        weather_race_data,
        left_on="LapStartTime",
        right_on="WeatherTime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=60)
    )

    # --------------------------------------------------------
    # Calculate time difference
    # --------------------------------------------------------

    matched["WeatherTimeDifference"] = (
        matched["LapStartTime"]
        - matched["WeatherTime"]
    ).abs()

    matched["WeatherMatched"] = (
        matched["WeatherTimeDifference"]
        <= pd.Timedelta(seconds=60)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT treat calendar-date mismatch as failure.
    #
    # Las Vegas crosses midnight, so race laps can occur
    # on the next calendar day while still belonging to the
    # same official race event.
    # --------------------------------------------------------

    matched["RaceDateValid"] = (
        (
            matched["LapStartDate"].dt.date
            == matched["EventDate"].dt.date
        )
        |
        (
            matched["Race"]
            == "Las Vegas Grand Prix"
        )
    )

    matched["DateMatch"] = (
        matched["LapStartDate"].dt.date
        == matched["EventDate"].dt.date
    )

    merged_races.append(matched)

    # --------------------------------------------------------
    # Race report
    # --------------------------------------------------------

    match_percentage = (
        matched["WeatherMatched"].mean() * 100
    )

    date_valid_percentage = (
        matched["RaceDateValid"].mean() * 100
    )

    print(
        f"Round {int(round_number):2d} | "
        f"{matched['Race'].iloc[0]:30s} | "
        f"Rows: {len(matched):5d} | "
        f"Weather matched: {match_percentage:6.2f}% | "
        f"Race date valid: {date_valid_percentage:6.2f}%"
    )


# ============================================================
# COMBINE ALL RACES
# ============================================================

final_df = pd.concat(
    merged_races,
    ignore_index=True
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

weather_match_percentage = (
    final_df["WeatherMatched"].mean() * 100
)

race_date_valid_percentage = (
    final_df["RaceDateValid"].mean() * 100
)

print(
    f"\nFinal dataset shape: {final_df.shape}"
)

print(
    f"Weather matched: "
    f"{final_df['WeatherMatched'].sum()} / "
    f"{len(final_df)}"
)

print(
    f"Weather match percentage: "
    f"{weather_match_percentage:.2f}%"
)

print(
    f"Race date validation: "
    f"{race_date_valid_percentage:.2f}%"
)


# ============================================================
# WEATHER MISSING VALUES
# ============================================================

weather_check_columns = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed"
]

print("\nWeather missing values:")

for col in weather_check_columns:

    missing = final_df[col].isna().sum()

    print(
        f"{col:15s}: {missing}"
    )


# ============================================================
# LAS VEGAS VALIDATION
# ============================================================

las_vegas = final_df[
    final_df["Race"] == "Las Vegas Grand Prix"
].copy()

if not las_vegas.empty:

    print("\n" + "=" * 70)
    print("LAS VEGAS SPECIAL VALIDATION")
    print("=" * 70)

    print(
        "Las Vegas rows:",
        len(las_vegas)
    )

    print(
        "Calendar DateMatch:",
        las_vegas["DateMatch"].value_counts().to_dict()
    )

    print(
        "RaceDateValid:",
        las_vegas["RaceDateValid"].value_counts().to_dict()
    )

    print(
        "WeatherMatched:",
        las_vegas["WeatherMatched"].value_counts().to_dict()
    )


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nFinal dataset shape: {final_df.shape}"
)

print("\nValidation summary:")
print(
    f"  Weather temporal matching : "
    f"{weather_match_percentage:.2f}%"
)

print(
    f"  Race date validation      : "
    f"{race_date_valid_percentage:.2f}%"
)

print(
    f"  Weather missing values    : "
    f"{final_df[weather_check_columns].isna().sum().sum()}"
)

print("\nDone.")