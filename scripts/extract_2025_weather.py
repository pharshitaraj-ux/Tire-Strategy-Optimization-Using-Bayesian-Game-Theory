import os
import pandas as pd
import fastf1

# ============================================================
# 2025 F1 EVENT SCHEDULE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "weather"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR = 2025

print("=" * 70)
print("EXTRACTING 2025 F1 EVENT SCHEDULE")
print("=" * 70)

schedule = fastf1.get_event_schedule(
    YEAR,
    include_testing=False
)

print("\nAvailable columns:")
print(schedule.columns.tolist())

print("\nNumber of events:")
print(len(schedule))

# Save the official schedule
output_file = os.path.join(
    OUTPUT_DIR,
    "f1_2025_event_schedule.csv"
)

schedule.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\nFirst events:")
print(
    schedule[
        [
            "RoundNumber",
            "EventName",
            "Country",
            "Location",
            "EventDate"
        ]
    ].head(10).to_string(index=False)
)
