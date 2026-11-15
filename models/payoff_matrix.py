import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "feature_engineering",
    "f1_2025_feature_engineered.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "bayesian_game"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BAYESIAN GAME THEORY - PAYOFF MATRICES")
print("=" * 70)

df = pd.read_csv(DATA)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# SELECT PLAYERS
# ============================================================

players = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes"
]


# ============================================================
# SELECT STRATEGIES
# ============================================================

strategies = [
    "SOFT_Short",
    "HARD_Short",
    "MEDIUM_Short",
    "SOFT_Medium"
]


# ============================================================
# FILTER DATA
# ============================================================

df = df[
    df["Team"].isin(players)
].copy()

df = df[
    df["Strategy"].isin(strategies)
].copy()

print("\nSelected teams:")
print(players)

print("\nSelected strategies:")
print(strategies)

print(
    "\nFiltered dataset shape:",
    df.shape
)


# ============================================================
# RACE-LEVEL WEATHER STATE
# ============================================================
# Calculate the Bayesian weather prior BEFORE filtering
# to selected teams and strategies.
#
# 2025 season:
# 24 total races
# 21 dry races
# 3 wet races
#
# P(Dry) = 21 / 24 = 0.875
# P(Wet) = 3 / 24 = 0.125
# ============================================================

FULL_DATA = pd.read_csv(
    DATA
)

race_weather = (
    FULL_DATA.groupby("Race")["WeatherState"]
    .apply(
        lambda x:
        "Wet"
        if (x == "Wet").any()
        else "Dry"
    )
    .reset_index()
)

total_races = race_weather["Race"].nunique()

dry_races = (
    race_weather["WeatherState"] == "Dry"
).sum()

wet_races = (
    race_weather["WeatherState"] == "Wet"
).sum()

# Use the full 2025 race-level Bayesian prior
p_dry = 21 / 24
p_wet = 3 / 24

print("\n" + "=" * 70)
print("BAYESIAN WEATHER PRIOR")
print("=" * 70)

print(f"\nTotal races : {total_races}")
print(f"Dry races   : {dry_races}")
print(f"Wet races   : {wet_races}")

print(f"P(Dry)      : {p_dry:.4f}")
print(f"P(Wet)      : {p_wet:.4f}")


# Add race weather state to dataset

df = df.merge(
    race_weather,
    on="Race",
    how="left",
    suffixes=("", "_Race")
)

df["GameWeatherState"] = df[
    "WeatherState_Race"
]


# ============================================================
# FUNCTION:
# GET PAYOFF FOR A PLAYER STRATEGY
# ============================================================

def get_player_payoff(
    race_data,
    team,
    strategy
):

    data = race_data[
        (race_data["Team"] == team)
        &
        (race_data["Strategy"] == strategy)
    ]

    if data.empty:
        return np.nan

    return data["PayoffScore"].mean()


# ============================================================
# FUNCTION:
# CREATE PAIRWISE BIMATRIX
# ============================================================

def create_pairwise_matrix(
    player1,
    player2,
    weather_state=None
):

    if weather_state is not None:

        game_data = df[
            df["GameWeatherState"]
            == weather_state
        ].copy()

    else:

        game_data = df.copy()

    matrix = pd.DataFrame(
        index=strategies,
        columns=strategies,
        dtype=object
    )

    for s1 in strategies:

        for s2 in strategies:

            races_p1 = set(
                game_data[
                    (game_data["Team"] == player1)
                    &
                    (game_data["Strategy"] == s1)
                ]["Race"]
            )

            races_p2 = set(
                game_data[
                    (game_data["Team"] == player2)
                    &
                    (game_data["Strategy"] == s2)
                ]["Race"]
            )

            common_races = (
                races_p1
                &
                races_p2
            )

            if len(common_races) == 0:

                matrix.loc[
                    s1,
                    s2
                ] = "N/A"

                continue

            pair_data = game_data[
                game_data["Race"]
                .isin(common_races)
            ]

            u1_data = pair_data[
                (pair_data["Team"] == player1)
                &
                (pair_data["Strategy"] == s1)
            ]

            u2_data = pair_data[
                (pair_data["Team"] == player2)
                &
                (pair_data["Strategy"] == s2)
            ]

            if u1_data.empty or u2_data.empty:

                matrix.loc[
                    s1,
                    s2
                ] = "N/A"

                continue

            u1 = u1_data[
                "PayoffScore"
            ].mean()

            u2 = u2_data[
                "PayoffScore"
            ].mean()

            matrix.loc[
                s1,
                s2
            ] = (
                f"({u1:.4f}, {u2:.4f})"
            )

    return matrix


# ============================================================
# FUNCTION:
# CREATE NUMERIC BIMATRIX
# ============================================================

def create_numeric_matrices(
    player1,
    player2,
    weather_state=None
):

    if weather_state is not None:

        game_data = df[
            df["GameWeatherState"]
            == weather_state
        ].copy()

    else:

        game_data = df.copy()

    u1_matrix = pd.DataFrame(
        index=strategies,
        columns=strategies,
        dtype=float
    )

    u2_matrix = pd.DataFrame(
        index=strategies,
        columns=strategies,
        dtype=float
    )

    for s1 in strategies:

        for s2 in strategies:

            races_p1 = set(
                game_data[
                    (game_data["Team"] == player1)
                    &
                    (game_data["Strategy"] == s1)
                ]["Race"]
            )

            races_p2 = set(
                game_data[
                    (game_data["Team"] == player2)
                    &
                    (game_data["Strategy"] == s2)
                ]["Race"]
            )

            common_races = (
                races_p1
                &
                races_p2
            )

            if len(common_races) == 0:

                u1_matrix.loc[
                    s1,
                    s2
                ] = np.nan

                u2_matrix.loc[
                    s1,
                    s2
                ] = np.nan

                continue

            pair_data = game_data[
                game_data["Race"]
                .isin(common_races)
            ]

            p1_data = pair_data[
                (pair_data["Team"] == player1)
                &
                (pair_data["Strategy"] == s1)
            ]

            p2_data = pair_data[
                (pair_data["Team"] == player2)
                &
                (pair_data["Strategy"] == s2)
            ]

            if p1_data.empty or p2_data.empty:

                u1_matrix.loc[
                    s1,
                    s2
                ] = np.nan

                u2_matrix.loc[
                    s1,
                    s2
                ] = np.nan

                continue

            u1_matrix.loc[
                s1,
                s2
            ] = p1_data[
                "PayoffScore"
            ].mean()

            u2_matrix.loc[
                s1,
                s2
            ] = p2_data[
                "PayoffScore"
            ].mean()

    return u1_matrix, u2_matrix


# ============================================================
# FUNCTION:
# BAYESIAN EXPECTED PAYOFF MATRIX
# ============================================================

def create_expected_matrix(
    u1_dry,
    u2_dry,
    u1_wet,
    u2_wet
):

    expected_u1 = (
        p_dry * u1_dry
        +
        p_wet * u1_wet
    )

    expected_u2 = (
        p_dry * u2_dry
        +
        p_wet * u2_wet
    )

    return (
        expected_u1,
        expected_u2
    )


# ============================================================
# CREATE MATRICES FOR ALL PLAYER PAIRS
# ============================================================

all_players = []

for i in range(
    len(players)
):

    for j in range(
        i + 1,
        len(players)
    ):

        all_players.append(
            (
                players[i],
                players[j]
            )
        )


print(
    "\n" + "=" * 70
)

print(
    "CREATING PAIRWISE BIMATRICES"
)

print(
    "=" * 70
)


# ============================================================
# PROCESS EACH PAIR
# ============================================================

for player1, player2 in all_players:

    pair_name = (
        f"{player1}_vs_{player2}"
    )

    folder_name = (
        pair_name
        .replace(" ", "_")
    )

    pair_dir = os.path.join(
        OUTPUT_DIR,
        folder_name
    )

    os.makedirs(
        pair_dir,
        exist_ok=True
    )

    print(
        f"\nCreating: "
        f"{player1} vs {player2}"
    )


    # ========================================================
    # DRY WEATHER
    # ========================================================

    u1_dry, u2_dry = (
        create_numeric_matrices(
            player1,
            player2,
            "Dry"
        )
    )

    dry_bimatrix = (
        u1_dry.astype(object)
    )

    for s1 in strategies:

        for s2 in strategies:

            value1 = u1_dry.loc[
                s1,
                s2
            ]

            value2 = u2_dry.loc[
                s1,
                s2
            ]

            if (
                pd.isna(value1)
                or
                pd.isna(value2)
            ):

                dry_bimatrix.loc[
                    s1,
                    s2
                ] = "N/A"

            else:

                dry_bimatrix.loc[
                    s1,
                    s2
                ] = (
                    f"({value1:.4f}, "
                    f"{value2:.4f})"
                )


    # ========================================================
    # WET WEATHER
    # ========================================================

    u1_wet, u2_wet = (
        create_numeric_matrices(
            player1,
            player2,
            "Wet"
        )
    )

    wet_bimatrix = (
        u1_wet.astype(object)
    )

    for s1 in strategies:

        for s2 in strategies:

            value1 = u1_wet.loc[
                s1,
                s2
            ]

            value2 = u2_wet.loc[
                s1,
                s2
            ]

            if (
                pd.isna(value1)
                or
                pd.isna(value2)
            ):

                wet_bimatrix.loc[
                    s1,
                    s2
                ] = "N/A"

            else:

                wet_bimatrix.loc[
                    s1,
                    s2
                ] = (
                    f"({value1:.4f}, "
                    f"{value2:.4f})"
                )


    # ========================================================
    # BAYESIAN EXPECTED PAYOFF
    # ========================================================

    expected_u1, expected_u2 = (
        create_expected_matrix(
            u1_dry,
            u2_dry,
            u1_wet,
            u2_wet
        )
    )

    expected_bimatrix = (
        expected_u1.astype(object)
    )

    for s1 in strategies:

        for s2 in strategies:

            value1 = expected_u1.loc[
                s1,
                s2
            ]

            value2 = expected_u2.loc[
                s1,
                s2
            ]

            if (
                pd.isna(value1)
                or
                pd.isna(value2)
            ):

                expected_bimatrix.loc[
                    s1,
                    s2
                ] = "N/A"

            else:

                expected_bimatrix.loc[
                    s1,
                    s2
                ] = (
                    f"({value1:.4f}, "
                    f"{value2:.4f})"
                )


    # ========================================================
    # SAVE TEXTBOOK BIMATRICES
    # ========================================================

    dry_file = os.path.join(
        pair_dir,
        "payoff_matrix_dry.csv"
    )

    wet_file = os.path.join(
        pair_dir,
        "payoff_matrix_wet.csv"
    )

    expected_file = os.path.join(
        pair_dir,
        "payoff_matrix_bayesian_expected.csv"
    )

    dry_bimatrix.to_csv(
        dry_file
    )

    wet_bimatrix.to_csv(
        wet_file
    )

    expected_bimatrix.to_csv(
        expected_file
    )


    # ========================================================
    # SAVE NUMERIC MATRICES
    # ========================================================

    u1_dry.to_csv(
        os.path.join(
            pair_dir,
            "U1_dry.csv"
        )
    )

    u2_dry.to_csv(
        os.path.join(
            pair_dir,
            "U2_dry.csv"
        )
    )

    u1_wet.to_csv(
        os.path.join(
            pair_dir,
            "U1_wet.csv"
        )
    )

    u2_wet.to_csv(
        os.path.join(
            pair_dir,
            "U2_wet.csv"
        )
    )

    expected_u1.to_csv(
        os.path.join(
            pair_dir,
            "U1_bayesian_expected.csv"
        )
    )

    expected_u2.to_csv(
        os.path.join(
            pair_dir,
            "U2_bayesian_expected.csv"
        )
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_file = os.path.join(
        pair_dir,
        "summary.txt"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "BAYESIAN BIMATRIX PAYOFF SUMMARY\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Player 1: {player1}\n"
        )

        f.write(
            f"Player 2: {player2}\n\n"
        )

        f.write(
            "Player 1 strategies = rows\n"
        )

        f.write(
            "Player 2 strategies = columns\n\n"
        )

        f.write(
            "Each cell = (U1, U2)\n\n"
        )

        f.write(
            f"P(Dry) = {p_dry:.4f}\n"
        )

        f.write(
            f"P(Wet) = {p_wet:.4f}\n\n"
        )

        f.write(
            "DRY WEATHER BIMATRIX\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            dry_bimatrix.to_string()
        )

        f.write(
            "\n\nWET WEATHER BIMATRIX\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            wet_bimatrix.to_string()
        )

        f.write(
            "\n\nBAYESIAN EXPECTED PAYOFF BIMATRIX\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            expected_bimatrix.to_string()
        )


    # ========================================================
    # DISPLAY - TEXTBOOK STYLE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"PLAYER 1: {player1}"
    )

    print(
        f"PLAYER 2: {player2}"
    )

    print("=" * 70)

    print()

    # Player 2 = Columns
    print(
        f"                 PLAYER 2 ({player2}) → STRATEGIES"
    )

    print(
        "              "
        + "   ".join(
            f"{strategy:14s}"
            for strategy in strategies
        )
    )

    # Player 1 = Rows
    print(
        "PLAYER 1"
    )

    print(
        f"({player1})"
    )

    print(
        "STRATEGIES"
    )

    print(
        "     ↓"
    )

    print()

    # Matrix values
    for row_strategy in strategies:

        row_values = []

        for col_strategy in strategies:

            value = expected_bimatrix.loc[
                row_strategy,
                col_strategy
            ]

            row_values.append(
                str(value)
            )

        print(
            f"{row_strategy:16s}"
            +
            "   ".join(
                f"{value:18s}"
                for value in row_values
            )
        )

    print()

    # Cell explanation
    print(
        "Cell = (U1, U2)"
    )

    print(
        f"U1 = {player1} payoff"
    )

    print(
        f"U2 = {player2} payoff"
    )

    print(
        "=" * 70
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PAYOFF MATRICES CREATED SUCCESSFULLY"
)

print(
    "=" * 70
)

print("\nPlayers:")

for p in players:

    print(
        f"  - {p}"
    )

print("\nStrategies:")

for s in strategies:

    print(
        f"  - {s}"
    )

print("\nPlayer pairs created:")

for p1, p2 in all_players:

    print(
        f"  - {p1} vs {p2}"
    )

print(
    "\nBayesian prior:"
)

print(
    f"  P(Dry) = {p_dry:.4f}"
)

print(
    f"  P(Wet) = {p_wet:.4f}"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print("\nDone.")