
from pathlib import Path

import numpy as np
import pandas as pd


SEASON = 2026

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}_{SEASON}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    return pd.read_csv(path)


def build_features() -> pd.DataFrame:
    # ---------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------
    race = load_csv("race_results")
    quali = load_csv("qualifying_results")
    practice = load_csv("practice_pace")

    for df in [race, quali, practice]:
        df["RoundNumber"] = pd.to_numeric(
            df["RoundNumber"], errors="raise"
        ).astype(int)

        df["Abbreviation"] = (
            df["Abbreviation"].str.upper()
            if "Abbreviation" in df.columns
            else None
        )

    # ---------------------------------------------
    # 2. Create the target table
    # ---------------------------------------------
    race["Position"] = pd.to_numeric(
        race["Position"], errors="coerce"
    )

    race["is_winner"] = (
        race["Position"] == 1
    ).astype("int8")

    base_columns = [
        "Season",
        "RoundNumber",
        "EventName",
        "EventDate",
        "Abbreviation",
        "DriverId",
        "TeamName",
        "Status",
        "Position",
        "is_winner",
    ]

    base = race[base_columns].copy()

    base = base.rename(
        columns={
            "Season": "season",
            "RoundNumber": "round_number",
            "EventName": "event_name",
            "EventDate": "event_date",
            "Abbreviation": "driver_code",
            "DriverId": "driver_id",
            "TeamName": "team_name",
            "Status": "result_status",
            "Position": "finish_position",
        }
    )

    # ---------------------------------------------
    # 3. Driver's historical race performance
    # ---------------------------------------------
    base = base.sort_values(
        ["driver_id", "round_number"]
    ).reset_index(drop=True)

    driver_group = base.groupby("driver_id", sort=False)

    base["prev_finish_position"] = (
        driver_group["finish_position"].shift(1)
    )

    # Shift first, then calculate rolling means.
    # This excludes the current race result.
    base["rolling_finish_avg_3"] = (
        base.groupby("driver_id", sort=False)[
            "finish_position"
        ]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3, min_periods=1
            ).mean()
        )
    )

    base["prev_win_count_3"] = (
        base.groupby("driver_id", sort=False)["is_winner"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3, min_periods=1
            ).sum()
        )
    )

    # ---------------------------------------------
    # 4. Qualifying history
    # ---------------------------------------------
    quali["Position"] = pd.to_numeric(
        quali["Position"], errors="coerce"
    )

    quali["quali_position"] = quali["Position"]

    # Convert qualifying segment durations to seconds.
    # Keep the final qualifying position as the primary
    # qualifying feature in this first version.
    for col in ["Q1", "Q2", "Q3"]:
        quali[col + "_seconds"] = (
            pd.to_timedelta(quali[col], errors="coerce")
            .dt.total_seconds()
        )

    quali_features = quali[
        ["RoundNumber", "Abbreviation", "quali_position"]
    ].copy()

    quali_features = quali_features.rename(
        columns={
            "RoundNumber": "round_number",
            "Abbreviation": "driver_code",
        }
    )

    # ---------------------------------------------
    # 5. Practice pace relative to each session
    # ---------------------------------------------
    practice["BestLapSeconds"] = pd.to_numeric(
        practice["BestLapSeconds"], errors="coerce"
    )

    practice = practice.dropna(
        subset=["Driver", "BestLapSeconds"]
    ).copy()

    # Compare only drivers within the same event/session.
    practice["practice_pace_delta"] = (
        practice["BestLapSeconds"]
        - practice.groupby(
            ["RoundNumber", "SessionType"]
        )["BestLapSeconds"].transform("median")
    )

    practice_features = (
        practice.groupby(
            ["RoundNumber", "Driver"], as_index=False
        )
        .agg(
            practice_pace_delta=("practice_pace_delta", "median"),
            practice_timed_laps=("TimedLaps", "sum"),
        )
        .rename(
            columns={
                "RoundNumber": "round_number",
                "Driver": "driver_code",
            }
        )
    )

    # ---------------------------------------------
    # 6. Merge same-event observations temporarily
    # ---------------------------------------------
    base = base.merge(
        quali_features,
        on=["round_number", "driver_code"],
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        practice_features,
        on=["round_number", "driver_code"],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------
    # 7. Shift all driver observations by one round
    # ---------------------------------------------
    base = base.sort_values(
        ["driver_id", "round_number"]
    ).reset_index(drop=True)

    driver_group = base.groupby("driver_id", sort=False)

    base["prev_quali_position"] = (
        driver_group["quali_position"].shift(1)
    )

    base["prev_practice_pace_delta"] = (
        driver_group["practice_pace_delta"].shift(1)
    )

    base["prev_practice_timed_laps"] = (
        driver_group["practice_timed_laps"].shift(1)
    )

    # ---------------------------------------------
    # 8. Team performance from previous events
    # ---------------------------------------------
    team_round = (
        base.groupby(
            ["round_number", "team_name"], as_index=False
        )
        .agg(
            team_finish_avg=("finish_position", "mean")
        )
        .sort_values(["team_name", "round_number"])
    )

    team_round["prev_team_finish_avg"] = (
        team_round.groupby("team_name")[
            "team_finish_avg"
        ].shift(1)
    )

    base = base.merge(
        team_round[
            ["round_number", "team_name", "prev_team_finish_avg"]
        ],
        on=["round_number", "team_name"],
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------
    # 9. Final modeling schema
    # ---------------------------------------------
    feature_columns = [
        "season",
        "round_number",
        "event_name",
        "event_date",
        "driver_code",
        "driver_id",
        "team_name",
        "prev_finish_position",
        "rolling_finish_avg_3",
        "prev_win_count_3",
        "prev_quali_position",
        "prev_practice_pace_delta",
        "prev_practice_timed_laps",
        "prev_team_finish_avg",
        "is_winner",
    ]

    output = base[feature_columns].copy()

    output = output.sort_values(
        ["round_number", "driver_code"]
    ).reset_index(drop=True)

    output_path = (
        PROCESSED_DIR
        / f"preweekend_features_{SEASON}.csv"
    )

    output.to_csv(output_path, index=False)

    print("Pre-weekend feature dataset created.")
    print("Shape:", output.shape)
    print("Saved to:", output_path)

    print("\nMissing values by feature:")
    print(
        output.isna().sum()
        .sort_values(ascending=False)
    )

    print("\nFirst five rows:")
    print(output.head().to_string(index=False))

    return output


if __name__ == "__main__":
    build_features()
