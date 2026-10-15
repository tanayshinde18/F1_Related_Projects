
from pathlib import Path

import pandas as pd


SEASON = 2026

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_master_dataframe() -> pd.DataFrame:
    path = RAW_DIR / f"race_results_{SEASON}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run data_collection.py first."
        )

    race = pd.read_csv(path)

    required_columns = {
        "Season",
        "RoundNumber",
        "EventName",
        "Abbreviation",
        "TeamName",
        "Position",
    }

    missing = required_columns - set(race.columns)

    if missing:
        raise ValueError(
            f"Race results are missing columns: {sorted(missing)}"
        )

    # Preserve the original result position for auditing.
    race["finish_position"] = pd.to_numeric(
        race["Position"],
        errors="coerce",
    )

    race["is_winner"] = (
        race["finish_position"] == 1
    ).astype("int8")

    # Keep only fields useful for the initial master table.
    desired_columns = [
        "Season",
        "RoundNumber",
        "EventName",
        "EventDate",
        "Abbreviation",
        "DriverNumber",
        "TeamName",
        "GridPosition",
        "finish_position",
        "Status",
        "Points",
        "is_winner",
    ]

    available_columns = [
        col for col in desired_columns
        if col in race.columns
    ]

    master = race[available_columns].copy()

    master = master.rename(
        columns={
            "Season": "season",
            "RoundNumber": "round_number",
            "EventName": "event_name",
            "EventDate": "event_date",
            "Abbreviation": "driver_code",
            "DriverNumber": "driver_number",
            "TeamName": "team_name",
            "GridPosition": "grid_position",
            "Status": "result_status",
            "Points": "points",
        }
    )

    master["event_date"] = pd.to_datetime(
        master["event_date"],
        errors="coerce",
    )

    master = master.sort_values(
        ["round_number", "driver_code"]
    ).reset_index(drop=True)

    # Basic integrity checks.
    if master["driver_code"].isna().any():
        raise ValueError("Some race-result rows lack driver codes.")

    duplicate_rows = master.duplicated(
        ["season", "round_number", "driver_code"]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate driver/event rows detected. "
            "Inspect the raw race results before proceeding."
        )

    winner_counts = master.groupby(
        ["season", "round_number"]
    )["is_winner"].sum()

    invalid_events = winner_counts[winner_counts != 1]

    if not invalid_events.empty:
        print(
            "WARNING: These events do not have exactly "
            "one identified winner:"
        )
        print(invalid_events)

    output_path = (
        PROCESSED_DIR
        / f"master_results_{SEASON}.csv"
    )

    master.to_csv(output_path, index=False)

    print("\nMaster dataframe created.")
    print(f"Shape: {master.shape}")
    print(f"Saved to: {output_path}")

    print("\nFirst five rows:")
    print(master.head())

    print("\nData types:")
    print(master.dtypes)

    print("\nMissing values:")
    print(master.isna().sum().sort_values(ascending=False))

    print("\nRace-level winner counts:")
    print(winner_counts)

    return master


if __name__ == "__main__":
    build_master_dataframe()
