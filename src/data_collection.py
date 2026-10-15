
from pathlib import Path

import fastf1
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEASON = 2026

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


# --------------------------------------------------
# 1. Load the season schedule
# --------------------------------------------------

def get_completed_events(season: int = SEASON) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(
        season,
        include_testing=False,
    )

    schedule.to_csv(
        RAW_DIR / f"event_schedule_{season}.csv",
        index=False,
    )

    schedule = schedule.copy()
    schedule["EventDate"] = pd.to_datetime(
        schedule["EventDate"],
        errors="coerce",
    ).dt.tz_localize(None)

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    completed = schedule.loc[
        (schedule["RoundNumber"] > 0)
        & (schedule["EventDate"] < today)
    ].copy()

    return completed.sort_values("RoundNumber")


# --------------------------------------------------
# 2. Add event metadata to a session dataframe
# --------------------------------------------------

def add_event_metadata(
    df: pd.DataFrame,
    event: pd.Series,
    season: int,
    session_type: str,
) -> pd.DataFrame:
    df = df.copy()

    df["Season"] = season
    df["RoundNumber"] = int(event["RoundNumber"])
    df["EventName"] = event["EventName"]
    df["EventDate"] = event["EventDate"]
    df["SessionType"] = session_type

    return df


# --------------------------------------------------
# 3. Collect race and qualifying results
# --------------------------------------------------

def collect_classification(
    event: pd.Series,
    session_type: str,
    season: int = SEASON,
) -> pd.DataFrame:
    session = fastf1.get_session(
        season,
        int(event["RoundNumber"]),
        session_type,
    )

    session.load()
    results = session.results.copy()

    if results.empty:
        print(
            f"No results: round {event['RoundNumber']} "
            f"{session_type}"
        )
        return pd.DataFrame()

    return add_event_metadata(
        results,
        event,
        season,
        session_type,
    )


# --------------------------------------------------
# 4. Collect practice pace summaries
# --------------------------------------------------

def collect_practice_pace(
    event: pd.Series,
    session_type: str,
    season: int = SEASON,
) -> pd.DataFrame:
    session = fastf1.get_session(
        season,
        int(event["RoundNumber"]),
        session_type,
    )

    session.load()
    laps = session.laps

    if laps.empty:
        return pd.DataFrame()

    # Retain reasonably representative timed laps.
    # We will refine the cleaning rules during EDA.
    quick_laps = laps.pick_quicklaps()

    if quick_laps.empty:
        return pd.DataFrame()

    pace = (
        quick_laps
        .groupby(["Driver", "Team"], as_index=False)
        .agg(
            BestLapTime=("LapTime", "min"),
            TimedLaps=("LapTime", "count"),
        )
    )

    pace["BestLapSeconds"] = (
        pace["BestLapTime"].dt.total_seconds()
    )

    return add_event_metadata(
        pace,
        event,
        season,
        session_type,
    )


# --------------------------------------------------
# 5. Download all completed events
# --------------------------------------------------

def main():
    events = get_completed_events()

    print(f"Completed events found: {len(events)}")

    race_results = []
    qualifying_results = []
    practice_pace = []

    for _, event in events.iterrows():
        round_number = int(event["RoundNumber"])
        event_name = event["EventName"]

        print(f"\nRound {round_number}: {event_name}")

        # Race results are our source for winner labels.
        try:
            race = collect_classification(event, "R")

            if not race.empty:
                race_results.append(race)

        except Exception as exc:
            print(f"Race data unavailable: {exc}")

        # Qualifying results are stored as raw data.
        # They will NOT be used to predict that same race.
        try:
            quali = collect_classification(event, "Q")

            if not quali.empty:
                qualifying_results.append(quali)

        except Exception as exc:
            print(f"Qualifying data unavailable: {exc}")

        # Practice data is also stored separately.
        # Same-event practice must not become a feature
        # for that event in our pre-weekend model.
        for session_type in ["FP1", "FP2", "FP3"]:
            try:
                pace = collect_practice_pace(
                    event,
                    session_type,
                )

                if not pace.empty:
                    practice_pace.append(pace)

            except Exception as exc:
                print(
                    f"{session_type} unavailable "
                    f"for round {round_number}: {exc}"
                )

    # Save raw session-level datasets.
    datasets = {
        "race_results": race_results,
        "qualifying_results": qualifying_results,
        "practice_pace": practice_pace,
    }

    for name, frames in datasets.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)

            df.to_csv(
                RAW_DIR / f"{name}_{SEASON}.csv",
                index=False,
            )

            print(
                f"Saved {name}: {df.shape[0]} rows, "
                f"{df.shape[1]} columns"
            )
        else:
            print(f"No data collected for {name}")


if __name__ == "__main__":
    main()
