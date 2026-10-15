
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "master_results_2026.csv"

df = pd.read_csv(PATH)

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("Unique rounds:", sorted(df["round_number"].unique()))
print("Unique drivers:", df["driver_code"].nunique())
print("Unique teams:", df["team_name"].nunique())

print("\n" + "=" * 60)
print("2. DUPLICATE DRIVER-EVENT RECORDS")
print("=" * 60)

duplicates = df.duplicated(
    ["season", "round_number", "driver_code"],
    keep=False,
)

print("Duplicate rows:", duplicates.sum())

if duplicates.any():
    print(
        df.loc[duplicates].sort_values(
            ["round_number", "driver_code"]
        )
    )

print("\n" + "=" * 60)
print("3. WINNER CONSISTENCY")
print("=" * 60)

winner_counts = df.groupby("round_number")["is_winner"].sum()
print(winner_counts)

print("\nRounds without exactly one winner:")
print(winner_counts[winner_counts != 1])

print("\nRows marked as winners:")
print(
    df.loc[
        df["is_winner"] == 1,
        [
            "round_number",
            "event_name",
            "driver_code",
            "team_name",
            "finish_position",
            "result_status",
        ],
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("4. DRIVER COUNTS PER ROUND")
print("=" * 60)

print(
    df.groupby("round_number")["driver_code"]
    .nunique()
    .rename("driver_count")
)

print("\n" + "=" * 60)
print("5. GRID AND FINISH POSITION RANGES")
print("=" * 60)

for col in ["grid_position", "finish_position"]:
    print(f"\n{col}")
    print(df[col].describe())
    print("Smallest values:", sorted(df[col].unique())[:10])
    print("Largest values:", sorted(df[col].unique())[-10:])

print("\n" + "=" * 60)
print("6. RESULT STATUS COUNTS")
print("=" * 60)

print(df["result_status"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("7. DRIVER TEAM CHANGES")
print("=" * 60)

team_counts = df.groupby("driver_code")["team_name"].nunique()
print(team_counts[team_counts > 1])

print("\nAudit complete.")
