
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "preweekend_features_2026.csv"

df = pd.read_csv(PATH)

print("Dataset shape:", df.shape)
print("Rounds:", sorted(df["round_number"].unique()))
print("Unique drivers:", df["driver_code"].nunique())
print("Rows per round:")
print(df.groupby("round_number").size())

print("\nWinner count per round:")
print(df.groupby("round_number")["is_winner"].sum())

print("\nTarget distribution:")
print(df["is_winner"].value_counts().sort_index())

print("\nRound 1 feature missingness:")
print(df.loc[df["round_number"] == 1].isna().sum())

print("\nSample historical features for Round 2:")
cols = [
    "round_number",
    "event_name",
    "driver_code",
    "prev_finish_position",
    "rolling_finish_avg_3",
    "prev_win_count_3",
    "prev_quali_position",
    "prev_practice_pace_delta",
    "prev_practice_timed_laps",
    "prev_team_finish_avg",
    "is_winner",
]
print(
    df.loc[df["round_number"] == 2, cols]
      .sort_values("driver_code")
      .to_string(index=False)
)

# Historical features should not contain current-round race outcomes.
assert df.groupby("round_number")["is_winner"].sum().eq(1).all()
assert not df.duplicated(["round_number", "driver_code"]).any()

print("\nBasic validation passed.")
