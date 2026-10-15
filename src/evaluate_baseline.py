
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "data" / "processed" / "master_results_2026.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "preweekend_features_2026.csv"
RF_PATH = ROOT / "data" / "processed" / "baseline_evaluation_2026.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "baseline_comparison_2026.csv"


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

for path in [MASTER_PATH, FEATURES_PATH, RF_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

master = pd.read_csv(MASTER_PATH)
df = pd.read_csv(FEATURES_PATH)
rf = pd.read_csv(RF_PATH)

# Normalize column names in case they contain unexpected whitespace.
master.columns = master.columns.str.strip()
df.columns = df.columns.str.strip()
rf.columns = rf.columns.str.strip()

# Ensure round numbers have consistent integer types.
for data in [master, df, rf]:
    data["round_number"] = pd.to_numeric(
        data["round_number"], errors="raise"
    ).astype(int)

# Ensure target values are numeric.
master["is_winner"] = pd.to_numeric(master["is_winner"], errors="raise")
df["is_winner"] = pd.to_numeric(df["is_winner"], errors="raise")


# --------------------------------------------------
# 3. Validate the datasets
# --------------------------------------------------

required_master_columns = {
    "round_number",
    "driver_code",
    "finish_position",
    "is_winner",
}

required_feature_columns = {
    "round_number",
    "driver_code",
    "event_name",
    "is_winner",
}

required_rf_columns = {
    "round_number",
    "predicted_winner",
    "top_1_correct",
    "top_3_hit",
}

if not required_master_columns.issubset(master.columns):
    missing = required_master_columns - set(master.columns)
    raise ValueError(f"Missing columns in master results: {missing}")

if not required_feature_columns.issubset(df.columns):
    missing = required_feature_columns - set(df.columns)
    raise ValueError(f"Missing columns in feature dataset: {missing}")

if not required_rf_columns.issubset(rf.columns):
    missing = required_rf_columns - set(rf.columns)
    raise ValueError(f"Missing columns in Random Forest results: {missing}")

if master.duplicated(["round_number", "driver_code"]).any():
    raise ValueError("Duplicate driver-round rows found in master results.")

if df.duplicated(["round_number", "driver_code"]).any():
    raise ValueError("Duplicate driver-round rows found in feature dataset.")

if rf["round_number"].duplicated().any():
    raise ValueError("Random Forest results contain duplicate rounds.")


# --------------------------------------------------
# 4. Evaluate chronological baselines
# --------------------------------------------------

results = []

all_rounds = sorted(df["round_number"].unique())

for round_number in all_rounds:

    # Use Rounds 1-3 as the initial training/history window.
    if round_number < 4:
        continue

    # Current race: use only to identify the actual outcome.
    current = df.loc[df["round_number"] == round_number].copy()

    # Historical race results: strictly earlier rounds only.
    history = master.loc[
        master["round_number"] < round_number
    ].copy()

    if history.empty:
        continue

    # Confirm exactly one winner exists for the current race.
    current_winners = current.loc[
        current["is_winner"] == 1, "driver_code"
    ]

    if len(current_winners) != 1:
        raise ValueError(
            f"Expected exactly one winner in Round {round_number}, "
            f"found {len(current_winners)}."
        )

    actual_winner = current_winners.iloc[0]

    # ----------------------------------------------
    # Baseline 1: Previous race winner
    # ----------------------------------------------

    previous_round = history["round_number"].max()

    previous_winners = history.loc[
        (history["round_number"] == previous_round)
        & (history["is_winner"] == 1),
        "driver_code",
    ]

    if len(previous_winners) != 1:
        raise ValueError(
            f"Expected exactly one winner in historical "
            f"Round {previous_round}."
        )

    previous_winner_pick = previous_winners.iloc[0]

    previous_winner_correct = int(
        previous_winner_pick == actual_winner
    )

    # ----------------------------------------------
    # Baseline 2: Best recent finishing form
    # ----------------------------------------------
    # For each driver, average their last three available
    # finishing positions before the target round.
    #
    # Lower finishing positions are better.
    # Ties are resolved alphabetically by driver code.

    recent_results = (
        history.sort_values(["round_number", "driver_code"])
        .groupby("driver_code", group_keys=False)
        .tail(3)
    )

    recent_form = (
        recent_results.groupby("driver_code")["finish_position"]
        .mean()
        .dropna()
    )

    if recent_form.empty:
        raise ValueError(
            f"No valid recent finishing positions before "
            f"Round {round_number}."
        )

    best_average = recent_form.min()

    tied_drivers = recent_form[
        recent_form == best_average
    ].index.tolist()

    best_form_driver = sorted(tied_drivers)[0]

    best_recent_form_correct = int(
        best_form_driver == actual_winner
    )

    results.append({
        "round_number": round_number,
        "event_name": current["event_name"].iloc[0],
        "actual_winner": actual_winner,
        "previous_winner_pick": previous_winner_pick,
        "previous_winner_correct": previous_winner_correct,
        "best_recent_form_pick": best_form_driver,
        "best_recent_form_correct": best_recent_form_correct,
    })


# --------------------------------------------------
# 5. Compare baselines against Random Forest
# --------------------------------------------------

baseline = pd.DataFrame(results)

if baseline.empty:
    raise ValueError("No rounds available for baseline evaluation.")

comparison = baseline.merge(
    rf[
        [
            "round_number",
            "predicted_winner",
            "top_1_correct",
            "top_3_hit",
        ]
    ],
    on="round_number",
    how="left",
    validate="one_to_one",
)

if comparison["predicted_winner"].isna().any():
    missing_rounds = comparison.loc[
        comparison["predicted_winner"].isna(), "round_number"
    ].tolist()

    raise ValueError(
        f"Random Forest predictions missing for rounds: {missing_rounds}"
    )

# Convert evaluation columns to numeric values.
for column in [
    "top_1_correct",
    "top_3_hit",
]:
    comparison[column] = pd.to_numeric(
        comparison[column], errors="raise"
    )


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n" + "=" * 72)
print("F1 WINNER PREDICTION: CHRONOLOGICAL BASELINE COMPARISON")
print("=" * 72)

print("\nRound-by-round comparison:")
print(comparison.to_string(index=False))

print("\n" + "-" * 45)
print("TOP-1 ACCURACY COMPARISON")
print("-" * 45)

rf_accuracy = comparison["top_1_correct"].mean()

previous_winner_accuracy = comparison[
    "previous_winner_correct"
].mean()

recent_form_accuracy = comparison[
    "best_recent_form_correct"
].mean()

print(
    f"Random Forest:        {rf_accuracy:.3f} "
    f"({comparison['top_1_correct'].sum():.0f}/"
    f"{len(comparison)})"
)

print(
    f"Previous race winner: {previous_winner_accuracy:.3f} "
    f"({comparison['previous_winner_correct'].sum():.0f}/"
    f"{len(comparison)})"
)

print(
    f"Best recent form:     {recent_form_accuracy:.3f} "
    f"({comparison['best_recent_form_correct'].sum():.0f}/"
    f"{len(comparison)})"
)

print(
    f"Random Forest Top-3 hit rate: "
    f"{comparison['top_3_hit'].mean():.3f}"
)

# Identify the highest-scoring approach.
scores = {
    "Random Forest": rf_accuracy,
    "Previous race winner": previous_winner_accuracy,
    "Best recent form": recent_form_accuracy,
}

best_method = max(scores, key=scores.get)

print(f"\nHighest Top-1 score: {best_method}")

# Save the comparison.
comparison.to_csv(OUTPUT_PATH, index=False)

print(f"\nComparison saved to:\n{OUTPUT_PATH}")
print("\nEvaluation completed successfully.")
