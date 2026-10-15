
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "preweekend_features_2026.csv"

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["round_number", "driver_code"]).reset_index(drop=True)

numeric_features = [
    "prev_finish_position",
    "rolling_finish_avg_3",
    "prev_win_count_3",
    "prev_quali_position",
    "prev_practice_pace_delta",
    "prev_practice_timed_laps",
    "prev_team_finish_avg",
]
categorical_features = ["team_name"]
feature_columns = numeric_features + categorical_features

# Start at Round 4 so the first training window includes three races.
test_rounds = sorted(df["round_number"].unique())
test_rounds = [r for r in test_rounds if r >= 4]

results = []

for test_round in test_rounds:
    train = df[df["round_number"] < test_round].copy()
    test = df[df["round_number"] == test_round].copy()

    X_train = train[feature_columns]
    y_train = train["is_winner"].astype(int)
    X_test = test[feature_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    test["predicted_probability"] = model.predict_proba(X_test)[:, 1]
    test = test.sort_values(
        "predicted_probability", ascending=False
    ).reset_index(drop=True)

    actual_winner = test.loc[test["is_winner"] == 1, "driver_code"].iloc[0]
    predicted_winner = test.loc[0, "driver_code"]
    top_three = test.head(3)["driver_code"].tolist()

    results.append({
        "round_number": test_round,
        "event_name": test["event_name"].iloc[0],
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner,
        "top_1_correct": int(predicted_winner == actual_winner),
        "top_3_hit": int(actual_winner in top_three),
        "top_3_drivers": ", ".join(top_three),
        "training_races": test_round - 1,
    })

results_df = pd.DataFrame(results)

print("\nChronological evaluation:")
print(results_df.to_string(index=False))

print("\nSummary:")
print("Races evaluated:", len(results_df))
print("Top-1 accuracy:", round(results_df["top_1_correct"].mean(), 3))
print("Top-3 hit rate:", round(results_df["top_3_hit"].mean(), 3))

output_path = ROOT / "data" / "processed" / "baseline_evaluation_2026.csv"
results_df.to_csv(output_path, index=False)
print("\nSaved results to:", output_path)
