
from pathlib import Path
import pandas as pd

root = Path("data/raw")

for name in [
    "race_results_2026.csv",
    "qualifying_results_2026.csv",
    "practice_pace_2026.csv",
]:
    path = root / name

    print(f"\n{'=' * 60}")
    print(name)

    if not path.exists():
        print("File not found")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
