
from pathlib import Path
from time import perf_counter
import pandas as pd

DATASET_PATH = Path(__file__).parent / "AI_Search_500k_Dataset.xlsx"

def load_and_validate():
    print("\nMAI 560 - DATASET VALIDATION")
    print("=" * 55)

    start = perf_counter()
    df = pd.read_excel(DATASET_PATH, sheet_name="Data")
    loading_time = perf_counter() - start

    print(f"Dataset shape: {df.shape}")
    print(f"Loading time: {loading_time:.3f} seconds")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nDATA QUALITY")
    print("-" * 55)
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate Record_IDs: {df['Record_ID'].duplicated().sum()}")
    print(f"IDs sorted: {df['Record_ID'].is_monotonic_increasing}")

    print("\nEXACT-ID QUERY")
    print("-" * 55)
    target = "STU-425000"
    result = df.loc[df["Record_ID"] == target]
    print(f"Target: {target}")
    print(f"Matches: {len(result)}")
    print(result.to_string(index=False))

    print("\nMULTI-CONDITION QUERY")
    print("-" * 55)

    mask = (
        (df["Country"] == "Colombia")
        & (df["Program"] == "MS AI")
        & (df["GPA"] >= 3.50)
        & (df["Attendance_Pct"] < 80)
    )

    matches = df.loc[mask]
    print(f"Matching records: {len(matches)}")
    print(matches.head(10).to_string(index=False))

    return df

if __name__ == "__main__":
    df = load_and_validate()

