
import sqlite3
import statistics
from pathlib import Path
from time import perf_counter

import pandas as pd

from search_methods import (
    linear_search_by_id,
    binary_search_by_id,
    hash_search_by_id,
    linear_search_multi,
)

# ============================================================
# MAI 560 - Planning, Search, and Reasoning Under Uncertainty
# Assignment: Search Methods - Speed and Effectiveness
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "AI_Search_500k_Dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "results"

REPETITIONS = 30
TARGET_ID = "STU-425000"

MULTI_PARAMS = ("Colombia", "MS AI", 3.50, 80)

EXACT_SQL = """
    SELECT Record_ID
    FROM students
    WHERE Record_ID = ?
"""

MULTI_SQL = """
    SELECT Record_ID
    FROM students
    WHERE Country = ?
      AND Program = ?
      AND GPA >= ?
      AND Attendance_Pct < ?
"""


# ============================================================
# 1. TIMING
# ============================================================

def measure(func):
    """Run a search 30 times and return its median time."""

    times = []
    result = None

    for _ in range(REPETITIONS):
        start = perf_counter()
        result = func()
        elapsed_ms = (perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return statistics.median(times), result


# ============================================================
# 2. PRECISION AND RECALL
# ============================================================

def evaluate(result_ids, reference_ids):
    """Compare retrieved IDs with an independent reference."""

    predicted = set(result_ids)
    expected = set(reference_ids)

    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else (1.0 if not expected else 0.0)
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    return precision * 100, recall * 100


# ============================================================
# 3. BENCHMARK EXECUTION
# ============================================================

def run_methods(methods, reference):
    """Benchmark methods and validate their returned records."""

    results = []

    for name, func in methods.items():

        median_ms, ids = measure(func)
        precision, recall = evaluate(ids, reference)

        if len(ids) != len(set(ids)):
            raise AssertionError(
                f"{name}: duplicate IDs returned"
            )

        if set(ids) != reference:
            raise AssertionError(
                f"{name}: results do not match reference"
            )

        results.append({
            "Method": name,
            "Median ms": median_ms,
            "Precision %": precision,
            "Recall %": recall,
        })

        print(
            f"{name:<15} "
            f"{median_ms:>12.6f} ms "
            f"| Precision: {precision:.1f}% "
            f"| Recall: {recall:.1f}%"
        )

    return results


# ============================================================
# 4. PERFORMANCE INDEX
# ============================================================

def finalize(results, include_na=False):

    fastest = min(
        r["Median ms"] for r in results
    )

    for r in results:

        speed_score = min(
            100.0,
            100 * fastest / r["Median ms"]
        )

        overall_index = (
            0.40 * r["Precision %"]
            + 0.30 * r["Recall %"]
            + 0.30 * speed_score
        )

        r["Speed score"] = speed_score
        r["Index /100"] = overall_index

    table = pd.DataFrame(results).sort_values(
        "Index /100",
        ascending=False
    )

    if include_na:
        na_rows = pd.DataFrame([
            {
                "Method": method,
                "Median ms": float("nan"),
                "Precision %": float("nan"),
                "Recall %": float("nan"),
                "Speed score": float("nan"),
                "Index /100": float("nan"),
            }
            for method in ("Binary", "Hash")
        ])

        table = pd.concat(
            [table, na_rows],
            ignore_index=True
        )

    return table


# ============================================================
# 5. SQLITE QUERY PLAN VERIFICATION
# ============================================================

def verify_query_plan(conn, query, params, index_name):

    print(f"\nChecking SQLite index: {index_name}")

    plan = conn.execute(
        "EXPLAIN QUERY PLAN " + query,
        params
    ).fetchall()

    for row in plan:
        print(row)

    details = " ".join(
        str(row[3]) for row in plan
    )

    if index_name not in details:
        raise AssertionError(
            f"Expected index {index_name} "
            "was not found in the execution plan."
        )

    print("Index usage verified successfully!")


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_times = {}

    print("\nMAI 560 - SEARCH BENCHMARK")
    print("=" * 65)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    start = perf_counter()

    df = pd.read_excel(
        DATASET,
        sheet_name="Data"
    )

    setup_times["Excel loading"] = (
        perf_counter() - start
    )

    assert len(df) == 500000
    assert df["Record_ID"].is_unique
    assert df["Record_ID"].is_monotonic_increasing

    print(f"Records loaded: {len(df):,}")

    # --------------------------------------------------------
    # Prepare Python records
    # --------------------------------------------------------

    start = perf_counter()

    records = df.to_dict("records")

    setup_times["Record conversion"] = (
        perf_counter() - start
    )

    start = perf_counter()

    lookup = {
        record["Record_ID"]: record
        for record in records
    }

    setup_times["Hash construction"] = (
        perf_counter() - start
    )

    # --------------------------------------------------------
    # Prepare SQLite
    # --------------------------------------------------------

    print("\nPreparing SQLite database...")

    conn = sqlite3.connect(":memory:")

    start = perf_counter()

    df.to_sql(
        "students",
        conn,
        index=False,
        if_exists="replace"
    )

    setup_times["SQLite loading"] = (
        perf_counter() - start
    )

    def sql_query(query, params=()):
        return [
            row[0]
            for row in conn.execute(
                query, params
            ).fetchall()
        ]

    # --------------------------------------------------------
    # Independent full-scan reference
    # --------------------------------------------------------

    print("\nBuilding independent reference results...")

    exact_reference = {
        record["Record_ID"]
        for record in records
        if record["Record_ID"] == TARGET_ID
    }

    multi_reference = {
        record["Record_ID"]
        for record in records
        if (
            record["Country"] == "Colombia"
            and record["Program"] == "MS AI"
            and record["GPA"] >= 3.50
            and record["Attendance_Pct"] < 80
        )
    }

    print(
        f"Exact-ID reference matches: "
        f"{len(exact_reference)}"
    )

    print(
        f"Multi-condition reference matches: "
        f"{len(multi_reference)}"
    )

    # --------------------------------------------------------
    # DataFrame queries
    # --------------------------------------------------------

    def dataframe_exact():
        return df.loc[
            df["Record_ID"] == TARGET_ID,
            "Record_ID"
        ].tolist()

    def dataframe_multi():

        mask = (
            (df["Country"] == "Colombia")
            & (df["Program"] == "MS AI")
            & (df["GPA"] >= 3.50)
            & (df["Attendance_Pct"] < 80)
        )

        return df.loc[
            mask, "Record_ID"
        ].tolist()

    def record_ids(result):
        return [
            record["Record_ID"]
            for record in result
        ]

    # ========================================================
    # BENCHMARK 1 - EXACT ID
    # ========================================================

    print("\n" + "=" * 65)
    print("BENCHMARK 1 - EXACT ID")
    print("=" * 65)

    exact_methods = {
        "Linear": lambda: record_ids(
            linear_search_by_id(records, TARGET_ID)
        ),
        "Binary": lambda: record_ids(
            binary_search_by_id(records, TARGET_ID)
        ),
        "Hash": lambda: record_ids(
            hash_search_by_id(lookup, TARGET_ID)
        ),
        "DataFrame": dataframe_exact,
        "SQL no index": lambda: sql_query(
            EXACT_SQL, (TARGET_ID,)
        ),
    }

    exact_results = run_methods(
        exact_methods,
        exact_reference
    )

    # Create exact-ID index

    start = perf_counter()

    conn.execute("""
        CREATE INDEX idx_record_id
        ON students(Record_ID)
    """)

    setup_times["Exact-ID index creation"] = (
        perf_counter() - start
    )

    verify_query_plan(
        conn,
        EXACT_SQL,
        (TARGET_ID,),
        "idx_record_id"
    )

    exact_results.extend(
        run_methods(
            {
                "SQL indexed": lambda: sql_query(
                    EXACT_SQL, (TARGET_ID,)
                )
            },
            exact_reference
        )
    )

    # ========================================================
    # BENCHMARK 2 - MULTI-CONDITION
    # ========================================================

    print("\n" + "=" * 65)
    print("BENCHMARK 2 - MULTI-CONDITION")
    print("=" * 65)

    multi_methods = {
        "Linear": lambda: record_ids(
            linear_search_multi(records)
        ),
        "DataFrame": dataframe_multi,
        "SQL no index": lambda: sql_query(
            MULTI_SQL, MULTI_PARAMS
        ),
    }

    multi_results = run_methods(
        multi_methods,
        multi_reference
    )

    # Create composite index

    start = perf_counter()

    conn.execute("""
        CREATE INDEX idx_multi
        ON students(
            Country,
            Program,
            GPA,
            Attendance_Pct
        )
    """)

    setup_times["Composite index creation"] = (
        perf_counter() - start
    )

    verify_query_plan(
        conn,
        MULTI_SQL,
        MULTI_PARAMS,
        "idx_multi"
    )

    multi_results.extend(
        run_methods(
            {
                "SQL indexed": lambda: sql_query(
                    MULTI_SQL, MULTI_PARAMS
                )
            },
            multi_reference
        )
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    exact_table = finalize(exact_results)
    multi_table = finalize(
        multi_results,
        include_na=True
    )

    print("\n" + "=" * 65)
    print("FINAL TABLE - EXACT ID")
    print("=" * 65)
    print(
        exact_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 65)
    print("FINAL TABLE - MULTI-CONDITION")
    print("=" * 65)
    print(
        multi_table.to_string(
            index=False,
            na_rep="N/A",
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 65)
    print("SETUP TIMES (SECONDS)")
    print("=" * 65)

    for name, seconds in setup_times.items():
        print(f"{name:<30} {seconds:.3f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    exact_table.to_csv(
        OUTPUT_DIR / "exact_id_results.csv",
        index=False
    )

    multi_table.to_csv(
        OUTPUT_DIR / "multi_condition_results.csv",
        index=False,
        na_rep="N/A"
    )

    pd.DataFrame(
        [
            {
                "Operation": name,
                "Seconds": seconds
            }
            for name, seconds in setup_times.items()
        ]
    ).to_csv(
        OUTPUT_DIR / "setup_times.csv",
        index=False
    )

    conn.close()

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("Results saved in the 'results' folder.")
    print("=" * 65)


if __name__ == "__main__":
    main()
