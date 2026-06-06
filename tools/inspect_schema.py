"""
tools/inspect_schema.py

Run this FIRST after downloading the Kaggle dataset to see your CSV columns.
It will print a schema report and suggest any COLUMN_MAP edits needed.

Usage:
    python tools/inspect_schema.py
"""
import sys
from pathlib import Path
import pandas as pd
import glob

ROOT = Path(__file__).parent.parent
KAGGLE_DIR = ROOT / "data" / "kaggle"


def inspect():
    csv_files = sorted(KAGGLE_DIR.glob("*.csv"))

    if not csv_files:
        print(f"\n⚠️  No CSV files found in {KAGGLE_DIR}")
        print("   Download the dataset first:")
        print("   kaggle datasets download -d gabriellecharlton/coffee-shop-financial-dataset-synthetic-2022-2023 \\")
        print(f"     -p {KAGGLE_DIR} --unzip\n")
        return

    print(f"\n{'='*60}")
    print("  CSV SCHEMA INSPECTOR")
    print(f"{'='*60}\n")

    for f in csv_files:
        df = pd.read_csv(f, nrows=5)
        print(f"📄 {f.name}")
        print(f"   Rows (sample): {len(df)}  |  Total cols: {len(df.columns)}")
        print(f"   Columns: {list(df.columns)}\n")

        for col in df.columns:
            sample_vals = df[col].dropna().head(3).tolist()
            dtype = str(df[col].dtype)
            print(f"   {col:<30} ({dtype:<10})  e.g. {sample_vals}")
        print()

    # Now check what the loader would map
    print(f"{'='*60}")
    print("  COLUMN MAPPING CHECK")
    print(f"{'='*60}\n")

    from data.loader import COLUMN_MAP
    all_cols = []
    for f in csv_files:
        df = pd.read_csv(f, nrows=1)
        all_cols.extend(df.columns.tolist())

    for std_name, candidates in COLUMN_MAP.items():
        matched = next((c for c in candidates if c in all_cols), None)
        status  = f"✅  mapped to '{matched}'" if matched else "❌  NOT FOUND — edit COLUMN_MAP in data/loader.py"
        print(f"  {std_name:<15} {status}")

    print("\n")


if __name__ == "__main__":
    inspect()
