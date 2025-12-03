
import argparse
import pandas as pd
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

def main():
    ap = argparse.ArgumentParser(description="Count duplicate rows in a Parquet file.")
    ap.add_argument("--input", required=True, help="Path to .parquet file (e.g., merged_outputs/combined.parquet)")
    ap.add_argument("--sample", type=int, default=None, help="Optional: limit number of rows for faster testing")
    args = ap.parse_args()

    print(f"[INFO] Loading Parquet: {args.input}")
    pf = pq.ParquetFile(args.input)
    dfs = []
    rows_loaded = 0

    for batch in pf.iter_batches(batch_size=100_000):
        df = batch.to_pandas()
        dfs.append(df)
        rows_loaded += len(df)
        if args.sample and rows_loaded >= args.sample:
            break

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Loaded {len(df):,} rows | {len(df.columns)} columns")

    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    # Compute duplicates using hash
    h = hash_pandas_object(df, index=False)
    dup_mask = h.duplicated(keep=False)

    total_rows = len(df)
    dup_count = dup_mask.sum()
    unique_count = total_rows - dup_count
    dup_ratio = dup_count / total_rows if total_rows > 0 else 0

    print("\n=== DUPLICATE SUMMARY ===")
    print(f"Total rows       : {total_rows:,}")
    print(f"Unique rows      : {unique_count:,}")
    print(f"Duplicate rows   : {dup_count:,}")
    print(f"Duplicate ratio  : {dup_ratio:.4%}")

    if "Scenario" in df.columns:
        scen_dupes = df.loc[dup_mask, "Scenario"].value_counts().sort_index()
        print("\n=== DUPLICATES PER SCENARIO ===")
        print(scen_dupes.to_string())

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
