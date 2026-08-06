"""
Task 1 deliverable: report on the raw scraped data.

Run after the spider finishes:
    python report_raw_stats.py
"""
import pandas as pd

RAW_PATH = "data/raw/books_raw.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    print("=" * 50)
    print("RAW SCRAPE REPORT")
    print("=" * 50)
    print(f"Total records scraped: {len(df)}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    # Empty strings can hide as non-null -- check those too (e.g. missing descriptions)
    print("\nEmpty-string values per column:")
    empty_counts = (df.astype(str).apply(lambda col: col.str.strip() == "")).sum()
    print(empty_counts.to_string())

    dupe_upcs = df["upc"].duplicated().sum()
    print(f"\nDuplicate UPC values: {dupe_upcs}")
    if dupe_upcs > 0:
        print("Sample duplicated UPCs:")
        print(df[df["upc"].duplicated(keep=False)].sort_values("upc")["upc"].unique()[:10])


if __name__ == "__main__":
    main()
