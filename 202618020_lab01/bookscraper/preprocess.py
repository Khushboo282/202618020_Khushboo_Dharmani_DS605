
import re
import pandas as pd

RAW_PATH = "data/raw/books_raw.csv"
CLEAN_PATH = "data/clean/books_clean.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_description(text):
    """
    books.toscrape.com bakes a truncated 'read more' preview AND the
    full description into the same paragraph, so the same sentence
    often appears twice back-to-back (first copy sometimes cut off
    mid-word). Detect the repeat and keep only the full copy.
    """
    text = clean_text(text)
    if not text:
        return text
    n = len(text)
    for split_point in range(n // 2 - 20, n // 2 + 20):
        if split_point <= 0 or split_point >= n:
            continue
        first_half = text[:split_point]
        second_half = text[split_point:]
        if len(first_half) > 30 and second_half.startswith(first_half[:30]):
            return second_half.strip()
    return text


def parse_price(price_str):
    if pd.isna(price_str):
        return None
    match = re.search(r"[\d.]+", str(price_str))
    return float(match.group()) if match else None


def parse_stock(availability_str):
    if pd.isna(availability_str):
        return 0
    match = re.search(r"\((\d+)\s+available\)", str(availability_str))
    return int(match.group(1)) if match else 0


def price_band(price):
    if price is None:
        return "Unknown"
    if price < 20:
        return "Low"
    elif price < 40:
        return "Medium"
    else:
        return "High"


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} raw records")

    for col in ["title", "category", "availability"]:
        df[col] = df[col].apply(clean_text)

    df["description"] = df["description"].apply(dedupe_description)
    df["description"] = df["description"].replace("", "No description available")
    missing_desc = (df["description"] == "No description available").sum()
    print(f"Descriptions filled with placeholder: {missing_desc}")

    before = len(df)
    df = df.drop_duplicates(subset="upc", keep="first")
    print(f"Removed {before - len(df)} duplicate UPC rows")

    df["price"] = df["price"].apply(parse_price)
    df["rating"] = df["rating"].map(RATING_MAP)
    df["stock_count"] = df["availability"].apply(parse_stock)

    df["description_word_count"] = df["description"].apply(lambda d: len(d.split()))
    df["price_band"] = df["price"].apply(price_band)
    df["value_score"] = (df["rating"] * 20) - df["price"]

    import os
    os.makedirs("data/clean", exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved {len(df)} cleaned records to {CLEAN_PATH}")
    print("\nSample:")
    print(df[["title", "category", "price", "rating", "stock_count",
              "price_band", "value_score", "description_word_count"]].head(5).to_string())


if __name__ == "__main__":
    main()