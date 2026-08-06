

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os

IN_PATH = "data/clean/books_clean.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    df = pd.read_csv(IN_PATH)

    # ---------- Plot 1: Price distribution ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price (£)")
    ax.set_ylabel("Number of Books")
    save(fig, "price_distribution.png")

    # ---------- Plot 2: Rating distribution ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = df["rating"].value_counts().sort_index()
    ax.bar(rating_counts.index.astype(str), rating_counts.values, color="#55A868")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating (1-5)")
    ax.set_ylabel("Number of Books")
    save(fig, "rating_distribution.png")

    # ---------- Plot 3: Average price by category ----------
    fig, ax = plt.subplots(figsize=(10, 8))
    avg_price = df.groupby("category")["price"].mean().sort_values()
    ax.barh(avg_price.index, avg_price.values, color="#C44E52")
    ax.set_title("Average Price by Category")
    ax.set_xlabel("Average Price (£)")
    save(fig, "avg_price_by_category.png")

    # ---------- Plot 4: Relationship plot - price vs rating ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["rating"], df["price"], alpha=0.5, color="#8172B2")
    ax.set_title("Price vs Rating")
    ax.set_xlabel("Rating (1-5)")
    ax.set_ylabel("Price (£)")
    save(fig, "price_vs_rating.png")

    # ---------- Word cloud from descriptions ----------
    text = " ".join(df["description"].dropna().astype(str).tolist())
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        stopwords=STOPWORDS,
        collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud - Book Descriptions")
    save(fig, "description_wordcloud.png")

    # ---------- Summary stats ----------
    print("\n=== SUMMARY STATS ===")
    print(f"Total books: {len(df)}")
    print(f"Price range: £{df['price'].min():.2f} - £{df['price'].max():.2f}")
    print(f"Average price: £{df['price'].mean():.2f}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Correlation (price vs rating): {df['price'].corr(df['rating']):.3f}")
    print("\nTop 5 most expensive categories (avg price):")
    print(avg_price.sort_values(ascending=False).head(5).round(2))
    print("\nTop 5 cheapest categories (avg price):")
    print(avg_price.sort_values().head(5).round(2))
    print("\nCategory counts (most represented):")
    print(df["category"].value_counts().head(5))
    print("\nStock count summary:")
    print(df["stock_count"].describe())


if __name__ == "__main__":
    main()