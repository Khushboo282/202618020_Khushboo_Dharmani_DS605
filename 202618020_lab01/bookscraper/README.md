# DS605 Lab 1 — Task 1: Data Scraping

Scrapy spider for https://books.toscrape.com/ that crawls catalog pagination
and each book's detail page, extracting: title, category, price, rating,
availability, description, UPC, number of reviews, and product URL.

## Setup

```bash
cd bookscraper
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
```

## Run the spider

Full crawl (all ~1000 books, 50 catalog pages):
```bash
scrapy crawl books
```

Quick run, first 5 pages only (~100 books — meets the assignment minimum):
```bash
scrapy crawl books -a max_pages=5
```

Output lands in `data/raw/books_raw.csv` and `data/raw/books_raw.json`
(configured in `bookscraper/settings.py`'s `FEEDS` setting — override with
`-o path.csv` on the command line if you want a different location).

## Check the raw data report

```bash
python report_raw_stats.py
```

This prints total records scraped, missing-value counts per column, and
duplicate UPC counts — the reporting requirement from Task 1.

## Next steps (Task 2+)

Preprocessing, feature engineering, visualization, and analysis scripts
will build on `data/raw/books_raw.csv`.
