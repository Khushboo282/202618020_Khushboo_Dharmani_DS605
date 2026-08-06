BOT_NAME = "bookscraper"

SPIDER_MODULES = ["bookscraper.spiders"]
NEWSPIDER_MODULE = "bookscraper.spiders"

# Be a polite scraper -- books.toscrape.com is a public sandbox site,
# but good practice matters for the assignment/grading too.
ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 0.5
CONCURRENT_REQUESTS_PER_DOMAIN = 4
USER_AGENT = "ds605-lab1-bookscraper (+educational assignment; DA-IICT)"

# Keep logs readable
LOG_LEVEL = "INFO"

ITEM_PIPELINES = {
    "bookscraper.pipelines.CleanTextPipeline": 300,
}

# Default feed export -- can be overridden from the command line with -o
FEEDS = {
    "data/raw/books_raw.csv": {
        "format": "csv",
        "overwrite": True,
    },
    "data/raw/books_raw.json": {
        "format": "json",
        "overwrite": True,
        "encoding": "utf8",
    },
}

REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
