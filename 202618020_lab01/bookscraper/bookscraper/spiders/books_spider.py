import scrapy

from bookscraper.items import BookItem


class BooksSpider(scrapy.Spider):
    """
    Spider for https://books.toscrape.com/

    Crawls catalog pages (paginated 20-books-per-page), follows each
    book's detail-page link, and scrapes the full record from there
    (the detail page carries every field the assignment asks for, so
    we don't need to scrape the catalog cards separately).

    Usage:
        scrapy crawl books
        scrapy crawl books -a max_pages=5      # limit to first 5 catalog pages
        scrapy crawl books -o data/raw/books_raw.csv   # override output path
    """

    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    def __init__(self, max_pages=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional cap so you can quickly test on 5 pages (~100 books)
        # before running a full crawl of all 50 pages (~1000 books).
        self.max_pages = int(max_pages) if max_pages else None
        self.pages_scraped = 0

    def parse(self, response):
        self.pages_scraped += 1

        book_links = response.css("article.product_pod h3 a::attr(href)").getall()
        for link in book_links:
            book_url = response.urljoin(link)
            yield response.follow(book_url, callback=self.parse_book)

        if self.max_pages and self.pages_scraped >= self.max_pages:
            return

        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):
        item = BookItem()

        item["title"] = response.css("div.product_main h1::text").get(default="").strip()

        # Breadcrumb: Home / Books / <real category> / <book title>
        item["category"] = response.xpath(
            '//ul[@class="breadcrumb"]/li[3]/a/text()'
        ).get(default="").strip()

        item["price"] = response.css("div.product_main p.price_color::text").get(default="").strip()

        rating_class = response.css("div.product_main p.star-rating::attr(class)").get(default="")
        # class looks like "star-rating Three" -- second token is the word rating
        item["rating"] = rating_class.replace("star-rating", "").strip()

        # Scoped to div.product_main so we don't also sweep up "In stock" text
        # from the "you may also like" related-books carousel further down the page.
        availability_text = " ".join(
            t.strip() for t in response.css("div.product_main p.availability::text").getall() if t.strip()
        )
        item["availability"] = availability_text

        description = response.xpath(
            '//div[@id="product_description"]/following-sibling::p[1]/text()'
        ).get()
        item["description"] = description.strip() if description else ""

        item["upc"] = response.xpath(
            '//table//tr[th="UPC"]/td/text()'
        ).get(default="").strip()

        item["num_reviews"] = response.xpath(
            '//table//tr[th="Number of reviews"]/td/text()'
        ).get(default="").strip()

        item["product_url"] = response.url

        yield item