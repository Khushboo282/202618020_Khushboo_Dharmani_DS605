import scrapy


class BookItem(scrapy.Item):
    """
    One record per book. Fields required by DS605 Lab Assignment 1:
    title, category, price, rating, availability, product description,
    UPC, number of reviews, and product URL.
    """
    title = scrapy.Field()
    category = scrapy.Field()
    price = scrapy.Field()              # raw string, e.g. "£51.77" -- cleaned in Task 2
    rating = scrapy.Field()             # raw word, e.g. "Three" -- mapped to int in Task 2
    availability = scrapy.Field()       # raw text, e.g. "In stock (22 available)"
    description = scrapy.Field()
    upc = scrapy.Field()
    num_reviews = scrapy.Field()
    product_url = scrapy.Field()
