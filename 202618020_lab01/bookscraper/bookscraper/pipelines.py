class CleanTextPipeline:
    """
    Light touch cleanup at scrape time -- collapse whitespace on string
    fields. Deeper cleaning (price parsing, rating mapping, dedup, etc.)
    happens in Task 2's preprocessing script, on purpose, so the raw
    export still reflects what was actually scraped.
    """

    def process_item(self, item, spider):
        for field in ("title", "category", "description", "availability"):
            value = item.get(field)
            if isinstance(value, str):
                item[field] = " ".join(value.split())
        return item
