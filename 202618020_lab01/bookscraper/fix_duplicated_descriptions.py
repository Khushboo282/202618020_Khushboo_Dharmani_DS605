

import pandas as pd

IN_PATH = "data/clean/books_clean.csv"
OUT_PATH = "data/clean/books_clean.csv"  # overwrite in place; change if you want a separate file


def fix_duplicated_description(desc: str, probe_len: int = 40, min_len: int = 60):
    """
    Detects the 'truncated-preview + full-copy' duplication bug.

    Heuristic: take the opening `probe_len` characters of the description and
    check whether that exact same opening re-appears later in the string. If
    it does, the text from that later occurrence onward is the real, complete
    description -- keep only that part.

    Returns (clean_text, was_fixed: bool)
    """
    if not isinstance(desc, str):
        return desc, False

    desc = desc.strip()
    n = len(desc)
    if n < min_len:
        return desc, False

    opening = desc[:probe_len]
    if len(opening) < probe_len:
        return desc, False

    # search for the opening snippet again, starting after it first appears
    second_occurrence = desc.find(opening, probe_len)
    if second_occurrence == -1:
        return desc, False

    cleaned = desc[second_occurrence:].strip()
    return cleaned, True


def main():
    df = pd.read_csv(IN_PATH)

    fixed_count = 0
    fixed_titles = []

    def apply_fix(row):
        nonlocal fixed_count
        cleaned, was_fixed = fix_duplicated_description(row["description"])
        if was_fixed:
            fixed_count += 1
            fixed_titles.append(row["title"])
        return cleaned

    df["description"] = df.apply(apply_fix, axis=1)

    # recompute word count from the cleaned description for every row
    # (cheap to just redo it for all rows, keeps things consistent)
    df["description_word_count"] = df["description"].fillna("").apply(
        lambda d: len(d.split())
    )

    df.to_csv(OUT_PATH, index=False)

    print(f"Rows checked: {len(df)}")
    print(f"Rows with duplicated description text found & fixed: {fixed_count}")
    if fixed_titles:
        print("Fixed rows:")
        for t in fixed_titles:
            print(f"  - {t}")
    print(f"Saved corrected file to: {OUT_PATH}")


if __name__ == "__main__":
    main()