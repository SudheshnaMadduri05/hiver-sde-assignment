import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/twcs.csv")
OUTPUT_PATH = Path("data/amazon_pairs.csv")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


def clean_id(value):
    """Convert tweet IDs like 272.0 into 272."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


print("=" * 70)
print("BUILDING AMAZONHELP CONVERSATION PAIRS")
print("=" * 70)

columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

# ---------------------------------------------------------
# STEP 1: Build tweet lookup
# ---------------------------------------------------------

print("\nSTEP 1: Building tweet lookup...")

tweet_lookup = {}

rows_read = 0

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=columns,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    rows_read += len(chunk)

    for row in chunk.itertuples(index=False):

        tweet_id = clean_id(row.tweet_id)

        if tweet_id is None:
            continue

        tweet_lookup[tweet_id] = {
            "author_id": str(row.author_id),
            "text": str(row.text),
            "created_at": row.created_at,
            "inbound": row.inbound
        }

    print(f"Rows processed: {rows_read:,}")

print("\nTotal tweets in lookup:", len(tweet_lookup))


# ---------------------------------------------------------
# STEP 2: Find AmazonHelp responses
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: FINDING AMAZONHELP CONVERSATIONS")
print("=" * 70)

pairs = []

rows_read = 0
amazon_count = 0
parent_found = 0

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=columns,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    rows_read += len(chunk)

    amazon_rows = chunk[
        chunk["author_id"].astype(str).str.strip() == BRAND
    ]

    amazon_count += len(amazon_rows)

    for row in amazon_rows.itertuples(index=False):

        parent_id = clean_id(row.in_response_to_tweet_id)

        if parent_id is None:
            continue

        parent = tweet_lookup.get(parent_id)

        if parent is None:
            continue

        parent_found += 1

        # Ignore AmazonHelp replying to itself
        if parent["author_id"] == BRAND:
            continue

        customer_text = parent["text"]
        brand_text = str(row.text)

        if customer_text == "nan":
            continue

        if brand_text == "nan":
            continue

        if len(customer_text.strip()) < 5:
            continue

        if len(brand_text.strip()) < 5:
            continue

        pairs.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": clean_id(row.tweet_id),
            "customer_text": customer_text,
            "brand_response": brand_text,
            "customer_created_at": parent["created_at"],
            "brand_created_at": row.created_at
        })

    print(
        f"Rows processed: {rows_read:,} | "
        f"AmazonHelp: {amazon_count:,} | "
        f"Pairs: {len(pairs):,}"
    )


# ---------------------------------------------------------
# STEP 3: Save
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: SAVING DATASET")
print("=" * 70)

print(f"\nAmazonHelp tweets found: {amazon_count:,}")
print(f"Parent tweets found: {parent_found:,}")
print(f"Conversation pairs: {len(pairs):,}")

pairs_df = pd.DataFrame(pairs)

if pairs_df.empty:

    print("\nERROR: Still no pairs found.")
    print("The ID matching needs further investigation.")

    raise SystemExit


pairs_df = pairs_df.drop_duplicates(
    subset=["customer_tweet_id", "brand_tweet_id"]
)

print(f"Unique conversation pairs: {len(pairs_df):,}")


pairs_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_PATH)


# ---------------------------------------------------------
# STEP 4: Show examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE CONVERSATIONS")
print("=" * 70)

for i, row in pairs_df.head(10).iterrows():

    print(f"\nExample {i + 1}")
    print("-" * 70)

    print("CUSTOMER:")
    print(row["customer_text"])

    print("\nAMAZONHELP:")
    print(row["brand_response"])


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)