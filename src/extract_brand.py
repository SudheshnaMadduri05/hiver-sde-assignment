import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/twcs.csv")
OUTPUT_PATH = Path("data/amazon_support.csv")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


# --------------------------------------------------
# STEP 1: Find all AmazonHelp tweet IDs
# --------------------------------------------------

print("=" * 60)
print("STEP 1: Finding AmazonHelp tweets")
print("=" * 60)

amazon_tweet_ids = set()

total_rows = 0
brand_rows = 0

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    brand_chunk = chunk[chunk["author_id"] == BRAND]

    brand_rows += len(brand_chunk)

    amazon_tweet_ids.update(
        brand_chunk["tweet_id"].astype(str)
    )

    print(
        f"Processed {total_rows:,} rows | "
        f"AmazonHelp tweets found: {brand_rows:,}"
    )


print("\nTotal AmazonHelp tweets:", len(amazon_tweet_ids))


# --------------------------------------------------
# STEP 2: Extract AmazonHelp tweets + customers
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: Extracting conversations")
print("=" * 60)

selected_chunks = []

total_rows = 0

for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    total_rows += len(chunk)

    # AmazonHelp's own tweets
    brand_messages = chunk[
        chunk["author_id"] == BRAND
    ]

    # Customer messages directly replying to AmazonHelp
    customer_messages = chunk[
        chunk["in_response_to_tweet_id"]
        .astype(str)
        .isin(amazon_tweet_ids)
    ]

    selected = pd.concat(
        [brand_messages, customer_messages],
        ignore_index=True
    )

    if not selected.empty:
        selected_chunks.append(selected)

    print(
        f"Processed {total_rows:,} rows | "
        f"Selected: {sum(len(x) for x in selected_chunks):,}"
    )


# --------------------------------------------------
# STEP 3: Combine and remove duplicates
# --------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: Creating final dataset")
print("=" * 60)

amazon_df = pd.concat(
    selected_chunks,
    ignore_index=True
)

amazon_df = amazon_df.drop_duplicates(
    subset=["tweet_id"]
)

# Sort chronologically
amazon_df["created_at"] = pd.to_datetime(
    amazon_df["created_at"],
    errors="coerce"
)

amazon_df = amazon_df.sort_values(
    "created_at"
)

# Save
amazon_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDONE!")

print("Final rows:", len(amazon_df))
print("Output:", OUTPUT_PATH)

print("\nAuthor distribution:")
print(
    amazon_df["author_id"]
    .value_counts()
    .head(10)
)

print("\nInbound distribution:")
print(
    amazon_df["inbound"]
    .value_counts()
)

print("\nFirst 10 rows:")
print(
    amazon_df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "in_response_to_tweet_id"
        ]
    ].head(10).to_string()
)