import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/amazon_pairs.csv")
OUTPUT_PATH = Path("data/amazon_clean_pairs.csv")

print("=" * 70)
print("AMAZONHELP DATASET ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: Load conversation pairs
# ---------------------------------------------------------

print("\nLoading conversation pairs...")

df = pd.read_csv(INPUT_PATH)

print(f"Total conversation pairs: {len(df):,}")


# ---------------------------------------------------------
# STEP 2: Basic cleaning
# ---------------------------------------------------------

print("\nCleaning data...")

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["brand_response"] = (
    df["brand_response"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove very short messages
df = df[
    (df["customer_text"].str.len() >= 15) &
    (df["brand_response"].str.len() >= 10)
].copy()

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_text"]
)

print(f"Useful unique conversations: {len(df):,}")


# ---------------------------------------------------------
# STEP 3: Message statistics
# ---------------------------------------------------------

df["customer_length"] = df["customer_text"].str.len()
df["response_length"] = df["brand_response"].str.len()

print("\n" + "=" * 70)
print("MESSAGE STATISTICS")
print("=" * 70)

print("\nCustomer message length:")
print(df["customer_length"].describe())

print("\nAmazonHelp response length:")
print(df["response_length"].describe())


# ---------------------------------------------------------
# STEP 4: Show customer messages
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CUSTOMER MESSAGE EXAMPLES")
print("=" * 70)

for i, text in enumerate(
    df["customer_text"].head(100),
    start=1
):
    print(f"\n{i}. {text}")


# ---------------------------------------------------------
# STEP 5: Save cleaned dataset
# ---------------------------------------------------------

df[
    [
        "customer_tweet_id",
        "brand_tweet_id",
        "customer_text",
        "brand_response",
        "customer_created_at",
        "brand_created_at"
    ]
].to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("CLEAN DATASET SAVED")
print("=" * 70)

print(f"\nFile: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")

print("\nDONE.")