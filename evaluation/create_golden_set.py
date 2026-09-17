import pandas as pd
import os

INPUT_FILE = "data/amazon_labeled.csv"
OUTPUT_FILE = "evaluation/golden_set.csv"

INTENTS = [
    "ORDER_STATUS",
    "DELIVERY_DELAY",
    "DELIVERY_PROBLEM",
    "REFUND_RETURN",
    "CANCELLATION",
    "PAYMENT_PROBLEM",
    "ACCOUNT_PROBLEM",
    "PRIME_SERVICE",
    "PRODUCT_PROBLEM",
    "DIGITAL_SERVICE",
]

df = pd.read_csv(INPUT_FILE)

text_col = "customer_text" if "customer_text" in df.columns else "text"
intent_col = "intent" if "intent" in df.columns else "label"

df = df[df[intent_col].isin(INTENTS)].copy()
df = df.dropna(subset=[text_col])

# 20 examples per intent = 200 examples
samples = []

for intent in INTENTS:
    subset = df[df[intent_col] == intent]

    sample = subset.sample(
        min(20, len(subset)),
        random_state=42
    ).copy()

    sample["gold_intent"] = intent
    samples.append(sample)


golden = pd.concat(samples)

golden = golden[
    [text_col, "gold_intent"]
].rename(
    columns={text_col: "customer_text"}
)

golden.insert(
    0,
    "example_id",
    range(1, len(golden) + 1)
)

# Add blank column for manual verification
golden["verified"] = ""

os.makedirs("evaluation", exist_ok=True)

golden.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("===================================")
print("GOLDEN EVALUATION SET CREATED")
print("===================================")
print(f"Examples: {len(golden)}")
print(f"Saved: {OUTPUT_FILE}")
print()
print(golden["gold_intent"].value_counts())