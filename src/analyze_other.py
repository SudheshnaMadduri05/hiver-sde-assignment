import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/amazon_labeled.csv")

print("=" * 70)
print("ANALYZING OTHER INTENT")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

other = df[df["intent"] == "OTHER"].copy()

print(f"\nTotal OTHER messages: {len(other):,}")

# ---------------------------------------------------------
# Show random examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM OTHER EXAMPLES")
print("=" * 70)

examples = other.sample(
    n=min(100, len(other)),
    random_state=42
)

for i, text in enumerate(
    examples["customer_text"],
    start=1
):
    print(f"\n{i}. {text}")


# ---------------------------------------------------------
# Message length
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("OTHER MESSAGE LENGTH")
print("=" * 70)

print(
    other["customer_text"]
    .str.len()
    .describe()
)


# ---------------------------------------------------------
# Common words
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COMMON WORDS IN OTHER")
print("=" * 70)

from collections import Counter
import re

words = []

for text in other["customer_text"].astype(str):

    found = re.findall(
        r"[a-zA-Z]{3,}",
        text.lower()
    )

    words.extend(found)


counter = Counter(words)

for word, count in counter.most_common(50):

    print(
        f"{word:<25} {count:,}"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)