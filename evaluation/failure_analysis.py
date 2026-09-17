import pandas as pd
import os

ERROR_FILE = "results/golden_errors.csv"
OUTPUT_FILE = "results/top_5_failures.txt"

df = pd.read_csv(ERROR_FILE)

if len(df) == 0:
    print("No errors found.")
    exit()

print("Total errors:", len(df))

# Most common confusion pairs
confusions = (
    df.groupby(["gold_intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\n===================================")
print("TOP CONFUSION PAIRS")
print("===================================")

print(confusions.head(10).to_string(index=False))


# Save top 5 failure types
top5 = confusions.head(5)

os.makedirs("results", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("TOP 5 FAILURE ANALYSIS\n")
    f.write("============================\n\n")

    for i, row in top5.iterrows():

        gold = row["gold_intent"]
        predicted = row["predicted_intent"]
        count = row["count"]

        f.write(
            f"{len(top5)}. GOLD={gold} -> "
            f"PREDICTED={predicted} | COUNT={count}\n"
        )

        examples = df[
            (df["gold_intent"] == gold) &
            (df["predicted_intent"] == predicted)
        ].head(3)

        for _, example in examples.iterrows():

            f.write(
                f"Example: {example['customer_text']}\n"
            )

        f.write("\n")

print(f"\nSaved: {OUTPUT_FILE}")


# Print real examples
print("\n===================================")
print("REAL FAILURE EXAMPLES")
print("===================================")

for _, row in df.head(10).iterrows():

    print("\nCustomer:")
    print(row["customer_text"])

    print("Expected:")
    print(row["gold_intent"])

    print("Predicted:")
    print(row["predicted_intent"])