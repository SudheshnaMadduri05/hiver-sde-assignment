import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path("data/amazon_clean_pairs.csv")
OUTPUT_PATH = Path("data/amazon_labeled.csv")


print("=" * 70)
print("CREATING AMAZON INTENT LABELS")
print("=" * 70)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nTotal conversations: {len(df):,}")


# ---------------------------------------------------------
# Intent rules
# ---------------------------------------------------------

INTENT_RULES = {

    "REFUND_RETURN": [
        r"\brefund\b",
        r"\brefunded\b",
        r"\breturn\b",
        r"\breturned\b",
        r"\bmoney back\b",
        r"\breimburse\b"
    ],

    "CANCELLATION": [
        r"\bcancel\b",
        r"\bcancelled\b",
        r"\bcanceled\b",
        r"\bcancellation\b"
    ],

    "PAYMENT_PROBLEM": [
        r"\bpayment\b",
        r"\bamazon pay\b",
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bdebit\b",
        r"\bcredit card\b",
        r"\bcard\b",
        r"\bpaid\b"
    ],

    "ACCOUNT_PROBLEM": [
        r"\baccount\b",
        r"\bpassword\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\bsign in\b",
        r"\bverification\b",
        r"\blocked\b"
    ],

    "PRIME_SERVICE": [
        r"\bprime\b",
        r"\bmembership\b",
        r"\bsubscribe\b",
        r"\bunsubscribe\b",
        r"\bsubscription\b"
    ],

    "DIGITAL_SERVICE": [
        r"\bprime video\b",
        r"\bvideo\b",
        r"\bstream\b",
        r"\bstreaming\b",
        r"\bapp\b",
        r"\bfire tv\b",
        r"\bkindle\b",
        r"\baudible\b"
    ],

    "PRODUCT_PROBLEM": [
        r"\bdamaged\b",
        r"\bdamage\b",
        r"\bdefective\b",
        r"\bbroken\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b",
        r"\bwrong item\b",
        r"\bwrong product\b",
        r"\bpoor quality\b",
        r"\bquality\b"
    ],

    "DELIVERY_PROBLEM": [
        r"\bdelivered\b",
        r"\bdelivery\b",
        r"\bcourier\b",
        r"\bcarrier\b",
        r"\bdriver\b",
        r"\bpackage\b",
        r"\bparcel\b",
        r"\bmissing\b",
        r"\bnot received\b"
    ],

    "DELIVERY_DELAY": [
        r"\blate\b",
        r"\blater\b",
        r"\bdelay\b",
        r"\bdelayed\b",
        r"\bwaiting\b",
        r"\bwait\b",
        r"\boverdue\b",
        r"\bdelivery date\b",
        r"\bshipping delay\b"
    ],

    "ORDER_STATUS": [
        r"\border\b",
        r"\btracking\b",
        r"\btrack\b",
        r"\bwhere is\b",
        r"\bwhen will\b",
        r"\bshipping\b",
        r"\bexpected delivery\b"
    ]
}


# ---------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------

def classify_intent(text):

    text = str(text).lower()

    scores = {}

    for intent, patterns in INTENT_RULES.items():

        score = 0

        for pattern in patterns:

            matches = re.findall(pattern, text)

            score += len(matches)

        scores[intent] = score


    # No matching intent
    if max(scores.values()) == 0:
        return "OTHER"


    # Highest scoring intent
    return max(
        scores,
        key=scores.get
    )


# ---------------------------------------------------------
# Apply labels
# ---------------------------------------------------------

print("\nAssigning intents...")

df["intent"] = df["customer_text"].apply(
    classify_intent
)


# ---------------------------------------------------------
# Display distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INTENT DISTRIBUTION")
print("=" * 70)

counts = df["intent"].value_counts()

print(counts)

print("\nPercentages:")

print(
    (df["intent"].value_counts(normalize=True) * 100)
    .round(2)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("DATASET SAVED")
print("=" * 70)

print(f"\nFile: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")

print("\nDONE.")
