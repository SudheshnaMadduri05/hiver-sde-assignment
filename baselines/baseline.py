import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INPUT_FILE = "data/amazon_labeled.csv"
RESULT_FILE = "results/baseline_results.txt"

VALID_INTENTS = [
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

# Detect columns
text_col = "customer_text" if "customer_text" in df.columns else "text"
intent_col = "intent" if "intent" in df.columns else "label"

df = df[df[intent_col].isin(VALID_INTENTS)].copy()
df = df.dropna(subset=[text_col])

df[text_col] = df[text_col].astype(str)

# Balance
MAX_PER_CLASS = 2000

samples = []

for intent in VALID_INTENTS:
    subset = df[df[intent_col] == intent]

    if len(subset) > 0:
        samples.append(
            subset.sample(
                min(len(subset), MAX_PER_CLASS),
                random_state=42
            )
        )

df = pd.concat(samples).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(
    df[text_col],
    df[intent_col],
    test_size=0.20,
    random_state=42,
    stratify=df[intent_col]
)


# =========================================================
# BASELINE 1
# Majority Class
# =========================================================

majority_class = y_train.value_counts().idxmax()

baseline1_predictions = [
    majority_class
] * len(y_test)

baseline1_accuracy = accuracy_score(
    y_test,
    baseline1_predictions
)


# =========================================================
# BASELINE 2
# TF-IDF + Logistic Regression
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_vec,
    y_train
)

baseline2_predictions = classifier.predict(
    X_test_vec
)

baseline2_accuracy = accuracy_score(
    y_test,
    baseline2_predictions
)


# =========================================================
# RESULTS
# =========================================================

print("\n====================================")
print("BASELINE RESULTS")
print("====================================")

print(
    f"\nBaseline 1 - Majority Class Accuracy: "
    f"{baseline1_accuracy:.4f}"
)

print(
    f"Baseline 2 - TF-IDF + Logistic Regression Accuracy: "
    f"{baseline2_accuracy:.4f}"
)

print("\nBaseline 2 Classification Report:")
print(
    classification_report(
        y_test,
        baseline2_predictions,
        zero_division=0
    )
)


os.makedirs("results", exist_ok=True)

with open(RESULT_FILE, "w", encoding="utf-8") as f:

    f.write("BASELINE RESULTS\n")
    f.write("============================\n\n")

    f.write(
        f"Baseline 1 - Majority Class Accuracy: "
        f"{baseline1_accuracy:.4f}\n"
    )

    f.write(
        f"Baseline 2 - TF-IDF + Logistic Regression Accuracy: "
        f"{baseline2_accuracy:.4f}\n\n"
    )

    f.write("Baseline 2 Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            baseline2_predictions,
            zero_division=0
        )
    )

print(f"\nSaved results to {RESULT_FILE}")