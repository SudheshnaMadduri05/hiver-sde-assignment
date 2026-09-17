import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


INPUT_FILE = "data/amazon_labeled.csv"
MODEL_FILE = "results/intent_classifier.pkl"

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


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())


# Automatically detect intent column
if "intent" in df.columns:
    INTENT_COLUMN = "intent"
elif "label" in df.columns:
    INTENT_COLUMN = "label"
elif "category" in df.columns:
    INTENT_COLUMN = "category"
else:
    raise ValueError(
        "Could not find intent column. Columns are: "
        + str(df.columns.tolist())
    )


# Automatically detect customer text column
if "customer_text" in df.columns:
    TEXT_COLUMN = "customer_text"
elif "text" in df.columns:
    TEXT_COLUMN = "text"
else:
    raise ValueError(
        "Could not find customer text column. Columns are: "
        + str(df.columns.tolist())
    )


print(f"\nUsing text column: {TEXT_COLUMN}")
print(f"Using intent column: {INTENT_COLUMN}")


# Keep only our 10 intents
df = df[df[INTENT_COLUMN].isin(VALID_INTENTS)].copy()

# Remove empty text
df = df.dropna(subset=[TEXT_COLUMN])
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print(f"\nExamples before balancing: {len(df)}")

print("\nOriginal class distribution:")
print(df[INTENT_COLUMN].value_counts())


# Balance dataset
MAX_PER_CLASS = 3000

samples = []

for intent in VALID_INTENTS:
    subset = df[df[INTENT_COLUMN] == intent]

    if len(subset) == 0:
        print(f"WARNING: No examples for {intent}")
        continue

    samples.append(
        subset.sample(
            min(len(subset), MAX_PER_CLASS),
            random_state=42
        )
    )

df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced training dataset: {len(df)}")

print("\nBalanced distribution:")
print(df[INTENT_COLUMN].value_counts())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df[INTENT_COLUMN],
    test_size=0.20,
    random_state=42,
    stratify=df[INTENT_COLUMN]
)


# ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining classifier...")

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("INTENT CLASSIFIER RESULTS")
print("================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save model
os.makedirs("results", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\n================================")
print("MODEL SAVED")
print("================================")
print(MODEL_FILE)


# Test predictions
test_messages = [
    "Where is my order?",
    "My package says delivered but I never received it",
    "I want a refund for my order",
    "My payment failed",
    "I cannot login to my account",
    "My Prime Video is not working",
    "I want to cancel my order",
    "The product I received is damaged",
]


print("\n================================")
print("TEST PREDICTIONS")
print("================================")

for message in test_messages:
    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)

    print(f"\nMessage: {message}")
    print(f"Intent: {prediction}")
    print(f"Confidence: {confidence:.2f}")