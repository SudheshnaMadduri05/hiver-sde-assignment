import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

MODEL_FILE = "results/intent_classifier.pkl"
GOLDEN_FILE = "evaluation/golden_set.csv"
RESULT_FILE = "results/golden_evaluation.txt"


model = joblib.load(MODEL_FILE)

df = pd.read_csv(GOLDEN_FILE)

X = df["customer_text"]
y_true = df["gold_intent"]

y_pred = model.predict(X)

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n===================================")
print("GOLDEN SET EVALUATION")
print("===================================")

print(f"Examples: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")


# Confusion matrix
labels = sorted(y_true.unique())

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(labels)
print(cm)


# Save incorrect examples
df["predicted_intent"] = y_pred
df["correct"] = df["gold_intent"] == df["predicted_intent"]

errors = df[df["correct"] == False]

errors.to_csv(
    "results/golden_errors.csv",
    index=False
)


os.makedirs("results", exist_ok=True)

with open(RESULT_FILE, "w", encoding="utf-8") as f:

    f.write("GOLDEN SET EVALUATION\n")
    f.write("============================\n\n")

    f.write(f"Examples: {len(df)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro Precision: {precision:.4f}\n")
    f.write(f"Macro Recall: {recall:.4f}\n")
    f.write(f"Macro F1: {f1:.4f}\n")

    f.write(f"\nIncorrect examples: {len(errors)}\n")

print("\nSaved:")
print(RESULT_FILE)
print("results/golden_errors.csv")