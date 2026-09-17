import pandas as pd
from scipy.stats import spearmanr

# --------------------------------------------------
# Human vs LLM-as-Judge Agreement
# --------------------------------------------------

# Put your manually assigned human scores here.
# Score each response from 1 to 5.
#
# 1 = Very poor
# 2 = Poor
# 3 = Acceptable
# 4 = Good
# 5 = Excellent

human_scores = [
    4, 5, 4, 3, 4,
    5, 4, 4, 3, 5,
    4, 3, 5, 4, 4,
    5, 4, 3, 4, 5,
    4, 5, 4, 3, 4,
    5, 4, 4, 5, 3
]

# Replace these with the actual scores produced
# by your LLM judge for the SAME 30 examples.
llm_scores = [
    4, 5, 5, 3, 4,
    5, 4, 4, 3, 5,
    4, 3, 5, 4, 5,
    5, 4, 3, 4, 5,
    4, 5, 4, 3, 4,
    5, 4, 4, 5, 3
]

# Check both lists contain the same number of examples
assert len(human_scores) == len(llm_scores)

# Calculate Spearman correlation
correlation, p_value = spearmanr(
    human_scores,
    llm_scores
)

# Calculate average scores
human_average = sum(human_scores) / len(human_scores)
llm_average = sum(llm_scores) / len(llm_scores)

print("=" * 50)
print("HUMAN–LLM JUDGE AGREEMENT")
print("=" * 50)

print(f"Number of examples: {len(human_scores)}")
print(f"Human average score: {human_average:.2f}/5")
print(f"LLM judge average score: {llm_average:.2f}/5")
print(f"Spearman correlation: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

print("=" * 50)

# Save result
with open("results/human_llm_agreement.txt", "w") as f:
    f.write("HUMAN–LLM JUDGE AGREEMENT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Number of examples: {len(human_scores)}\n")
    f.write(f"Human average score: {human_average:.2f}/5\n")
    f.write(f"LLM judge average score: {llm_average:.2f}/5\n")
    f.write(f"Spearman correlation: {correlation:.2f}\n")
    f.write(f"P-value: {p_value:.4f}\n")

print("Saved to results/human_llm_agreement.txt")