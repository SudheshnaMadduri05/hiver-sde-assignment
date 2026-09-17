import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

INPUT_PATH = Path("data/amazon_clean_pairs.csv")

print("=" * 70)
print("DISCOVERING AMAZON SUPPORT INTENTS")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: Load cleaned conversations
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nTotal conversations: {len(df):,}")


# ---------------------------------------------------------
# STEP 2: Sample data
# ---------------------------------------------------------

# We don't need all 168k messages for initial intent discovery.
# 20,000 is enough to discover the major issue groups.

sample_size = min(20_000, len(df))

sample = df.sample(
    n=sample_size,
    random_state=42
).copy()

print(f"Using {len(sample):,} messages for intent discovery.")


# ---------------------------------------------------------
# STEP 3: Convert customer messages into TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)

X = vectorizer.fit_transform(
    sample["customer_text"].fillna("")
)

print("TF-IDF matrix created.")


# ---------------------------------------------------------
# STEP 4: Cluster messages
# ---------------------------------------------------------

NUMBER_OF_CLUSTERS = 10

print(f"\nCreating {NUMBER_OF_CLUSTERS} clusters...")

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

sample["cluster"] = kmeans.fit_predict(X)


# ---------------------------------------------------------
# STEP 5: Display keywords for each cluster
# ---------------------------------------------------------

terms = vectorizer.get_feature_names_out()

print("\n" + "=" * 70)
print("DISCOVERED TOPICS")
print("=" * 70)

for cluster_id in range(NUMBER_OF_CLUSTERS):

    cluster_indices = (
        sample["cluster"] == cluster_id
    )

    cluster_size = cluster_indices.sum()

    center = kmeans.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-15:][::-1]

    keywords = [
        terms[index]
        for index in top_indices
    ]

    print("\n" + "-" * 70)
    print(f"CLUSTER {cluster_id}")
    print(f"Messages: {cluster_size:,}")
    print("Keywords:")
    print(", ".join(keywords))


    # Show example customer messages
    examples = sample[
        cluster_indices
    ]["customer_text"].head(5)

    print("\nExamples:")

    for example in examples:
        print(f"  - {example}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)