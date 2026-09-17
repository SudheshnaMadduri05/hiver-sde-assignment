import pandas as pd

DATA_PATH = "data/twcs.csv"

print("Reading dataset sample...\n")

# Read only the first 100,000 rows
df = pd.read_csv(
    DATA_PATH,
    nrows=100_000
)

print("Rows loaded:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nUnique authors in sample:")
print(df["author_id"].nunique())

print("\nMost common authors:")
print(df["author_id"].value_counts().head(30))