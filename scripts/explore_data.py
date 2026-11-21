"""Quick look at the raw data before writing data.py."""
import pandas as pd

df = pd.read_parquet("data/raw/GBG500.parquet")
labels = pd.read_csv("data/raw/GBG500_labels.csv")

# 1. Structure: how many rows/columns, what types
print("=== GBG500.parquet ===")
print("shape:", df.shape)
print("index:", df.index.names)
print(df.dtypes, "\n")
print(df.head(10), "\n")

# 2. Value ranges: helps check units and the need to normalise
print(df.describe().T, "\n")

# 3. Labels: which column links a ride to its class
print("=== GBG500_labels.csv ===")
print(labels.shape)
print(labels.head(10), "\n")
for col in labels.columns:
    if labels[col].nunique() < 20:
        print(labels[col].value_counts(), "\n")

# 4. Find the ride id column: should have ~500 unique values
print("=== Candidate ride id columns ===")
for col in df.columns:
    n = df[col].nunique()
    if 100 <= n <= 1000:
        sizes = df[col].value_counts()
        print(f"{col}: {n} rides, rows per ride min={sizes.min()} median={sizes.median()} max={sizes.max()}")