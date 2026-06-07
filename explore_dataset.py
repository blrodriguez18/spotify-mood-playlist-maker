import pandas as pd

df = pd.read_csv("dataset.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nSample row:\n{df.iloc[0]}")
print(f"\nUnique genres ({df['track_genre'].nunique()}):")
print(sorted(df['track_genre'].unique()))