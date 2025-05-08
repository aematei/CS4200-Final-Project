# Test script to verify dataset
import pandas as pd

df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None,
    names=['target', 'id', 'date', 'flag', 'user', 'text']
)
print(f"Dataset loaded successfully with {len(df)} rows")
print("\nFirst few rows:")
print(df.head())