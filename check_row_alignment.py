# check_row_alignment.py
import pandas as pd

FILE_A = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake.csv"
FILE_B = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake_en.csv"

df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

print("Columns in fake.csv:   ", list(df_a.columns))
print("Columns in fake_en.csv:", list(df_b.columns))
print()

# Print first 3 rows side by side (all non-text columns) to check for
# shared IDs, dates, source URLs, or titles that would confirm 1:1 pairing
non_text_cols_a = [c for c in df_a.columns if c != "text"]
non_text_cols_b = [c for c in df_b.columns if c != "text"]

print("First 3 rows, fake.csv (non-text columns):")
print(df_a[non_text_cols_a].head(3).to_string())
print("\nFirst 3 rows, fake_en.csv (non-text columns):")
print(df_b[non_text_cols_b].head(3).to_string())

# Rough length-ratio check: if row i is a translation of row i, the
# translated text should track the native text's length loosely
# (not exact, but roughly proportional across the whole set)
if len(df_a) == len(df_b):
    len_a = df_a["text"].str.len()
    len_b = df_b["text"].str.len()
    corr = len_a.corr(len_b)
    print(f"\nRow-wise text-length correlation (fake.csv vs fake_en.csv): {corr:.3f}")
    print("(a high positive correlation here is consistent with row-aligned translation pairs)")