# confirm_translation_pairs.py
import pandas as pd

FILE_A = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake.csv"
FILE_B = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake_en.csv"

for f in ["malaysian_real.csv", "malaysian_real_ms_raw.csv"]:
    df = pd.read_csv(rf"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\{f}")
    print(f, "-> rows:", len(df), "columns:", list(df.columns))
    print(df.head(1).to_string())
    print()

df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

def normalize(s):
    return str(s).strip().lower()

matches = (df_a["text"].map(normalize) == df_b["text_original_ms"].map(normalize)).sum()
print(f"Exact row-i match between fake.csv[text] and fake_en.csv[text_original_ms]: "
      f"{matches}/{len(df_a)} ({matches/len(df_a)*100:.1f}%)")

# Spot check a few near-mismatches, if any, to see if it's whitespace/encoding noise
mismatches = df_a.index[df_a["text"].map(normalize) != df_b["text_original_ms"].map(normalize)]
print(f"\nMismatched rows: {len(mismatches)}")
if len(mismatches) > 0:
    i = mismatches[0]
    print(f"\nRow {i} fake.csv text:\n{df_a.loc[i, 'text'][:200]}")
    print(f"\nRow {i} fake_en.csv text_original_ms:\n{df_b.loc[i, 'text_original_ms'][:200]}")