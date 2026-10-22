import pandas as pd

df = pd.read_csv(r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_real.csv")
print("Full malaysian_real.csv, was_translated counts:")
print(df["was_translated"].value_counts(dropna=False))

if "extraction_method" in df.columns:
    filtered = df[df["extraction_method"] == "full_article"]
    print(f"\nAfter filtering to extraction_method == 'full_article' "
          f"({len(filtered)}/{len(df)} rows survive — this is what retrain_with_malaysian.py actually trains on):")
    print(filtered["was_translated"].value_counts(dropna=False))