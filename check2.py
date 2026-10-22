import pandas as pd, re

fake_df = pd.read_csv("data/malaysian/malaysian_fake_en.csv")
def has_marker(t):
    return bool(re.search(r"according to,|truth:", str(t).lower()))

print("raw malaysian_fake_en.csv:", fake_df["text"].apply(has_marker).mean())

manifest = pd.read_csv("malaysian_val_manifest.csv")
fake_manifest = manifest[manifest["label"] == 0]
print("val manifest fake rows:", fake_manifest["text"].apply(has_marker).mean())