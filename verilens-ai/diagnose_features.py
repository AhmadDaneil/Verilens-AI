"""
Diagnostic: isolate whether the engineered features (the 6-feature vector
fed alongside DistilBERT+BiLSTM) or the text pathway is driving a
false-positive FAKE prediction on a known-real article.

Run this from inside the verilens-ai project, in the activated venv:
    python diagnose_features.py
"""
import torch
import torch.nn.functional as F

from model import load_model
from tokenizer import Tokenizer
from features import clean_text, get_features_tensor

MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cpu")

ARTICLE = """KUALA LUMPUR: Former Malaysian Anti-Corruption Commission (MACC) chief commissioner Tan Sri Azam Baki has distanced himself from the RM178.29 million contract awarded to a company linked to his brother, saying he has no knowledge of the matter.
Responding to an article published by The Edge Malaysia, Azam said his brother's business affairs had nothing to do with him. It was reported that Integriti Padu Sdn Bhd had secured a six-year contract to develop and operate the Road Charge and Vehicle Entry Permit (RC/VEP) system at the Malaysia-Singapore border. "I don't know anything about this. It is his business. "So does this mean whoever is associated with me is going to be questioned? "I have retired and it is up to the Transport Ministry to award the project to his company," he told the New Straits Times. Azam, who retired as MACC chief commissioner on May 12, also defended his family, saying they should not come under scrutiny solely because of their relationship with him. "My family's DNA is all good. I have a sister who is a well-known ear, nose and throat specialist in this country."Does this mean she will also be put under the spotlight? Whoever wants to be safe, please don't get close to me," he said. According to The Edge, the Transport Ministry awarded the RM178.29 million contract to Integriti Padu on May 26 following a request for proposals issued in June last year. The report said Integriti Padu's executive chairman is Datuk Nasir Baki, Azam's brother. The company is tasked with developing, operating and maintaining the RC/VEP system for six years. Azam's tenure as MACC chief commissioner was previously overshadowed by a controversy over his shareholdings after allegations surfaced in late 2021 that he held millions of shares and warrants in several public-listed companies while serving as a senior civil servant.In January 2022, Azam denied any wrongdoing, saying the shares had been purchased by his brother using his trading account. Earlier this year, Azam again came under scrutiny following Bloomberg reports alleging irregularities involving his shareholdings in a public-listed company while he was MACC chief. He has since filed a RM100 million defamation suit against Bloomberg over the reports."""

print("Loading model...")
model = load_model(MODEL_PATH, DEVICE)
tok = Tokenizer(max_length=128)

cleaned = clean_text(ARTICLE)
tokens = tok.tokenize(cleaned)
input_ids = tokens["input_ids"].to(DEVICE)
attn_mask = tokens["attention_mask"].to(DEVICE)
features = get_features_tensor(cleaned).to(DEVICE)

print("\n=== Raw engineered feature values ===")
print(features)
print(f"Shape: {features.shape}")

def run(name, feat_tensor):
    with torch.no_grad():
        logits = model(input_ids, attn_mask, feat_tensor)
        probs = F.softmax(logits, dim=1)[0]
    fake_score = float(probs[0])
    print(f"{name:35s} fake_score={fake_score:.4f}")
    return fake_score

print("\n=== Ablation: real features vs. zeroed-out features ===")
baseline = run("Real engineered features", features)
zeroed = run("Zeroed engineered features", torch.zeros_like(features))

print(f"\nDelta (real - zeroed): {baseline - zeroed:+.4f}")
if abs(baseline - zeroed) > 0.15:
    print(">>> Large swing: engineered features are a major driver of this prediction.")
else:
    print(">>> Small swing: engineered features are NOT the main driver — look at the text pathway instead.")

print("\n=== Per-feature zero-out (which single feature matters most?) ===")
for i in range(features.shape[-1]):
    modified = features.clone()
    modified[..., i] = 0.0
    score = run(f"Feature[{i}] zeroed (value was {features.flatten()[i].item():.4f})", modified)
    print(f"    -> shift from baseline: {score - baseline:+.4f}")