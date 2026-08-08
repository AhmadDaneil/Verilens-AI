"""
Length-sensitivity stress test.

Takes the same real, known-genuine article and truncates it to different
lengths, re-running prediction each time. If fake_score climbs as length
increases -- independent of content -- that confirms feature[1] (or whichever
index proved dominant in diagnose_features.py) is a raw, unnormalized
length signal the model is leaning on rather than genuine content analysis.

Run from inside verilens-ai, in the activated venv:
    python length_stress_test.py
"""
import torch
import torch.nn.functional as F

from model import load_model
from tokenizer import Tokenizer
from features import clean_text, get_features_tensor

MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cpu")

FULL_ARTICLE = """KUALA LUMPUR: Former Malaysian Anti-Corruption Commission (MACC) chief commissioner Tan Sri Azam Baki has distanced himself from the RM178.29 million contract awarded to a company linked to his brother, saying he has no knowledge of the matter.
Responding to an article published by The Edge Malaysia, Azam said his brother's business affairs had nothing to do with him. It was reported that Integriti Padu Sdn Bhd had secured a six-year contract to develop and operate the Road Charge and Vehicle Entry Permit (RC/VEP) system at the Malaysia-Singapore border. "I don't know anything about this. It is his business. "So does this mean whoever is associated with me is going to be questioned? "I have retired and it is up to the Transport Ministry to award the project to his company," he told the New Straits Times. Azam, who retired as MACC chief commissioner on May 12, also defended his family, saying they should not come under scrutiny solely because of their relationship with him. "My family's DNA is all good. I have a sister who is a well-known ear, nose and throat specialist in this country."Does this mean she will also be put under the spotlight? Whoever wants to be safe, please don't get close to me," he said. According to The Edge, the Transport Ministry awarded the RM178.29 million contract to Integriti Padu on May 26 following a request for proposals issued in June last year. The report said Integriti Padu's executive chairman is Datuk Nasir Baki, Azam's brother. The company is tasked with developing, operating and maintaining the RC/VEP system for six years. Azam's tenure as MACC chief commissioner was previously overshadowed by a controversy over his shareholdings after allegations surfaced in late 2021 that he held millions of shares and warrants in several public-listed companies while serving as a senior civil servant.In January 2022, Azam denied any wrongdoing, saying the shares had been purchased by his brother using his trading account. Earlier this year, Azam again came under scrutiny following Bloomberg reports alleging irregularities involving his shareholdings in a public-listed company while he was MACC chief. He has since filed a RM100 million defamation suit against Bloomberg over the reports."""

print("Loading model...")
model = load_model(MODEL_PATH, DEVICE)
tok = Tokenizer(max_length=128)

words = FULL_ARTICLE.split()
print(f"Full article: {len(words)} words\n")

def predict(text):
    cleaned = clean_text(text)
    tokens = tok.tokenize(cleaned)
    input_ids = tokens["input_ids"].to(DEVICE)
    attn_mask = tokens["attention_mask"].to(DEVICE)
    feats = get_features_tensor(cleaned).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids, attn_mask, feats)
        probs = F.softmax(logits, dim=1)[0]
    return float(probs[0]), feats

print(f"{'Word cutoff':>12} | {'fake_score':>10} | {'feature[1] value':>18}")
print("-" * 48)
for cutoff in [20, 40, 60, 80, 100, 150, 200, 250, 300, len(words)]:
    if cutoff > len(words):
        cutoff = len(words)
    truncated = " ".join(words[:cutoff])
    score, feats = predict(truncated)
    f1_val = feats.flatten()[1].item()
    print(f"{cutoff:>12} | {score:>10.4f} | {f1_val:>18.2f}")