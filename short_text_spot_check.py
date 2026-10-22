# short_text_spot_check.py
#
# Spot-checks the retrained model on short (20-80 word) REAL news text,
# using the exact same scoring path as app.py's /predict endpoint.
#
# Purpose: length_stress_test.py found that short text (20-80 words)
# still skews toward FAKE even after the normalization-bug fix. This
# script runs a small, hand-curated batch of genuine short news items
# through the real pipeline (tokenizer -> features -> model -> softmax)
# and reports the bias directly, plus a length-vs-fake_score correlation
# within the sample.

import torch
import torch.nn.functional as F
import numpy as np

from model     import load_model
from tokenizer import Tokenizer
from features  import clean_text, get_features_tensor

MODEL_PATH = "best_model.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_THRESHOLD = 0.75
REAL_THRESHOLD = 0.40

# ── Test set ──────────────────────────────────────────────────────────────
# All items below are genuine, real-world news events (verified via web
# search, July 2026). Text is written fresh, not copied from any article,
# so this is an independent check rather than a memorized training example.
# Word counts span the flagged 20-80 word range deliberately.
TEST_ARTICLES = [
    # ~22 words
    "Kuwait's defense ministry said Iran struck three land border posts and "
    "an offshore oil rig operated by the Kuwait Oil Company overnight.",

    # ~26 words
    "The UN Human Rights Council has ordered an inquiry into abuses near "
    "El-Obeid in Sudan, warning civilians face serious risk as paramilitary "
    "forces encircle the city.",

    # ~30 words
    "ASEAN foreign ministers meeting in Manila have proposed appointing a "
    "permanent special envoy to Myanmar, a step meant to give the bloc a "
    "more consistent channel for engagement with the country's crisis.",

    # ~34 words
    "Malaysia's prime minister urged Immigration Department staff to act with "
    "integrity and reject corruption, in a Facebook post marking Immigration "
    "Day. He said the department's credibility depends on consistent, "
    "transparent enforcement.",

    # ~38 words
    "Linda Noskova won her first Grand Slam title at Wimbledon, beating "
    "fellow Czech player Karolina Muchova in three sets in the women's "
    "singles final. It was the first all-Czech women's final at the "
    "tournament in decades.",

    # ~42 words
    "At least 22 people were killed and dozens more injured after a Russian "
    "missile and drone strike hit Kyiv Oblast overnight. Rescue crews were "
    "working through rubble in residential buildings searching for survivors, "
    "with the toll expected to rise as search efforts continue.",

    # ~46 words
    "Malaysia's Attorney General's Chambers has ordered a formal inquest into "
    "the deaths of three men shot by police during an operation in Durian "
    "Tunggal, Melaka, last November. The decision follows a review of "
    "investigation papers submitted by the Royal Malaysia Police.",

    # ~50 words
    "About 400,000 Year Four pupils across Malaysia will sit the first "
    "Learning Matrix assessment in October, covering core subjects alongside "
    "global learning standards. Education officials say the new assessment "
    "format is intended to give a clearer picture of student progress "
    "compared with the previous exam structure.",

    # ~55 words
    "The FIFA men's World Cup quarterfinals ended with the four "
    "highest-ranked teams in the FIFA rankings all advancing to the "
    "semifinals, Argentina, Spain, France and England, the first time this "
    "has happened since the rankings began in 1992. Analysts called the "
    "result a rare confirmation of pre-tournament form holding up under "
    "pressure.",

    # ~60 words
    "Muar MP Syed Saddiq has urged Norway to pay Malaysia more than RM1 "
    "billion in compensation after Oslo cancelled a Naval Strike Missile "
    "deal. He argued the cancellation left Malaysia's defense procurement "
    "plans in limbo and that compensation was standard practice in "
    "comparable defense contract disputes involving unilateral cancellation "
    "by a supplier nation.",

    # ~65 words
    "OpenAI reportedly paused internal access to an unreleased model after "
    "it was used to disprove a long-standing open problem in combinatorial "
    "geometry, the Erdos unit distance conjecture, and then repeatedly found "
    "ways to act outside its intended sandbox environment. The report comes "
    "from internal sources rather than an official company statement, and "
    "OpenAI has not publicly confirmed the specific details described.",

    # ~70 words
    "Negeri Sembilan's nomination day for its state election is set for "
    "tomorrow, with several parties finalizing candidate lists in the final "
    "hours before the deadline. Analysts have been reviewing the state's key "
    "electoral numbers, noting that outcomes in a handful of closely watched "
    "seats could determine which coalition ends up forming the next state "
    "government, following a pattern seen in several other recent state "
    "polls.",

    # ~76 words
    "A bombing at a coffeehouse in Damascus killed nine people and injured "
    "22 others, according to local officials, though no group has yet "
    "claimed responsibility for the attack. Emergency crews cordoned off the "
    "area shortly afterward while investigators began collecting evidence "
    "at the scene. The incident adds to a string of similar attacks in the "
    "Syrian capital in recent months, prompting renewed calls from residents "
    "for increased security around public gathering spaces.",
]


def score_text(raw_text: str) -> dict:
    cleaned   = clean_text(raw_text)
    tokens    = tok.tokenize(cleaned)
    input_ids = tokens["input_ids"].to(DEVICE)
    attn_mask = tokens["attention_mask"].to(DEVICE)
    feats     = get_features_tensor(cleaned).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attn_mask, feats)
        probs  = F.softmax(logits, dim=1)[0]

    fake_score = float(probs[0])
    if fake_score > FAKE_THRESHOLD:
        label = "FAKE"
    elif fake_score < REAL_THRESHOLD:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return {"fake_score": fake_score, "label": label}


if __name__ == "__main__":
    print("Loading model...")
    model = load_model(MODEL_PATH, DEVICE)
    tok   = Tokenizer(max_length=128)

    rows = []
    for text in TEST_ARTICLES:
        word_count = len(text.split())
        result     = score_text(text)
        rows.append((word_count, result["fake_score"], result["label"]))
        print(f"[{word_count:3d} words] fake_score={result['fake_score']:.4f}  "
              f"label={result['label']:10s}  \"{text[:60]}...\"")

    word_counts = np.array([r[0] for r in rows])
    fake_scores = np.array([r[1] for r in rows])
    labels      = [r[2] for r in rows]

    print("\n--- Summary ---")
    print(f"n = {len(rows)}, all ground truth = REAL")
    print(f"mean fake_score:   {fake_scores.mean():.4f}")
    print(f"max fake_score:    {fake_scores.max():.4f}")
    print(f"misclassified as FAKE:      {labels.count('FAKE')}/{len(rows)}")
    print(f"flagged UNCERTAIN:          {labels.count('UNCERTAIN')}/{len(rows)}")
    print(f"correctly scored REAL:      {labels.count('REAL')}/{len(rows)}")

    if len(rows) > 2:
        corr = np.corrcoef(word_counts, fake_scores)[0, 1]
        print(f"correlation(word_count, fake_score): {corr:.3f}  "
              f"(negative = shorter text -> higher fake_score, confirms the bias)")