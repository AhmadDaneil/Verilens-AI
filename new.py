from huggingface_hub import HfApi

api = HfApi()
REPO = "MadDane/verilens-ai"

# The actual bug fix — corrected feature definitions + dynamic stats loading
api.upload_file(
    path_or_fileobj="verilens-ai/features.py",
    path_in_repo="features.py",
    repo_id=REPO,
    repo_type="space",
)

# REQUIRED alongside the fixed features.py above — the Space currently
# has no copy of this at all, and the fixed features.py raises a
# RuntimeError at inference time without it
api.upload_file(
    path_or_fileobj="feature_stats_malaysian.npz",
    path_in_repo="feature_stats_malaysian.npz",
    repo_id=REPO,
    repo_type="space",
)

# This session's retrained weights (cleaned data, 1045 real / 1880 fake,
# validated: boilerplate 0.0pp, length ~1.35pp, translated-gap 6.92pp,
# 9/9 manual spot-check)
api.upload_file(
    path_or_fileobj="best_model.pt",
    path_in_repo="best_model.pt",
    repo_id=REPO,
    repo_type="space",
)