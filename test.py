from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files(repo_id="MadDane/verilens-ai", repo_type="space")
print("feature_stats_malaysian.npz" in files)
print(sorted(files))