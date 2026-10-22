# test_api.py
# Run this BEFORE connecting Flutter to verify the API works end-to-end.
# Usage: python test_api.py

import requests
import json

BASE_URL = "http://localhost:5000"

TEST_CASES = [
    {
        "label": "Obvious scam",
        "text": "CONGRATULATIONS! You have won RM10,000! Click here to claim your prize NOW. Send your IC number and bank account to collect.",
    },
    {
        "label": "Legitimate message",
        "text": "Hi, just a reminder that your dental appointment is tomorrow at 2pm. Please call us if you need to reschedule.",
    },
    {
        "label": "Phishing attempt",
        "text": "Your Maybank account has been suspended. Verify your account immediately at http://maybank-secure-login.xyz or lose access permanently.",
    },
    {
        "label": "Normal conversation",
        "text": "Can we meet for lunch tomorrow? I was thinking of trying that new restaurant near the office.",
    },
]

print("=" * 60)
print("🧪 ScamShield API Test")
print("=" * 60)

# Health check
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"\n✅ Health check: {r.json()}")
except Exception as e:
    print(f"\n❌ API not reachable: {e}")
    print("   Make sure the Flask server is running: python app.py")
    exit(1)

# Prediction tests
print()
for tc in TEST_CASES:
    try:
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"text": tc["text"]},
            timeout=30,
        )
        data = r.json()
        if r.status_code == 200:
            label      = data["label"]
            confidence = data["confidence"]
            elapsed    = data["elapsed_ms"]
            fake_s     = data["fake_score"]
            real_s     = data["real_score"]
            icon       = "🚨" if data["is_fake"] else "✅"
            print(f"{icon} [{tc['label']}]")
            print(f"   Result     : {label} (confidence: {confidence:.1%})")
            print(f"   Fake/Real  : {fake_s:.4f} / {real_s:.4f}")
            print(f"   Server time: {elapsed}ms")
        else:
            print(f"❌ [{tc['label']}] Error: {data.get('error')}")
    except Exception as e:
        print(f"❌ [{tc['label']}] Exception: {e}")
    print()