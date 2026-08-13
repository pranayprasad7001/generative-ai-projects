import time
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Test both localhost and 127.0.0.1
for base_url in ["http://localhost:4000", "http://127.0.0.1:4000"]:
    print(f"\n--- Testing Completion with base_url: {base_url} ---")
    payload = {
        "model": "gpt-oss-120b-groq",
        "messages": [{"role": "user", "content": "say hi"}],
        "temperature": 0.2
    }
    headers = {
        "Authorization": "Bearer -----",
        "Content-Type": "application/json"
    }

    for i in range(2):
        start_time = time.time()
        try:
            r = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers)
            elapsed = time.time() - start_time
            print(f"Call {i+1} Status Code: {r.status_code} in {elapsed:.4f} seconds")
            if r.status_code == 200:
                resp = r.json()
                print("Response:", resp["choices"][0]["message"]["content"])
        except Exception as e:
            print("Error:", e)
