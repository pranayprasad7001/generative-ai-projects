import requests
import json

base_url = "http://127.0.0.1:4000"
payload = {
    "model": "gpt-oss-120b-groq",
    "messages": [{"role": "user", "content": "say hi"}],
    "temperature": 0.2
}
headers = {
    "Authorization": "Bearer ------------",
    "Content-Type": "application/json"
}

try:
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers)
    print("Status:", r.status_code)
    print("Full Response JSON:")
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)
