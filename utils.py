import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {API_KEY}",
}


def generate(model, prompt, role, result_path, **kwargs):
    if role:
        messages = [{"role": "system", "content": role}]
    else:
        messages = []
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        **kwargs,
    }

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    response = requests.post(URL, json=payload, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"API returned status {response.status_code}: {response.text}")

    try:
        response_json = response.json()
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, indent=4)

    except requests.exceptions.JSONDecodeError as e:
        raise Exception(
            f"Failed to parse JSON. Raw response: {response.text}. Error: {e}"
        )
