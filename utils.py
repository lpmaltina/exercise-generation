import json
import os
import re

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
        "max_tokens": 3072,
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
        return response_json["choices"][0]["message"]["content"]

    except requests.exceptions.JSONDecodeError as e:
        raise Exception(
            f"Failed to parse JSON. Raw response: {response.text}. Error: {e}"
        )


def parse_exercise(exercise: str) -> dict[str, str | None]:
    topic = re.search(r"Topic:\s*(.+?)(?:\n)", exercise)
    if topic:
        topic = topic.group(1).strip()

    text = re.search(r"Text:\s*(.+?)(?:\n\s*\n(?=Questions:))", exercise, re.DOTALL)
    if text:
        text = text.group(1).strip()

    questions = re.search(r"Questions:\s*(.+)", exercise, re.DOTALL)
    if questions:
        questions = questions.group(1).strip()

    return {"topic": topic, "text": text, "questions": questions}
