import json
import os
import time

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

READING_COMPREHENSION_ROLE = (
    "prompts/reading_comprehension/reading_comprehension_role.txt"
)
READING_COMPREHENSION_TEXT = (
    "prompts/reading_comprehension/reading_comprehension_text.txt"
)
READING_COMPREHENSION_EVALUATION_PROMPT_PATH = (
    "prompts/reading_comprehension/reading_comprehension_evaluation.txt"
)
READING_COMPREHENSION_BAD_EXAMPLE_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_bad.txt"
)
READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_medium.txt"
)
READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH = "prompts/reading_comprehension/reading_comprehension_generation_example_excellent.txt"


def generate(prompt, result_path):
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": "low",
        "stream": False,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 2048,
    }
    time.sleep(10)

    response = requests.post(URL, json=payload, headers=HEADERS)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(response.json(), f, indent=4)


with open(READING_COMPREHENSION_ROLE, encoding="utf-8") as f:
    role = f.read()

with open(READING_COMPREHENSION_TEXT, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_EVALUATION_PROMPT_PATH, encoding="utf-8") as f:
    template = f.read()

exercise_paths = (
    READING_COMPREHENSION_BAD_EXAMPLE_PATH,
    READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH,
    READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH,
)

for exercise_path in exercise_paths:
    with open(exercise_path, encoding="utf-8") as f:
        exercise = f.read()

    prompt = template.format(text=text, exercise=exercise)
    result_path = exercise_path.split(".")[0] + ".json"
    generate(prompt, result_path)
