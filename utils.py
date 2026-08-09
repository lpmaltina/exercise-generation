import os
import time

from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

LLM_TEXT_CRITERIA = (
    "Matching the Topic",
    "Logic & Commonsense",
    "Vocabulary & Grammar",
)
ALL_TEXT_CRITERIA = (
    "Word Count",
    "Target Word Usage",
    "Target Word Distribution",
    "Matching the CEFR Level",
) + LLM_TEXT_CRITERIA

LLM_INDIVIDUAL_QUESTION_CRITERIA = (
    "Logic & Commonsense",
    "Vocabulary & Grammar",
    "Text-Based Answerability",
    "Reading Dependency",
    "Answer Unambiguity",
    "Distractor Plausibility",
    "Using Paraphrases",
    "Explanation Quality",
)
ALL_INDIVIDUAL_QUESTION_CRITERIA = LLM_INDIVIDUAL_QUESTION_CRITERIA

LLM_OVERALL_QUESTION_CRITERIA = ("Text Coverage",)
ALL_OVERALL_QUESTION_CRITERIA = (
    "Matching the CEFR Level",
) + LLM_OVERALL_QUESTION_CRITERIA

ALL_CRITERIA = [
    ("text", ALL_TEXT_CRITERIA),
    ("individual question", ALL_INDIVIDUAL_QUESTION_CRITERIA),
    ("overall question", ALL_OVERALL_QUESTION_CRITERIA),
]


def create_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file.")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return client


def call_LLM(
    client: OpenAI, model_name: str, user_prompt: str, system_prompt: str = "", **kwargs
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            print(f"  ! API error on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Giving up after {MAX_RETRIES} attempts: {last_error}")
