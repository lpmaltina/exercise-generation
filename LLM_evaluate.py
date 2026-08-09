import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils import call_LLM, create_client

load_dotenv()

MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def load_evaluation_prompt() -> str:
    prompt_path = Path("prompts/evaluation_template.txt")
    return prompt_path.read_text(encoding="utf-8")


def parse_exercises(content: str) -> list[dict]:
    exercises = []
    parts = re.split(r"(?=Topic:)", content.strip())

    for part in parts:
        if not part.strip():
            continue

        topic_match = re.search(r"Topic:\s*(.+?)(?:\n)", part)
        text_match = re.search(
            r"Text:\s*(.+?)(?:\n\s*\n(?=Questions:))", part, re.DOTALL
        )
        questions_match = re.search(r"Questions:\s*(.+)", part, re.DOTALL)

        if topic_match and text_match and questions_match:
            exercises.append(
                {
                    "topic": topic_match.group(1).strip(),
                    "text": text_match.group(1).strip(),
                    "questions": questions_match.group(1).strip(),
                }
            )

    return exercises


def evaluate_exercises_with_llm(exercises: list[dict], client) -> list[str]:
    evaluation_template = load_evaluation_prompt()
    results = []

    for i, exercise in enumerate(exercises, start=1):
        print(f"[{i}/{len(exercises)}] Evaluating (topic: {exercise['topic']})...")

        exercise_str = f"Topic: {exercise['topic']}\n\nText: {exercise['text']}\n\nQuestions: {exercise['questions']}"
        response_text = call_LLM(
            client,
            MODEL_NAME,
            user_prompt=exercise_str,
            system_prompt=evaluation_template,
            extra_body={"thinking": {"type": "disabled"}},
        )
        results.append(response_text)
        print(f"  ok ({len(response_text)} chars)")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("output", type=Path, help="Output file")
    args = parser.parse_args()

    client = create_client()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    exercises = parse_exercises(content)

    if not exercises:
        raise ValueError("No exercises found in input file.")

    print(f"Loaded {len(exercises)} exercise(s) from {args.input}")
    results = evaluate_exercises_with_llm(exercises, client)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n\n\n".join(results), encoding="utf-8")
    print(f"Done. Wrote {len(results)} evaluation(s) to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
