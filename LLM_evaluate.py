import argparse
from pathlib import Path

from dotenv import load_dotenv

from utils import SEP, call_LLM, create_client, parse_exercise, stringify_exercise

load_dotenv()

MODEL_NAME = "deepseek-v4-flash"


def load_evaluation_prompt() -> str:
    prompt_path = Path("prompts/evaluation_template.txt")
    return prompt_path.read_text(encoding="utf-8")


def evaluate_exercises_with_llm(exercises: list[dict], client) -> list[str]:
    evaluation_template = load_evaluation_prompt()
    results = []

    for i, exercise in enumerate(exercises, start=1):
        print(f"[{i}/{len(exercises)}] Evaluating (topic: {exercise['topic']})...")
        exercise_str = stringify_exercise(exercise)
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

    client = create_client(
        api_key_name="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"
    )

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read().split(SEP)

    exercises = []
    for exercise in content:
        exercise = parse_exercise(exercise)
        if exercise:
            exercises.append(exercise)

    if not exercises:
        raise ValueError("No exercises found.")

    print(f"Loaded {len(exercises)} exercise(s) from {args.input}")
    results = evaluate_exercises_with_llm(exercises, client)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(SEP.join(results), encoding="utf-8")
    print(f"Done. Wrote {len(results)} evaluation(s) to {args.output}")


if __name__ == "__main__":
    main()
