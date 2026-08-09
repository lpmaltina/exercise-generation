import json
from pathlib import Path

from CEFR_level import CEFRLevelParser
from LLM_evaluate import evaluate_exercises_with_llm
from parse_evaluations import (
    calculate_all_exercises_average,
    create_nlp,
    create_structured_evaluations,
    save_structured_evaluations,
)
from utils import SEP, create_client, parse_exercise


def main():
    output_path = Path(
        "results/structured_evaluations/structured_evaluations_few-shot_examples.json"
    )
    example_files = (
        Path("prompts/example_A2.txt"),
        Path("prompts/example_B1.txt"),
        Path("prompts/example_B2.txt"),
    )

    for filepath in example_files:
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

    nlp = create_nlp()
    CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)
    with open("config_few-shot.json", "r", encoding="utf-8") as f:
        configs = json.load(f)["exercise_configs"]

    parsed_exercises = []
    for filepath in example_files:
        with open(filepath, "r", encoding="utf-8") as f:
            exercises = f.read().split(SEP)
        for exercise in exercises:
            parsed_exercise = parse_exercise(exercise)
            if parsed_exercise:
                parsed_exercises.append(parsed_exercise)

    client = create_client(
        api_key_name="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"
    )
    raw_evaluations = evaluate_exercises_with_llm(parsed_exercises, client)
    structured_evaluations = create_structured_evaluations(
        parsed_exercises, raw_evaluations, configs, nlp, CEFR_parser
    )
    all_exercises_averages = calculate_all_exercises_average(structured_evaluations)
    save_structured_evaluations(
        structured_evaluations, all_exercises_averages, output_path
    )


if __name__ == "__main__":
    main()
