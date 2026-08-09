import sys
from pathlib import Path

from CEFR_level import CEFRLevelParser
from LLM_evaluate import evaluate_exercises_with_llm, parse_exercises
from parse_evaluations import (
    calculate_all_exercises_average,
    create_nlp,
    create_structured_evaluations,
    save_structured_evaluations,
)
from utils import create_client


def create_example_configs():
    return [
        {
            "config_id": 0,
            "word_count": 100,
            "CEFR_level": "A2",
            "wordlist": [
                "best",
                "better",
                "fun",
                "have",
                "home",
                "maths",
                "need",
                "start",
                "talk",
                "worst",
            ],
            "n_questions": 3,
            "n_options": 4,
        },
        {
            "config_id": 1,
            "word_count": 150,
            "CEFR_level": "B1",
            "wordlist": [
                "brave",
                "concentrate",
                "countryside",
                "decade",
                "fairly",
                "involved",
                "mainly",
                "studio",
                "surface",
                "tyre",
            ],
            "n_questions": 4,
            "n_options": 4,
        },
        {
            "config_id": 2,
            "word_count": 200,
            "CEFR_level": "B2",
            "wordlist": [
                "absorb",
                "barely",
                "bold",
                "challenging",
                "inch",
                "maintain",
                "nearby",
                "sense",
                "slope",
                "visible",
            ],
            "n_questions": 5,
            "n_options": 4,
        },
    ]


def main():
    output_path = Path(
        "results/structured_evaluations/structured_evaluations_few-shot_examples.json"
    )
    example_files = (
        "prompts/example_A2.txt",
        "prompts/example_B1.txt",
        "prompts/example_B2.txt",
    )

    for filepath in example_files:
        if not Path(filepath).is_file():
            raise FileNotFoundError(f"File not found: {filepath}")

    nlp = create_nlp()
    CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)
    configs = create_example_configs()

    parsed_exercises = []
    for filepath in example_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parsed_part = parse_exercises(content)
        parsed_exercises.extend(parsed_part)

    exercise_texts = [
        f"Topic: {ex['topic']}\n\nText: {ex['text']}\n\nQuestions: {ex['questions']}"
        for ex in parsed_exercises
    ]

    client = create_client()
    raw_evaluations = evaluate_exercises_with_llm(parsed_exercises, client)
    structured_evaluations = create_structured_evaluations(
        exercise_texts, raw_evaluations, configs, nlp, CEFR_parser
    )
    all_exercises_averages = calculate_all_exercises_average(structured_evaluations)
    save_structured_evaluations(
        structured_evaluations, all_exercises_averages, output_path
    )


if __name__ == "__main__":
    sys.exit(main())
