import json
import os

import config
import evaluation_experiment
from CEFR_level import CEFRLevelParser

if __name__ == "__main__":
    with open(evaluation_experiment.ROLE_PROMPT_PATH, encoding="utf-8") as f:
        role = f.read()

    with open(evaluation_experiment.EVALUATION_PROMPT_PATH, encoding="utf-8") as f:
        evaluation_template = f.read()

    nlp = evaluation_experiment.create_nlp()
    CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)
    variants = ("bad_example", "medium_example", "excellent_example")
    words = config.words

    for variant in variants:
        print(variant)
        generation_path = os.path.join(
            "results", "generation", f"reading_comprehension_{variant}.txt"
        )
        evaluation_path = os.path.join(
            "results", "evaluation", f"reading_comprehension_{variant}.json"
        )
        summary_path = os.path.join(
            "results",
            "summary",
            f"reading_comprehension_{variant}.json",
        )
        evaluation_results = evaluation_experiment.evaluate(
            nlp,
            CEFR_parser,
            words,
            role,
            evaluation_template,
            generation_path,
            evaluation_path,
            summary_path,
        )
        print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

    print()
    CEFR_parser.quit()
