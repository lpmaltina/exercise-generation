import json
import re

READING_COMPREHENSION_BAD_EXAMPLE_RESULT_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_bad.json"
)
READING_COMPREHENSION_MEDIUM_EXAMPLE_RESULT_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_medium.json"
)
READING_COMPREHENSION_EXCELLENT_EXAMPLE_RESULT_PATH = "prompts/reading_comprehension/reading_comprehension_generation_example_excellent.json"

result_paths = (
    READING_COMPREHENSION_BAD_EXAMPLE_RESULT_PATH,
    READING_COMPREHENSION_MEDIUM_EXAMPLE_RESULT_PATH,
    READING_COMPREHENSION_EXCELLENT_EXAMPLE_RESULT_PATH,
)

question_criteria = (
    "Clarity & Grammar",
    r"Text[\u2011-]Based Answerability",
    "Reading Dependency",
    "Answer Unambiguity",
    "Distractor Plausibility",
)
overall_criteria = ("Text Coverage", "Exercise Usability")


def parse_criteria(text, criteria):
    evaluation_result = {}
    for criterion in criteria:
        match = re.search(
            rf"\**{criterion}.\**\s+(.+?\.)\s+\**(\d)/5\**", text, re.DOTALL
        )
        if match:
            reasoning = match.group(1).strip()
            score = int(match.group(2))
        else:
            reasoning = ""
            score = 0
        evaluation_result[criterion] = {"reasoning": reasoning, "score": score}
    return evaluation_result


for result_path in result_paths:
    print(result_path.upper())
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    content = data["choices"][0]["message"]["content"].strip()
    content = content.split("---")
    n_questions = len(content) - 1
    evaluation_results = []

    for i in range(n_questions):
        question = content[i].strip()
        evaluation_result = parse_criteria(question, question_criteria)
        evaluation_results.append(evaluation_result)

    evaluation_result = parse_criteria(content[-1], overall_criteria)
    evaluation_results.append(evaluation_result)

    evaluation_results = json.dumps(evaluation_results, indent=4, ensure_ascii=False)
    print(evaluation_results)
