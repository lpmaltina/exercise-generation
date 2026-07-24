import json
from pathlib import Path

import numpy as np
from scipy import stats

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


def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        all_results = json.load(f)["Exercises"]

    text_scores = {criterion: [] for criterion in ALL_TEXT_CRITERIA}
    question_scores = {criterion: [] for criterion in ALL_INDIVIDUAL_QUESTION_CRITERIA}
    overall_question_scores = {
        criterion: [] for criterion in ALL_OVERALL_QUESTION_CRITERIA
    }

    for result in all_results:
        for criterion in ALL_TEXT_CRITERIA:
            text_scores[criterion].append(result["Text"][criterion]["Score"])

        for criterion in ALL_INDIVIDUAL_QUESTION_CRITERIA:
            key = f"{criterion} (Exercise Mean Score)"
            question_scores[criterion].append(result["Questions"]["Overall"][key])

        for criterion in ALL_OVERALL_QUESTION_CRITERIA:
            overall_question_scores[criterion].append(
                result["Questions"]["Overall"][criterion]["Score"]
            )

    scores = {
        "Text": text_scores,
        "Questions": {
            "Individual": question_scores,
            "Overall": overall_question_scores,
        },
    }

    return scores


def compare_groups(baseline_scores, other_scores, criteria, group_type):
    for criterion in criteria:
        if group_type == "text":
            baseline = baseline_scores["Text"][criterion]
            other = other_scores["Text"][criterion]
        elif group_type == "individual question":
            baseline = baseline_scores["Questions"]["Individual"][criterion]
            other = other_scores["Questions"]["Individual"][criterion]
        elif group_type == "overall question":
            baseline = baseline_scores["Questions"]["Overall"][criterion]
            other = other_scores["Questions"]["Overall"][criterion]

        diff = np.array(other) - np.array(baseline)
        if np.all(diff == 0):
            continue

        stat, p_value = stats.wilcoxon(other, baseline, alternative="greater")
        if p_value < P_VALUE_THRESHOLD:
            print(f"{(group_type + ' criteria').upper()} {criterion}:")
            print(f"  Statistic: {stat}")
            print(f"  P-value: {p_value}")
            print()


def run_comparison(name, baseline_scores, other_scores):
    print(f"H0: {name} is not better than baseline\n")
    for group_type, criteria in ALL_CRITERIA:
        compare_groups(baseline_scores, other_scores, criteria, group_type)
    print()


P_VALUE_THRESHOLD = 0.05

structured_evaluation_path = Path("results/structured_evaluations")
baseline_path = Path(structured_evaluation_path, "structured_evaluations_baseline.json")
role_path = Path(
    structured_evaluation_path, "structured_evaluations_baseline_with_role.json"
)
few_shot_path = Path(
    structured_evaluation_path, "structured_evaluations_baseline_few-shot.json"
)
cot_path = Path(structured_evaluation_path, "structured_evaluations_CoT.json")

baseline_scores = load_scores(baseline_path)
role_scores = load_scores(role_path)
few_shot_scores = load_scores(few_shot_path)
cot_scores = load_scores(cot_path)

run_comparison("baseline + role", baseline_scores, role_scores)
run_comparison("baseline + few-shot", baseline_scores, few_shot_scores)
run_comparison("CoT", baseline_scores, cot_scores)
