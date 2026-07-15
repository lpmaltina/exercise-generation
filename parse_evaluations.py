import argparse
import json
import os
import re
import sys

TEXT_CRITERIA = ("Matching the Topic", "Logic & Commonsense", "Vocabulary & Grammar")
INDIVIDUAL_QUESTION_CRITERIA = (
    "Logic & Commonsense",
    "Vocabulary & Grammar",
    "Text-Based Answerability",
    "Reading Dependency",
    "Answer Unambiguity",
    "Distractor Plausibility",
    "Using Paraphrases",
    "Explanation Quality",
)
OVERALL_QUESTION_CRITERIA = ("Text Coverage",)


def standarize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def parse_criterion(text: str, criterion: str) -> tuple[str, float]:
    """Parse a single criterion from evaluation text."""
    reasoning = ""
    score = 0

    match = re.search(rf"{criterion}\.\s+(.+?)\s+<?(\d)>?/5>?", text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        score = int(match.group(2))

    standarized_score = standarize(score, min_value=1, max_value=5)
    return reasoning, standarized_score


def parse_raw_evaluation(raw_text: str) -> dict | None:
    """Parse raw LLM evaluation output into structured format."""
    raw_text = raw_text.replace("*", "")
    parts = [s.strip() for s in raw_text.split("---") if s.strip()]

    if not parts:
        return None

    text_part = parts[0]
    individual_question_parts = parts[1:-1] if len(parts) > 2 else []
    overall_question_part = parts[-1] if len(parts) > 1 else ""

    evaluation_results = {"Text": {}, "Questions": {"Overall": {}}}

    for criterion in TEXT_CRITERIA:
        reasoning, score = parse_criterion(text_part, criterion)
        evaluation_results["Text"][criterion] = {
            "Reasoning": reasoning,
            "Score": round(score, 4),
        }

    for i, question_part in enumerate(individual_question_parts, start=1):
        evaluation_results["Questions"][f"Q{i}"] = {}
        for criterion in INDIVIDUAL_QUESTION_CRITERIA:
            reasoning, score = parse_criterion(question_part, criterion)
            evaluation_results["Questions"][f"Q{i}"][criterion] = {
                "Reasoning": reasoning,
                "Score": round(score, 4),
            }

    if individual_question_parts:
        for criterion in INDIVIDUAL_QUESTION_CRITERIA:
            avg_score = 0
            for i in range(1, len(individual_question_parts) + 1):
                avg_score += evaluation_results["Questions"][f"Q{i}"][criterion][
                    "Score"
                ]
            avg_score /= len(individual_question_parts)
            evaluation_results["Questions"]["Overall"][f"{criterion} (Average)"] = (
                round(avg_score, 4)
            )

    if overall_question_part:
        for criterion in OVERALL_QUESTION_CRITERIA:
            reasoning, score = parse_criterion(overall_question_part, criterion)
            evaluation_results["Questions"]["Overall"][criterion] = {
                "Reasoning": reasoning,
                "Score": round(score, 4),
            }

    return evaluation_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input file (e.g., raw_evaluation_baseline.txt)")
    parser.add_argument(
        "output", help="Output file (e.g., structured_evaluation_baseline.txt)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    exercises = re.split(r"\n\n\n\n", content)

    all_results = []
    for i, exercise in enumerate(exercises, start=1):
        exercise = exercise.strip()
        if not exercise:
            continue

        parsed = parse_raw_evaluation(exercise)
        if parsed:
            parsed["Exercise"] = i
            all_results.append(parsed)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"exercises": all_results}, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_results)} exercises -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
