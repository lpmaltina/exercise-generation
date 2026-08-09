import argparse
import json
import re
from pathlib import Path

import spacy
from tqdm import tqdm

from CEFR_level import CEFRLevelParser
from utils import (
    ALL_INDIVIDUAL_QUESTION_CRITERIA,
    ALL_OVERALL_QUESTION_CRITERIA,
    ALL_TEXT_CRITERIA,
    LLM_INDIVIDUAL_QUESTION_CRITERIA,
    LLM_OVERALL_QUESTION_CRITERIA,
    LLM_TEXT_CRITERIA,
    SEP,
    parse_exercise,
)


def normalize(
    value: int | float, min_value: int | float, max_value: int | float
) -> float:
    return (value - min_value) / (max_value - min_value)


def create_nlp():
    nlp = spacy.load("en_core_web_sm")
    infixes = list(nlp.Defaults.infixes)

    # Делаем так, чтобы токены не разбивались по дефисам и апострофам
    patterns_to_remove = {
        r"(?<=[{a}])-(?=[{a}])".format(a=spacy.lang.char_classes.ALPHA),
        r"(?<=[{a}])'(?=[{a}])".format(a=spacy.lang.char_classes.ALPHA),
    }

    infixes = [p for p in infixes if p not in patterns_to_remove]
    infix_re = re.compile("|".join(infixes)) if infixes else None

    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer if infix_re else None,
        token_match=nlp.tokenizer.token_match,
        url_match=nlp.tokenizer.url_match,
    )
    return nlp


def tokenize(nlp, text: str) -> list[spacy.tokens.token.Token]:
    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not token.is_punct and not token.is_space and not token.is_digit
    ]
    return tokens


def evaluate_word_count(
    tokens: list[spacy.tokens.token.Token], target_word_count: int
) -> float:
    real_word_count = len(tokens)
    return min(real_word_count, target_word_count) / max(
        real_word_count, target_word_count
    )


def check_words_from_wordlist(
    tokens: list[spacy.tokens.token.Token], words: set[str]
) -> tuple[int, int, set[str]]:
    used = set()
    unused = words.copy()
    for token in tokens:
        lemma = token.lemma_.lower()
        word_form = token.text.lower()
        if lemma in words:
            used.add(lemma)
            unused.discard(lemma)
        elif word_form in words:
            used.add(word_form)
            unused.discard(word_form)
    return len(used), len(words), unused


def evaluate_evenness(
    tokens: spacy.tokens.doc.Doc, words: set[str]
) -> tuple[float, float, float]:
    words_copy = words.copy()
    len_text = len(tokens)
    positions = []

    for pos, token in enumerate(tokens):
        lemma = token.lemma_.lower()
        word_form = token.text.lower()
        if lemma in words_copy:
            positions.append(pos)
            words_copy.discard(lemma)
        elif word_form in words_copy:
            positions.append(pos)
            words_copy.discard(word_form)

    gaps = [positions[0]]

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i] - 1
        gaps.append(gap)

    gap = len_text - positions[-1] - 1
    gaps.append(gap)

    n = len(gaps)
    sum_gaps = sum(gaps)
    max_gap = max(gaps)
    max_gap_share = max_gap / sum_gaps
    ideal_share = 1.0 / n
    score = 1.0 - (max_gap_share - ideal_share) / (1.0 - ideal_share)
    return score, max_gap_share, ideal_share


def calculate_CEFR_match(real_CEFR_level: str, target_CEFR_level: str) -> float:
    levels = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    CEFR_diff = abs(levels[real_CEFR_level] - levels[target_CEFR_level])
    return 1 - normalize(CEFR_diff, min_value=0, max_value=5)


def parse_criterion(text: str, criterion: str) -> tuple[str, float]:
    reasoning = ""
    score = 0

    match = re.search(
        rf"{criterion}\.\s+(.+?)\s*<?(1|2|3|4|5)>?/5>?", text.strip(), re.DOTALL
    )

    if match:
        reasoning = match.group(1).strip()
        score = int(match.group(2))

    normalized_score = normalize(score, min_value=1, max_value=5)
    return reasoning, normalized_score


def validate_and_print(text_id, eval_type, criterion, reasoning, score):
    if not 0 <= score <= 1:
        print(f"Text ID: {text_id}")
        print(f"Evaluation type: {eval_type}")
        print(f"Name of criteria: {criterion}")
        print("Issue: wrong score")
    if not reasoning:
        print(f"Text ID: {text_id}")
        print(f"Evaluation type: {eval_type}")
        print(f"Name of criteria: {criterion}")
        print("Issue: no reasoning")


def evaluate_non_LLM_metrics(generated_exercise, config, nlp, CEFR_parser):
    target_text_size = config["word_count"]
    words = set(config["wordlist"])
    target_CEFR_level = config["CEFR_level"]

    tokens = tokenize(nlp, generated_exercise["text"])

    word_count_score = evaluate_word_count(tokens, target_text_size)
    word_count_reasoning = (
        f"Words: {len(tokens)}. Target word count: {target_text_size}."
    )

    used_count, target_word_count, unused = check_words_from_wordlist(tokens, words)
    target_word_usage_score = used_count / target_word_count
    target_word_usage_reasoning = [
        f"Target word usage: {used_count}/{target_word_count}."
    ]
    if unused:
        target_word_usage_reasoning.append(f"Unused words: {', '.join(unused)}.")
    target_word_usage_reasoning = " ".join(target_word_usage_reasoning)

    target_word_distribution_score, max_gap_share, ideal_share = evaluate_evenness(
        tokens, words
    )
    target_word_distribution_reasoning = (
        f"Max gap share: {max_gap_share:.4f}. Ideal share: {ideal_share:.4f}."
    )

    CEFR_level_text = CEFR_parser.get_CEFR_level(generated_exercise["text"])
    CEFR_level_text_score = calculate_CEFR_match(CEFR_level_text, target_CEFR_level)
    CEFR_level_text_reasoning = f"Target CEFR level: {target_CEFR_level}. Determined CEFR level (text): {CEFR_level_text}"

    questions_replaced = (
        generated_exercise["questions"]
        .replace("Distractors:", "Wrong:")
        .replace("Explanation:", "Why:")
    )
    CEFR_level_questions = CEFR_parser.get_CEFR_level(questions_replaced)
    CEFR_level_questions_score = calculate_CEFR_match(
        CEFR_level_questions, target_CEFR_level
    )
    CEFR_level_questions_reasoning = f"Target CEFR level: {target_CEFR_level}. Determined CEFR level (questions): {CEFR_level_questions}"

    structured_evaluation = {
        "Text": {
            "Word Count": {
                "Reasoning": word_count_reasoning,
                "Score": word_count_score,
            },
            "Target Word Usage": {
                "Reasoning": target_word_usage_reasoning,
                "Score": target_word_usage_score,
            },
            "Target Word Distribution": {
                "Reasoning": target_word_distribution_reasoning,
                "Score": target_word_distribution_score,
            },
            "Matching the CEFR Level": {
                "Reasoning": CEFR_level_text_reasoning,
                "Score": CEFR_level_text_score,
            },
        },
        "Questions": {
            "Individual": {},
            "Overall": {
                "Matching the CEFR Level": {
                    "Reasoning": CEFR_level_questions_reasoning,
                    "Score": CEFR_level_questions_score,
                }
            },
        },
    }

    return structured_evaluation


def add_LLM_evaluation(
    raw_evaluation: str, structured_evaluation: dict, ex_id: int
) -> dict:
    parts = [s.strip() for s in raw_evaluation.split("---") if s.strip()]
    text_part, *individual_question_parts, overall_question_part = parts

    for criterion in LLM_TEXT_CRITERIA:
        reasoning, score = parse_criterion(text_part, criterion)
        structured_evaluation["Text"][criterion] = {
            "Reasoning": reasoning,
            "Score": score,
        }
        validate_and_print(ex_id, "text evaluation", criterion, reasoning, score)

    for i, question_part in enumerate(individual_question_parts, start=1):
        structured_evaluation["Questions"]["Individual"][f"Q{i}"] = {}
        for criterion in LLM_INDIVIDUAL_QUESTION_CRITERIA:
            reasoning, score = parse_criterion(question_part, criterion)
            structured_evaluation["Questions"]["Individual"][f"Q{i}"][criterion] = {
                "Reasoning": reasoning,
                "Score": score,
            }
            validate_and_print(
                ex_id, "question evaluation", criterion, reasoning, score
            )

    for criterion in LLM_INDIVIDUAL_QUESTION_CRITERIA:
        avg_score = 0
        for i in range(1, len(individual_question_parts) + 1):
            avg_score += structured_evaluation["Questions"]["Individual"][f"Q{i}"][
                criterion
            ]["Score"]
        avg_score /= len(individual_question_parts)
        structured_evaluation["Questions"]["Overall"][
            f"{criterion} (Exercise Mean Score)"
        ] = avg_score

    for criterion in LLM_OVERALL_QUESTION_CRITERIA:
        reasoning, score = parse_criterion(overall_question_part, criterion)
        structured_evaluation["Questions"]["Overall"][criterion] = {
            "Reasoning": reasoning,
            "Score": score,
        }
        validate_and_print(ex_id, "question evaluation", criterion, reasoning, score)

    return structured_evaluation


def calculate_all_exercises_average(all_results: list) -> dict:
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

    averages = {
        "Text": {
            criterion: sum(scores) / len(scores)
            for criterion, scores in text_scores.items()
        },
        "Questions": {
            "Individual": {
                criterion: sum(scores) / len(scores)
                for criterion, scores in question_scores.items()
            },
            "Overall": {
                criterion: sum(scores) / len(scores)
                for criterion, scores in overall_question_scores.items()
            },
        },
    }

    return averages


def create_structured_evaluations(
    generated_exercises, raw_evaluations, configs, nlp, CEFR_parser
):
    structured_evaluations = []

    for i in tqdm(range(len(generated_exercises))):
        generated_exercise = generated_exercises[i]
        raw_evaluation = raw_evaluations[i].strip()
        config = configs[i]
        structured_evaluation = evaluate_non_LLM_metrics(
            generated_exercise, config, nlp, CEFR_parser
        )
        structured_evaluation = add_LLM_evaluation(
            raw_evaluation, structured_evaluation, i
        )
        structured_evaluation["Exercise ID"] = i
        structured_evaluations.append(structured_evaluation)

    return structured_evaluations


def save_structured_evaluations(
    structured_evaluations, all_exercises_averages, output_path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Exercises": structured_evaluations,
                "Average for All Exercises": all_exercises_averages,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Processed {len(structured_evaluations)} exercises -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_path", help="File with generated exercises")
    parser.add_argument("raw_evaluation_path", help="File with raw evaluation results")
    parser.add_argument(
        "structured_evaluation_path", help="File with structured evaluation results"
    )
    args = parser.parse_args()

    if not Path(args.generated_path).is_file():
        raise FileNotFoundError(
            f"File with generated exercises is not found: {args.generated_path}"
        )

    if not Path(args.raw_evaluation_path).is_file():
        raise FileNotFoundError(
            f"File with raw evaluations is not found: {args.raw_evaluation_path}"
        )

    nlp = create_nlp()
    CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)

    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)["exercise_configs"]

    with open(args.generated_path, "r", encoding="utf-8") as f:
        generated_exercises = f.read().split(SEP)

    parsed_exercises = []
    for exercise in generated_exercises:
        parsed_exercise = parse_exercise(exercise)
        if parsed_exercise:
            parsed_exercises.append(parsed_exercise)

    with open(args.raw_evaluation_path, "r", encoding="utf-8") as f:
        raw_evaluations = f.read().split(SEP)

    structured_evaluations = create_structured_evaluations(
        parsed_exercises, raw_evaluations, configs, nlp, CEFR_parser
    )
    all_exercises_averages = calculate_all_exercises_average(structured_evaluations)
    save_structured_evaluations(
        structured_evaluations,
        all_exercises_averages,
        output_path=Path(args.structured_evaluation_path),
    )


if __name__ == "__main__":
    main()
