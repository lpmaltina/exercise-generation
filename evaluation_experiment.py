import json
import os
import re
import time

import spacy
from spacy.lang.char_classes import ALPHA
from spacy.tokenizer import Tokenizer

import config
from CEFR_level import CEFRLevelParser
from utils import generate

ROLE_PROMPT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_role.txt"
)
EVALUATION_PROMPT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_evaluation.txt"
)
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


def create_nlp():
    nlp = spacy.load("en_core_web_sm")
    infixes = list(nlp.Defaults.infixes)

    # Делаем так, чтобы токены не разбивались по дефисам и апострофам
    patterns_to_remove = {
        r"(?<=[{a}])-(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])'(?=[{a}])".format(a=ALPHA),
    }

    infixes = [p for p in infixes if p not in patterns_to_remove]
    infix_re = re.compile("|".join(infixes)) if infixes else None

    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer if infix_re else None,
        token_match=nlp.tokenizer.token_match,
        url_match=nlp.tokenizer.url_match,
    )
    return nlp


def get_text_and_questions(filepath: str) -> tuple[str, str]:
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        generated = data["choices"][0]["message"]["content"]

    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as file:
            generated = file.read()

    parts = [
        part.strip().replace("*", "").strip() for part in generated.split("---") if part
    ]

    text = ""
    questions = []

    for part in parts:
        if part.startswith("Text:"):
            text = part
        elif re.search(
            r"Q\d+:(?:.+?)\nCorrect:(?:.+?)\nDistractors:(?:.+?)\nExplanation:(?:.+?)",
            part,
            re.DOTALL,
        ):
            questions.append(part)

    return text.split(":", maxsplit=1)[1].strip(), "\n---\n".join(questions)


def tokenize(nlp, text: str) -> list[spacy.tokens.token.Token]:
    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not token.is_punct and not token.is_space and not token.is_digit
    ]
    return tokens


def check_word_count(
    tokens: list[spacy.tokens.token.Token], target_word_count: int
) -> bool:
    return target_word_count * 0.9 <= len(tokens) <= target_word_count * 1.1


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


def check_evenness(tokens: spacy.tokens.doc.Doc, words: set[str]) -> bool:
    words_copy = words.copy()
    len_text = len(tokens)
    len_wordlist = len(words)
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

    mean_distance = len_text / len_wordlist
    threshold = 3 * mean_distance

    max_gap = positions[0]

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i] - 1
        max_gap = max(max_gap, gap)

    gap = len_text - positions[-1] - 1
    max_gap = max(max_gap, gap)

    return max_gap <= threshold


def parse_criterion(text: str, criterion: str) -> tuple[str, int]:
    reasoning = ""
    score = 0

    match = re.search(rf"{criterion}.\s+(.+?)\s+(\d)/5", text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        score = int(match.group(2))

    return reasoning, score


def LLM_evaluate(
    text: str,
    questions: str,
    topic: str,
    evaluation_path: str,
    role: str,
    evaluation_template: str,
):
    evaluation_model = "openai/gpt-oss-120b"
    prompt = evaluation_template.format(text=text, questions=questions, topic=topic)

    evaluation_result = generate(
        model=evaluation_model,
        prompt=prompt,
        role=role,
        result_path=evaluation_path,
    )
    time.sleep(10)

    return evaluation_result


def evaluate(
    nlp,
    CEFR_parser,
    words,
    role,
    evaluation_template,
    generation_path,
    evaluation_path,
    summary_path,
):
    text, questions = get_text_and_questions(generation_path)
    tokens = tokenize(nlp, text)

    word_count_score = int(
        check_word_count(tokens, target_word_count=config.word_count)
    )
    word_count_reasoning = (
        f"Words: {len(tokens)}. Target word count: {config.word_count}."
    )

    used_count, target_word_count, unused = check_words_from_wordlist(tokens, words)
    target_word_usage_score = used_count / target_word_count
    target_word_usage_reasoning = [
        f"Target word usage: {used_count}/{target_word_count}."
    ]
    if unused:
        target_word_usage_reasoning.append(f"Unused words: {', '.join(unused)}.")
    target_word_usage_reasoning = " ".join(target_word_usage_reasoning)

    target_word_distribution_score = int(check_evenness(tokens, words))
    target_word_distribution_reasoning = (
        "The distance between target words should not be more than "
        "3 * number of words in the text / number of target words."
    )

    CEFR_level_text = CEFR_parser.get_CEFR_level(text)
    CEFR_level_text_score = int(config.CEFR_level == CEFR_level_text)
    CEFR_level_text_reasoning = f"Target CEFR level: {config.CEFR_level}. Determined CEFR level (text): {CEFR_level_text}"

    CEFR_level_questions = CEFR_parser.get_CEFR_level(questions)
    CEFR_level_questions_score = int(config.CEFR_level == CEFR_level_questions)
    CEFR_level_questions_reasoning = f"Target CEFR level: {config.CEFR_level}. Determined CEFR level (questions): {CEFR_level_questions}"

    evaluation_raw = LLM_evaluate(
        text, questions, config.topic, evaluation_path, role, evaluation_template
    )

    evaluation_results = {
        "Text": {
            "Content": text,
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
            "Matching the Topic": {"Reasoning": "", "Score": 0},
            "Logic & Commonsense": {"Reasoning": "", "Score": 0},
            "Vocabulary & Grammar": {"Reasoning": "", "Score": 0},
        },
        "Questions": {
            "Content": questions,
            "Overall": {
                "Matching the CEFR Level": {
                    "Reasoning": CEFR_level_questions_reasoning,
                    "Score": CEFR_level_questions_score,
                },
                "Text Coverage": {"Reasoning": "", "Score": 0},
            },
        },
    }

    evaluation_raw = evaluation_raw.replace("*", "")
    parts = [s.strip() for s in evaluation_raw.split("---") if s.strip()]
    text_part, *individual_question_parts, overall_question_part = parts

    for text_criterion in TEXT_CRITERIA:
        reasoning, score = parse_criterion(text_part, text_criterion)
        evaluation_results["Text"][text_criterion]["Reasoning"] = reasoning
        evaluation_results["Text"][text_criterion]["Score"] = score

    for i, question in enumerate(individual_question_parts, start=1):
        evaluation_results["Questions"][f"Q{i}"] = {}

    for question_criterion in INDIVIDUAL_QUESTION_CRITERIA:
        avg_score = 0
        for i, question in enumerate(individual_question_parts, start=1):
            reasoning, score = parse_criterion(question, question_criterion)
            evaluation_results["Questions"][f"Q{i}"][question_criterion] = {
                "Reasoning": reasoning,
                "Score": score,
            }
            avg_score += score
        avg_score /= config.n_questions
        evaluation_results["Questions"]["Overall"][
            f"{question_criterion} (Average)"
        ] = avg_score

    for question_criterion in OVERALL_QUESTION_CRITERIA:
        reasoning, score = parse_criterion(overall_question_part, question_criterion)
        evaluation_results["Questions"]["Overall"][question_criterion][
            "Reasoning"
        ] = reasoning
        evaluation_results["Questions"]["Overall"][question_criterion]["Score"] = score

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    return evaluation_results


if __name__ == "__main__":
    with open(ROLE_PROMPT_PATH, encoding="utf-8") as f:
        role = f.read()

    with open(EVALUATION_PROMPT_PATH, encoding="utf-8") as f:
        evaluation_template = f.read()

    nlp = create_nlp()
    CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)
    variants = ("baseline", "baseline_with_role", "baseline_few-shot", "CoT")
    words = config.words

    for variant in variants:
        generation_model = "ministral-14b-instruct-2512"
        generation_path = os.path.join(
            "results",
            "generation",
            generation_model,
            "reading_comprehension",
            f"reading_comprehension_{variant}.json",
        )
        evaluation_path = os.path.join(
            "results",
            "evaluation",
            generation_model,
            "reading_comprehension",
            f"reading_comprehension_{variant}.json",
        )
        summary_path = os.path.join(
            "results",
            "summary",
            generation_model,
            "reading_comprehension",
            f"reading_comprehension_{variant}.json",
        )
        evaluation_results = evaluate(
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
