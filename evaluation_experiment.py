import json
import os
import re

import spacy
from spacy.lang.char_classes import ALPHA
from spacy.tokenizer import Tokenizer

import config
from CEFR_level import CEFRLevelParser


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


def get_text_and_questions(filepath: str) -> tuple[str, list[str]]:
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    generated = data["choices"][0]["message"]["content"]
    parts = [
        part.strip().replace("*", "").strip() for part in generated.split("---") if part
    ]
    text_idx = next((i for i, p in enumerate(parts) if p.startswith("Text:")), 0)
    text, *questions = parts[text_idx:]
    text = text.replace("Text:", "").strip()
    return text, questions


def tokenize(text: str) -> list[spacy.tokens.token.Token]:
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


nlp = create_nlp()
CEFR_parser = CEFRLevelParser("https://textlevel.org/", headless=True)
variants = ("baseline", "baseline_with_role", "baseline_few-shot", "CoT")
words = config.words

for variant in variants:
    print(variant.upper())
    result_path = os.path.join(
        "results",
        "ministral-14b-instruct-2512",
        "reading_comprehension",
        f"reading_comprehension_{variant}.json",
    )
    text, questions = get_text_and_questions(result_path)
    print(text)
    tokens = tokenize(text)

    word_count_passed = int(
        check_word_count(tokens, target_word_count=config.word_count)
    )
    print(f"Word count passed: {word_count_passed}/1")

    used_count, word_count, unused = check_words_from_wordlist(tokens, words)
    print(f"Words used: {used_count}/{word_count}")
    if unused:
        print(f"Unused words: {', '.join(unused)}")

    CEFR_level = CEFR_parser.get_CEFR_level(text)
    print(f"Target CEFR level: {config.CEFR_level}")
    print(f"Determined CEFR level: {CEFR_level}")
    print("Do levels match:", config.CEFR_level == CEFR_level)

    are_words_evenly_distributed = check_evenness(tokens, words)
    print("Are the words evenly distributed:", are_words_evenly_distributed)
    print()


CEFR_parser.quit()
