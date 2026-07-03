import json
import os
import re

import spacy

import config


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:[-'’–][a-zA-Z0-9]+)*", text)


def check_word_count(tokens: list[str], target_word_count: int) -> bool:
    return target_word_count * 0.9 <= len(tokens) <= target_word_count * 1.1


def check_words_from_wordlist(
    doc: spacy.tokens.doc.Doc, words: set[str]
) -> tuple[int, int, set[str]]:
    used = set()
    unused = set(words)
    for token in doc:
        lemma = token.lemma_.lower()
        word_form = token.text.lower()
        if lemma in words:
            used.add(lemma)
            unused.discard(lemma)
        elif word_form in words:
            used.add(word_form)
            unused.discard(word_form)
    return len(used), len(words), unused


nlp = spacy.load("en_core_web_sm")
variants = ("baseline", "baseline_with_role", "baseline_few-shot", "CoT")

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
    doc = nlp(text)
    tokens = tokenize(text)

    word_count_passed = int(
        check_word_count(tokens, target_word_count=config.word_count)
    )
    print(f"Word count passed: {word_count_passed}/1")

    used_count, word_count, unused = check_words_from_wordlist(doc, words=config.words)
    print(f"Words used: {used_count}/{word_count}")
    if unused:
        print(f"Unused words: {', '.join(unused)}")
    print()
