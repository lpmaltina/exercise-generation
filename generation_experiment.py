import os
import time

import config
from utils import generate

READING_COMPREHENSION_ROLE_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_role.txt"
)
READING_COMPREHENSION_EXAMPLE_PATH = os.path.join(
    "prompts",
    "reading_comprehension",
    "reading_comprehension_generation_example_excellent.txt",
)
READING_COMPREHENSION_BASELINE_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_baseline.txt"
)
READING_COMPREHENSION_COT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_CoT.txt"
)

with open(READING_COMPREHENSION_ROLE_PATH, encoding="utf-8") as f:
    role = f.read()

with open(READING_COMPREHENSION_EXAMPLE_PATH, encoding="utf-8") as f:
    example = f.read()

with open(READING_COMPREHENSION_BASELINE_PATH, encoding="utf-8") as f:
    baseline_template = f.read()

with open(READING_COMPREHENSION_COT_PATH, encoding="utf-8") as f:
    CoT_template = f.read()


def run_experiment(
    template,
    template_type="baseline",
    n_questions=config.n_questions,
    n_options=config.n_options,
    word_count=config.word_count,
    topic=config.topic,
    CEFR_level=config.CEFR_level,
    words=config.words,
    role="",
    example="",
):
    n_distractors = n_options - 1
    prompt = template.format(
        n_questions=n_questions,
        n_options=n_options,
        n_distractors=n_distractors,
        word_count=word_count,
        topic=topic,
        CEFR_level=CEFR_level,
        words=words,
    )
    if example:
        prompt = "\n".join([prompt, "\nExample:", example])

    model = "mistralai/ministral-14b-instruct-2512"
    # model = "deepseek-ai/deepseek-v4-flash"
    # model = "openai/gpt-oss-120b"

    parts = [f"reading_comprehension_{template_type}"]
    if role:
        parts.append("with_role")
    if example:
        parts.append("few-shot")
    filename = f"{'_'.join(parts)}.json"

    result_path = os.path.join(
        "results", model.split("/")[-1], "reading_comprehension", filename
    )

    generate(
        model=model,
        prompt=prompt,
        role=role,
        result_path=result_path,
        # reasoning_effort="low",
        # extra_body={"chat_template_kwargs": {"thinking": False}},
    )

    time.sleep(10)


run_experiment(baseline_template)
run_experiment(baseline_template, role=role)
run_experiment(baseline_template, example=example)
run_experiment(CoT_template, template_type="CoT")
