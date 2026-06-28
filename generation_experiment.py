import os
import time

from utils import generate

READING_COMPREHENSION_TEXT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_text.txt"
)
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

with open(READING_COMPREHENSION_TEXT_PATH, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_ROLE_PATH, encoding="utf-8") as f:
    role = f.read()

with open(READING_COMPREHENSION_EXAMPLE_PATH, encoding="utf-8") as f:
    example = f.read()

with open(READING_COMPREHENSION_BASELINE_PATH, encoding="utf-8") as f:
    baseline_template = f.read()

with open(READING_COMPREHENSION_COT_PATH, encoding="utf-8") as f:
    CoT_template = f.read()

n_questions = 3
n_options = 4


def run_experiment(
    text,
    template,
    template_type="baseline",
    n_questions=n_questions,
    n_options=n_options,
    role="",
    example="",
):
    n_distractors = n_options - 1
    prompt = template.format(
        text=text,
        n_questions=n_questions,
        n_options=n_options,
        n_distractors=n_distractors,
    )
    if example:
        prompt = "\n".join([prompt, "\nExample:", example])

    # model = "deepseek-ai/deepseek-v4-flash"
    model = "openai/gpt-oss-120b"

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
        reasoning_effort="low",
        # extra_body={"chat_template_kwargs": {"thinking": False}},
    )

    time.sleep(10)


run_experiment(text, baseline_template)
run_experiment(text, baseline_template, role=role)
run_experiment(text, baseline_template, example=example)
run_experiment(text, CoT_template, template_type="CoT")
