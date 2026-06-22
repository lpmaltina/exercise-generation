import os
import time

from utils import generate

READING_COMPREHENSION_ROLE_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_role.txt"
)
READING_COMPREHENSION_TEXT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_text.txt"
)
READING_COMPREHENSION_EVALUATION_PROMPT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_evaluation.txt"
)

READING_COMPREHENSION_BAD_EXAMPLE_PATH_PARTS = (
    "prompts",
    "reading_comprehension",
    "reading_comprehension_generation_example_bad.txt",
)
READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH_PARTS = (
    "prompts",
    "reading_comprehension",
    "reading_comprehension_generation_example_medium.txt",
)
READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH_PARTS = (
    "prompts",
    "reading_comprehension",
    "reading_comprehension_generation_example_excellent.txt",
)

with open(READING_COMPREHENSION_ROLE_PATH, encoding="utf-8") as f:
    role = f.read()

with open(READING_COMPREHENSION_TEXT_PATH, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_EVALUATION_PROMPT_PATH, encoding="utf-8") as f:
    template = f.read()

exercise_paths = (
    READING_COMPREHENSION_BAD_EXAMPLE_PATH_PARTS,
    READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH_PARTS,
    READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH_PARTS,
)

# model = "openai/gpt-oss-120b"
model = "deepseek-ai/deepseek-v4-flash"

for exercise_path_parts in exercise_paths:
    exercise_path = os.path.join(*exercise_path_parts)
    with open(exercise_path, encoding="utf-8") as f:
        exercise = f.read()

    prompt = template.format(text=text, exercise=exercise)
    result_filename = exercise_path_parts[-1].split(".")[0] + ".json"
    result_path = os.path.join(
        "results", model.split("/")[-1], "reading_comprehension", result_filename
    )

    # generate(
    #     model=model,
    #     prompt=prompt,
    #     role="",
    #     result_path=result_path,
    #     reasoning_effort="low"
    # )
    generate(
        model=model,
        prompt=prompt,
        role="",
        result_path=result_path,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )

    time.sleep(10)
