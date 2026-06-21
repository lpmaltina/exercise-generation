import time

from utils import generate

READING_COMPREHENSION_ROLE = (
    "prompts/reading_comprehension/reading_comprehension_role.txt"
)
READING_COMPREHENSION_TEXT = (
    "prompts/reading_comprehension/reading_comprehension_text.txt"
)
READING_COMPREHENSION_EVALUATION_PROMPT_PATH = (
    "prompts/reading_comprehension/reading_comprehension_evaluation.txt"
)
READING_COMPREHENSION_BAD_EXAMPLE_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_bad.txt"
)
READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH = (
    "prompts/reading_comprehension/reading_comprehension_generation_example_medium.txt"
)
READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH = "prompts/reading_comprehension/reading_comprehension_generation_example_excellent.txt"


with open(READING_COMPREHENSION_ROLE, encoding="utf-8") as f:
    role = f.read()

with open(READING_COMPREHENSION_TEXT, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_EVALUATION_PROMPT_PATH, encoding="utf-8") as f:
    template = f.read()

exercise_paths = (
    READING_COMPREHENSION_BAD_EXAMPLE_PATH,
    READING_COMPREHENSION_MEDIUM_EXAMPLE_PATH,
    READING_COMPREHENSION_EXCELLENT_EXAMPLE_PATH,
)

for exercise_path in exercise_paths:
    with open(exercise_path, encoding="utf-8") as f:
        exercise = f.read()

    prompt = template.format(text=text, exercise=exercise)
    result_path = exercise_path.split(".")[0] + ".json"
    generate(prompt, role, result_path)
    time.sleep(10)
