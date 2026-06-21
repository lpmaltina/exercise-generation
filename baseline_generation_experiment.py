from utils import generate

READING_COMPREHENSION_TEXT = (
    "prompts/" "reading_comprehension/reading_comprehension_text.txt"
)
READING_COMPREHENSION_BASELINE_PATH = (
    "prompts/reading_comprehension/reading_comprehension_baseline.txt"
)

with open(READING_COMPREHENSION_TEXT, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_BASELINE_PATH, encoding="utf-8") as f:
    template = f.read()

n_options = 4
n_distractors = n_options - 1
prompt = template.format(
    text=text, n_questions=3, n_options=n_options, n_distractors=n_distractors
)
result_path = READING_COMPREHENSION_BASELINE_PATH.split(".")[0] + ".json"
generate(prompt, role="", result_path=result_path)
