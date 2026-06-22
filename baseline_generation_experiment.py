import os

from utils import generate

READING_COMPREHENSION_TEXT_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_text.txt"
)
READING_COMPREHENSION_BASELINE_PATH = os.path.join(
    "prompts", "reading_comprehension", "reading_comprehension_baseline.txt"
)

with open(READING_COMPREHENSION_TEXT_PATH, encoding="utf-8") as f:
    text = f.read()

with open(READING_COMPREHENSION_BASELINE_PATH, encoding="utf-8") as f:
    template = f.read()

n_options = 4
n_distractors = n_options - 1
prompt = template.format(
    text=text, n_questions=3, n_options=n_options, n_distractors=n_distractors
)

# model = "openai/gpt-oss-120b"
model = "deepseek-ai/deepseek-v4-flash"
result_path = os.path.join(
    "results",
    model.split("/")[-1],
    "reading_comprehension",
    "reading_comprehension_baseline.json",
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
