import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import SEP, call_LLM, create_client

load_dotenv()

MODEL_NAME = "ministral-14b-2512"

ROLE_PATH = Path("prompts/role.txt")
LEVELS = ("A2", "B1", "B2")
EXAMPLE_PATHS = {level: Path(f"prompts/example_{level}.txt") for level in LEVELS}
BASELINE_TEMPLATE_PATH = Path("prompts/baseline_template.txt")
COT_TEMPLATE_PATH = Path("prompts/CoT_template.txt")

with open(ROLE_PATH, encoding="utf-8") as f:
    role = f.read()

EXAMPLES = {}
for level, example_path in EXAMPLE_PATHS.items():
    with open(example_path, encoding="utf-8") as f:
        EXAMPLES[level] = f.read()

with open(BASELINE_TEMPLATE_PATH, encoding="utf-8") as f:
    baseline_template = f.read()

with open(COT_TEMPLATE_PATH, encoding="utf-8") as f:
    CoT_template = f.read()


def run_generation_experiment(
    client: OpenAI,
    configs: list,
    template: str,
    template_type: str = "baseline",
    role: str = "",
    with_example: bool = False,
) -> None:
    all_raw_outputs = []
    all_outputs = []

    parts = [f"generated_{template_type}"]
    if role:
        parts.append("with_role")
    if with_example:
        parts.append("few-shot")
    filename = "_".join(parts)

    result_path = Path(f"results/generated/{filename}.txt")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    for config in tqdm(configs):
        n_distractors = config["n_options"] - 1
        prompt = template.format(
            n_questions=config["n_questions"],
            n_options=config["n_options"],
            n_distractors=n_distractors,
            word_count=config["word_count"],
            topic=config["topic"],
            CEFR_level=config["CEFR_level"],
            words=", ".join(config["wordlist"]),
        )
        if with_example:
            example = EXAMPLES[config["CEFR_level"]]
            prompt = "\n".join([prompt, "\nExample:", example])

        text = ""
        questions = []

        while not text or not questions or len(text) < 100:
            response = call_LLM(
                client=client,
                user_prompt=prompt,
                system_prompt=role,
                model_name=MODEL_NAME,
            )
            response = response.replace("*", "").replace("#", "")

            response_parts = [part.strip() for part in response.split("---")]
            text = ""
            questions = []

            for part in response_parts:
                text_search = re.search(r"Text:\s*(.+)", part, re.DOTALL)
                if text_search:
                    text = text_search.group(1).strip()
                current_questions = re.findall(
                    r"Q\d+:\s*(?:.+?)\s*Correct:\s*(?:.+?)\s*Distractors:\s*(?:.+?)\s*(?:Explanation:\s*(?:.+?))?(?=Q\d+|$)",
                    part,
                    re.DOTALL,
                )
                if current_questions:
                    questions.extend(current_questions)

            questions = [question.strip() for question in questions][
                -config["n_questions"] :
            ]

            if not text or len(text) < 100:
                print(f"{filename}, {config['config_id']}: text not found, retrying...")
            elif not questions:
                print(
                    f"{filename}, {config['config_id']}: questions not found, retrying..."
                )
            elif template_type == "CoT":
                all_raw_outputs.append(response)

        output = f"""Topic: {config['topic']}

Text:
{text}

Questions:
{'\n---\n'.join(questions)}"""
        all_outputs.append(output)
        time.sleep(5)

    result_path.write_text(SEP.join(all_outputs), encoding="utf-8")

    if template_type == "CoT":
        raw_result_path = Path(f"results/generated/{filename}_raw.txt")
        raw_result_path.write_text(SEP.join(all_raw_outputs), encoding="utf-8")

    print(f"Saved {len(all_outputs)} generation(s) to {result_path}")


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)["exercise_configs"]

    client = create_client(
        api_key_name="MISTRAL_API_KEY", base_url="https://api.mistral.ai/v1"
    )

    run_generation_experiment(client, configs, baseline_template)
    run_generation_experiment(client, configs, baseline_template, role=role)
    run_generation_experiment(client, configs, baseline_template, with_example=True)
    run_generation_experiment(client, configs, CoT_template, template_type="CoT")
    run_generation_experiment(
        client, configs, CoT_template, template_type="CoT", role=role, with_example=True
    )


if __name__ == "__main__":
    main()
