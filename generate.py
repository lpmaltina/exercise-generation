import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils import call_LLM

load_dotenv()

MODEL_NAME = "ministral-14b-2512"
BASE_URL = "https://api.mistral.ai/v1"

ROLE_PATH = Path("prompts/role.txt")
GENERATION_EXAMPLE_PATH = Path("prompts/generation_example.txt")
BASELINE_TEMPLATE_PATH = Path("prompts/baseline_template.txt")
COT_TEMPLATE_PATH = Path("prompts/CoT_template.txt")

with open(ROLE_PATH, encoding="utf-8") as f:
    role = f.read()

with open(GENERATION_EXAMPLE_PATH, encoding="utf-8") as f:
    example = f.read()

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
    example: str = "",
) -> None:
    all_outputs = []

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
        if example:
            prompt = "\n".join([prompt, "\nExample:", example])

        response = call_LLM(
            client=client, user_prompt=prompt, system_prompt=role, model_name=MODEL_NAME
        )
        parts = [
            part.strip()
            for part in response.replace("*", "").split("---")
            if part.strip()
        ]
        text, *questions = parts
        output = f"""Topic: {config['topic']}

{text}

Questions:
{'\n---\n'.join(questions)}"""
        all_outputs.append(output)
        time.sleep(10)

    parts = [f"generated_{template_type}"]
    if role:
        parts.append("with_role")
    if example:
        parts.append("few-shot")
    filename = f"{'_'.join(parts)}.txt"

    result_path = Path(f"results/generated/{filename}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("\n\n\n\n".join(all_outputs), encoding="utf-8")

    print(f"Saved {len(all_outputs)} generation(s) to {result_path}")


def main() -> int:
    with open("config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)["exercise_configs"]

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not found in .env file.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    run_generation_experiment(client, configs, baseline_template)
    run_generation_experiment(client, configs, baseline_template, role=role)
    run_generation_experiment(client, configs, baseline_template, example=example)
    run_generation_experiment(client, configs, CoT_template, template_type="CoT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
