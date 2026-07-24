import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_cefr_heatmap(target_levels, real_levels, title, ax=None):
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    matrix = np.zeros((len(levels), len(levels)))

    for target, real in zip(target_levels, real_levels):
        if target in levels and real in levels:
            matrix[levels.index(real), levels.index(target)] += 1

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(levels)))
    ax.set_yticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticklabels(levels)
    ax.set_xlabel("Target CEFR")
    ax.set_ylabel("Real CEFR")
    ax.set_title(title)

    for i in range(len(levels)):
        for j in range(len(levels)):
            if matrix[i, j] > 0:
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
    return ax


def extract_cefr(reasoning, text_type):
    target = re.search(r"Target CEFR level: ((?:A|B|C)(?:1|2))", reasoning).group(1)
    determined = re.search(
        rf"Determined CEFR level \({text_type}\): ((?:A|B|C)(?:1|2))", reasoning
    ).group(1)
    return target, determined


def load_exercises(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["Exercises"]


def plot_technique_heatmaps(technique, path, save_dir):
    exercises = load_exercises(path)

    target_text, real_text = zip(
        *[
            extract_cefr(ex["Text"]["Matching the CEFR Level"]["Reasoning"], "text")
            for ex in exercises
        ]
    )

    target_questions, real_questions = zip(
        *[
            extract_cefr(
                ex["Questions"]["Overall"]["Matching the CEFR Level"]["Reasoning"],
                "questions",
            )
            for ex in exercises
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    build_cefr_heatmap(target_text, real_text, "Text", ax1)
    build_cefr_heatmap(target_questions, real_questions, "Questions", ax2)
    fig.suptitle(f"{technique} CEFR Levels")
    plt.tight_layout()

    filename = technique + ".png"
    plt.savefig(save_dir / filename, bbox_inches="tight")
    plt.close()


base_path = Path("results/structured_evaluations")
save_dir = Path("images")
save_dir.mkdir(exist_ok=True)

techniques = {
    "Baseline": "structured_evaluations_baseline.json",
    "Baseline + Role": "structured_evaluations_baseline_with_role.json",
    "Baseline + Few-Shot": "structured_evaluations_baseline_few-shot.json",
    "Chain-of-Thought": "structured_evaluations_CoT.json",
}

for technique, filename in techniques.items():
    plot_technique_heatmaps(technique, base_path / filename, save_dir)
