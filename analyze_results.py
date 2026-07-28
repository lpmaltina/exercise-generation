import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from utils import (
    ALL_CRITERIA,
    ALL_INDIVIDUAL_QUESTION_CRITERIA,
    ALL_OVERALL_QUESTION_CRITERIA,
    ALL_TEXT_CRITERIA,
)

P_VALUE_THRESHOLD = 0.05


def load_scores_from_data(data):
    exercises = data["Exercises"]

    text_scores = {c: [] for c in ALL_TEXT_CRITERIA}
    question_scores = {c: [] for c in ALL_INDIVIDUAL_QUESTION_CRITERIA}
    overall_question_scores = {c: [] for c in ALL_OVERALL_QUESTION_CRITERIA}

    for ex in exercises:
        for c in ALL_TEXT_CRITERIA:
            text_scores[c].append(ex["Text"][c]["Score"])

        for c in ALL_INDIVIDUAL_QUESTION_CRITERIA:
            question_scores[c].append(
                ex["Questions"]["Overall"][f"{c} (Exercise Mean Score)"]
            )

        for c in ALL_OVERALL_QUESTION_CRITERIA:
            overall_question_scores[c].append(ex["Questions"]["Overall"][c]["Score"])

    return {
        "Text": text_scores,
        "Questions": {
            "Individual": question_scores,
            "Overall": overall_question_scores,
        },
    }


def extract_word_counts_from_data(data):
    counts = []
    for ex in data["Exercises"]:
        reasoning = ex["Text"]["Word Count"]["Reasoning"]
        counts.append(int(reasoning.split("Words:")[1].split(".")[0].strip()))
    return counts


def split_by_level(data):
    return {"A2": data[:10], "B1": data[10:20], "B2": data[20:30]}


def get_significant_criteria(all_scores):
    baseline_scores = all_scores["Baseline"]
    significant = {}

    for name, scores in all_scores.items():
        if name == "Baseline":
            continue

        print(f"H0: {name} is not better than Baseline\n")

        significant[name] = []

        for group_type, criteria in ALL_CRITERIA:
            for criterion in criteria:
                if group_type == "text":
                    baseline = baseline_scores["Text"][criterion]
                    other = scores["Text"][criterion]
                elif group_type == "individual question":
                    baseline = baseline_scores["Questions"]["Individual"][criterion]
                    other = scores["Questions"]["Individual"][criterion]
                else:
                    baseline = baseline_scores["Questions"]["Overall"][criterion]
                    other = scores["Questions"]["Overall"][criterion]

                print(f"{(group_type + ' criteria').upper()} {criterion}:")

                diff = np.array(other) - np.array(baseline)
                if np.all(diff == 0):
                    print(f"  P-value: 1.0")
                    print()
                    continue

                statistic, p_value = stats.wilcoxon(
                    other, baseline, alternative="greater"
                )
                print(f"  Statistic: {statistic}")
                print(f"  P-value: {p_value}")
                print()

                if p_value < P_VALUE_THRESHOLD:
                    significant[name].append((group_type, criterion, p_value))

    return significant


def extract_CEFR(reasoning, text_type):
    target = re.search(r"Target CEFR level: ((?:A|B|C)(?:1|2))", reasoning).group(1)
    determined = re.search(
        rf"Determined CEFR level \({text_type}\): ((?:A|B|C)(?:1|2))", reasoning
    ).group(1)
    return target, determined


def plot_boxplot(data_dict, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set3", n_colors=len(data_dict))

    bp = ax.boxplot(list(data_dict.values()), patch_artist=True)
    ax.set_xticklabels(data_dict.keys(), rotation=45, ha="right")

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_word_counts(all_data, techniques, save_dir):
    word_counts = {}
    for name, data in all_data.items():
        word_counts[name] = extract_word_counts_from_data(data)

    targets = {"A2": 100, "B1": 150, "B2": 200}
    level_data = {
        level: {name: split_by_level(word_counts[name])[level] for name in techniques}
        for level in targets
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = sns.color_palette("Set3", n_colors=4)

    for idx, level in enumerate(targets):
        ax = axes[idx]
        data_to_plot = [level_data[level][name] for name in techniques.keys()]

        bp = ax.boxplot(data_to_plot, patch_artist=True)
        ax.set_xticklabels(techniques.keys(), rotation=45, ha="right")

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        for median in bp["medians"]:
            median.set_color("black")

        ax.set_title(level)
        ax.set_ylabel("Word count")
        ax.axhline(y=targets[level], color="green", linestyle="--")
        ax.axhspan(
            targets[level] * 0.9, targets[level] * 1.1, color="lightgrey", alpha=0.7
        )

    plt.suptitle("Word Count")
    plt.tight_layout()
    plt.savefig(save_dir / "word_count_boxplots.png", bbox_inches="tight")
    plt.close()


def plot_CEFR_heatmaps(technique, exercises, save_dir):
    data_pairs = {
        "Text": zip(
            *[
                extract_CEFR(ex["Text"]["Matching the CEFR Level"]["Reasoning"], "text")
                for ex in exercises
            ]
        ),
        "Questions": zip(
            *[
                extract_CEFR(
                    ex["Questions"]["Overall"]["Matching the CEFR Level"]["Reasoning"],
                    "questions",
                )
                for ex in exercises
            ]
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    levels = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    n_levels = len(levels)

    for ax, (title, (targets, reals)) in zip(axes, data_pairs.items()):
        matrix = np.zeros((n_levels, n_levels))
        for target, real in zip(targets, reals):
            matrix[levels[real], levels[target]] += 1

        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(n_levels))
        ax.set_yticks(range(n_levels))
        ax.set_xticklabels(levels)
        ax.set_yticklabels(levels)
        ax.set_xlabel("Target CEFR")
        ax.set_ylabel("Real CEFR")
        ax.set_title(title)

        for i in range(n_levels):
            for j in range(n_levels):
                if matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        int(matrix[i, j]),
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    )

    title = f"{technique} CEFR Levels"
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_dir / f"{title}.png", bbox_inches="tight")
    plt.close()


def main():
    base_path = Path("results/structured_evaluations")
    save_dir = Path("images")
    save_dir.mkdir(exist_ok=True)

    techniques = {
        "Baseline": base_path / "structured_evaluations_baseline.json",
        "Baseline + Role": base_path / "structured_evaluations_baseline_with_role.json",
        "Baseline + Few-Shot": base_path
        / "structured_evaluations_baseline_few-shot.json",
        "Chain-of-Thought": base_path / "structured_evaluations_CoT.json",
    }

    all_raw_data = {}
    for name, path in techniques.items():
        with open(path, "r", encoding="utf-8") as f:
            all_raw_data[name] = json.load(f)

    all_scores = {
        name: load_scores_from_data(data) for name, data in all_raw_data.items()
    }
    significant = get_significant_criteria(all_scores)

    for criteria_list in significant.values():
        for group_type, criterion, _ in criteria_list:
            data = {}
            for name in techniques:
                if group_type == "text":
                    data[name] = all_scores[name]["Text"][criterion]
                elif group_type == "individual question":
                    data[name] = all_scores[name]["Questions"]["Individual"][criterion]
                else:
                    data[name] = all_scores[name]["Questions"]["Overall"][criterion]

            plot_boxplot(data, criterion, save_dir / f"{criterion} ({group_type}).png")

    plot_word_counts(all_raw_data, techniques, save_dir)

    for name, data in all_raw_data.items():
        plot_CEFR_heatmaps(name, data["Exercises"], save_dir)


if __name__ == "__main__":
    main()
