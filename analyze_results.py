import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    return {"A2": data[:100], "B1": data[100:200], "B2": data[200:300]}


def create_significance_results_csv(all_scores, results_dir):
    baseline_scores = all_scores["Baseline"]

    csv_path = results_dir / "statistical_significance_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Alternative", "Criterion Type", "Criterion", "p-value"])

        for name, scores in all_scores.items():
            if name == "Baseline":
                continue

            for group_type, criteria in ALL_CRITERIA:
                if group_type == "text":
                    group_name = "Text Criteria"
                elif group_type == "individual question":
                    group_name = "Individual Question Criteria"
                else:
                    group_name = "Overall Question Criteria"

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

                    diff = np.array(other) - np.array(baseline)
                    if np.all(diff == 0):
                        writer.writerow([name, group_name, criterion, 1.0])
                        continue

                    _, p_value = stats.wilcoxon(other, baseline, alternative="greater")
                    writer.writerow([name, group_name, criterion, p_value])

    print(f"P-values written to {csv_path}")


def extract_CEFR(reasoning, text_type):
    target = re.search(r"Target CEFR level: ((?:A|B|C)(?:1|2))", reasoning).group(1)
    determined = re.search(
        rf"Determined CEFR level \({text_type}\): ((?:A|B|C)(?:1|2))", reasoning
    ).group(1)
    return target, determined


def plot_word_counts(all_data, techniques, save_dir):
    word_counts = {}
    for name, data in all_data.items():
        word_counts[name] = extract_word_counts_from_data(data)

    targets = {"A2": 100, "B1": 150, "B2": 200}
    level_data = {
        level: {name: split_by_level(word_counts[name])[level] for name in techniques}
        for level in targets
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
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
    plt.savefig(save_dir / "word_count_boxplots.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(save_dir / f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_results_csv(all_scores, techniques, save_dir):
    csv_path = save_dir / "results.csv"
    data_rows = []

    for group_type, criteria in ALL_CRITERIA:
        group_name = (
            "Text Criteria"
            if group_type == "text"
            else (
                "Individual Question Criteria"
                if group_type == "individual question"
                else "Overall Question Criteria"
            )
        )
        for criterion in criteria:
            row = [group_name, criterion]
            for name in techniques:
                if group_type == "text":
                    scores = all_scores[name]["Text"][criterion]
                elif group_type == "individual question":
                    scores = all_scores[name]["Questions"]["Individual"][criterion]
                else:
                    scores = all_scores[name]["Questions"]["Overall"][criterion]
                row.append(round(sum(scores) / len(scores), 4))
            data_rows.append(row)

    columns = ["Criterion Type", "Criterion"]
    columns.extend(techniques.keys())
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Results written to {csv_path}")
    return df


def create_radar_plot(df, title, y_min, y_max, save_dir):
    criteria = df["Criterion"]
    n_criteria = len(criteria)

    angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    techniques = df.columns.drop(["Criterion Type", "Criterion"])

    for technique in techniques:
        values = df[technique].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=technique)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.set_ylim(y_min, y_max)
    plt.title(title, y=1.05)
    plt.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.savefig(save_dir / f"{title}.png", dpi=300, bbox_inches="tight")


def main():
    base_path = Path("results/structured_evaluations")
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    techniques = {
        "Baseline": base_path / "structured_evaluations_baseline.json",
        "Baseline + Role": base_path / "structured_evaluations_baseline_with_role.json",
        "Baseline + Few-Shot": base_path
        / "structured_evaluations_baseline_few-shot.json",
        "Chain-of-Thought": base_path / "structured_evaluations_CoT.json",
        "Chain-of-Thought + Role + Few-Shot": base_path
        / "structured_evaluations_CoT_with_role_few-shot.json",
    }

    all_raw_data = {}
    for name, path in techniques.items():
        with open(path, "r", encoding="utf-8") as f:
            all_raw_data[name] = json.load(f)

    all_scores = {
        name: load_scores_from_data(data) for name, data in all_raw_data.items()
    }

    df = create_results_csv(all_scores, techniques, results_dir)
    create_significance_results_csv(all_scores, results_dir)
    plot_word_counts(all_raw_data, techniques, images_dir)

    for name, data in all_raw_data.items():
        plot_CEFR_heatmaps(name, data["Exercises"], images_dir)

    text_data = df[df["Criterion Type"] == "Text Criteria"]
    question_data = df[
        df["Criterion Type"].isin(
            ["Individual Question Criteria", "Overall Question Criteria"]
        )
    ]
    create_radar_plot(
        text_data, "Text Criteria", y_min=0.7, y_max=1.0, save_dir=images_dir
    )
    create_radar_plot(
        question_data, "Question Criteria", y_min=0.5, y_max=1.0, save_dir=images_dir
    )


if __name__ == "__main__":
    main()
