import json
import random


def read_topics(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line if line.strip()]
    return topics


TOPICS_A2 = read_topics("topics/topics_A2.txt")
TOPICS_B1 = read_topics("topics/topics_B1.txt")
TOPICS_B2 = read_topics("topics/topics_B2.txt")


def read_wordlists() -> dict[str, list[str]]:
    levels = ("A1", "A2", "B1", "B2", "C1")
    words_by_level = {level: [] for level in levels}
    for level in levels:
        with open(f"wordlists/en-british-{level}.txt", "r", encoding="utf-8") as f:
            for line in f:
                word, _ = line.split("\t")
                words_by_level[level].append(word.split("(")[0].strip())
    return words_by_level


def get_random_words_by_level(
    words_by_level: dict[str, list[str]], level: str, n_words: int = 10
) -> list[str]:
    return random.sample(words_by_level[level], n_words)


def generate_configs(words_by_level: dict) -> list[dict]:
    configs = []
    config_id = 0

    for topic in TOPICS_A2:
        configs.append(
            {
                "config_id": config_id,
                "word_count": 100,
                "topic": topic,
                "CEFR_level": "A2",
                "wordlist": get_random_words_by_level(words_by_level, "A2", 10),
                "n_questions": 3,
                "n_options": 4,
            }
        )
        config_id += 1

    for topic in TOPICS_B1:
        configs.append(
            {
                "config_id": config_id,
                "word_count": 150,
                "topic": topic,
                "CEFR_level": "B1",
                "wordlist": get_random_words_by_level(words_by_level, "B1", 10),
                "n_questions": 4,
                "n_options": 4,
            }
        )
        config_id += 1

    for topic in TOPICS_B2:
        configs.append(
            {
                "config_id": config_id,
                "word_count": 200,
                "topic": topic,
                "CEFR_level": "B2",
                "wordlist": get_random_words_by_level(words_by_level, "B2", 10),
                "n_questions": 5,
                "n_options": 4,
            }
        )
        config_id += 1

    return configs


if __name__ == "__main__":
    words_by_level = read_wordlists()

    configs = generate_configs(words_by_level)
    config_path = "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"exercise_configs": configs}, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(configs)} exercise configurations")
    print(f"Saved to {config_path}")
