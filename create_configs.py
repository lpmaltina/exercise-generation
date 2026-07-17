import json
import random

TOPICS_A2 = [
    "a story about an unusual holiday trip",
    "a story about an unusual family celebration",
    "a description of a morning routine that makes me happy",
    "a funny story that happened to my friends and me",
    "an adventure story about getting lost in the forest",
    "an adventure story about finding a secret door in the house",
    "an adventure story about finding a treasure",
    "a letter to my future self",
    "a story about a person who inspired me",
    "a story about a person who won the lottery and spent it on charity",
]

TOPICS_B1 = [
    "a personal story about learning a new skill",
    "an adventure story about a mountain trip",
    "a story about an athlete who had an injury and worked hard to recover and return to competition",
    "a story about moving to a new city and starting over",
    "a story about an unexpected act of kindness",
    "a personal story about achieving a long-term goal",
    "a story about a day in the life of an actor",
    "a story about the book that changed my life",
    "a story about a volunteer experience",
    "a personal story of an artist telling about their works",
]

TOPICS_B2 = [
    "a news article about an ecological project",
    "a sci-fi story about space travel",
    "an adventure story about time travel",
    "a story about a cross-cultural misunderstanding",
    "an adventure story about an accidental invention",
    "a news article about the advantages and disadvantages of social media",
    "an adventure story about exploring a hidden underground city",
    "a personal story about overcoming a significant fear",
    "a sci-fi story about artificial intelligence",
    "a personal story about achieving a work-life balance",
]


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

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump({"exercise_configs": configs}, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(configs)} exercise configurations")
    print(f"Saved to config.json")
