"""The module parsing.py creates or updates the dataset
with articles in English and German.
The data is taken from
- https://www.newsinlevels.com/
- https://germaninlevels.com/
respectively.
"""

import os
import random
from collections import namedtuple
from pathlib import Path
from time import sleep
from typing import Dict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from parsing.levels_db import LevelsDB

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 "
        "Safari/537.36"
    )
}
Article = namedtuple("Article", "date level title text question")


class LevelsParser:
    """The class LevelsParser parses the websites and
    adds the articles to the database"""

    def __init__(self, lang2url: Dict[str, str]) -> None:
        """Create the instance of the LevelsParser class.
        lang2url is a dictionary with languages names as keys and
        parts of the URLs to the webpages as values"""
        self._n_levels = 3
        self._lang2url = lang2url

    def get_soup(self, url: str) -> BeautifulSoup:
        """Return the contents of a page with a given URL"""
        while True:
            try:
                response = requests.get(url, HEADERS, timeout=60)
                break
            except requests.exceptions.RequestException:
                sleep(random.randint(5, 10))

        return BeautifulSoup(response.text, "html.parser")

    def get_n_pages(self, language: str, level: int) -> int:
        """Return a number of pages for a given language and level"""
        url = f"{self._lang2url[language]}-{level}/"
        soup_page = self.get_soup(url)
        pagination = soup_page.find("ul", {"class": "pagination"})
        last_article_link = pagination.find_all(href=True)[-1].get("href")
        n_pages = int(last_article_link.split("/page/")[1].split("/")[0])
        return n_pages

    def get_titles_and_article_urls(
        self, language: str, level: int, n_page: int
    ) -> Dict[str, str]:
        """Get titles and article URLs for a given language, level and page number.
        Return a dictionary with titles as keys and article URLs as values."""
        page_url = f"{self._lang2url[language]}-{level}/page/{n_page}/"
        soup_page = self.get_soup(page_url)
        blocks = soup_page.find_all("div", {"class": "news-block-right"})
        title2article_url = {}

        for block in blocks:
            heading = block.find(href=True)
            article_url = heading.get("href")
            title = heading.text.split("–")[0].split("-")[0].strip()
            title2article_url[title] = article_url

        return title2article_url

    def get_article(self, title: str, article_url: str, level: int) -> Article:
        """Create article by a given title, article URL and level.
        Return a NamedTuple with fields:
        - date (str)
        - level (int)
        - title (str)
        - text (str)
        """
        soup_article = self.get_soup(article_url)
        article_content = soup_article.find("div", {"id": "nContent"})
        contents = article_content.find_all("p")
        contents = [content.get_text().strip() for content in contents if content]
        date = contents[0]
        text_list = []

        for i in range(1, len(contents) - 1):
            element = contents[i]
            if element.startswith("Difficult words:"):
                break
            text_list.append(element)

        text = " ".join(text_list)

        question = soup_article.find("div", {"class": "fancy-facebook-block"})
        if question:
            question = question.get_text().strip()
        else:
            question = ""

        return Article(date, level, title, text, question)

    def create_or_update_dataset(self, dataset_path: str) -> None:
        levels_db = LevelsDB(db_name=dataset_path)

        for lang in self._lang2url:
            for level in range(1, self._n_levels + 1):
                n_pages = self.get_n_pages(lang, level)
                for n_page in tqdm(range(1, n_pages + 1)):
                    title2article_url = self.get_titles_and_article_urls(
                        lang, level, n_page
                    )
                    for title, article_url in title2article_url.items():
                        article = self.get_article(title, article_url, level)
                        article_id = levels_db.add_article(
                            article.date, lang, level, title, article.text
                        )
                        if article.question:
                            levels_db.add_questions([(article_id, article.question)])
                    sleep(random.randint(1, 3))

        del levels_db


if __name__ == "__main__":
    url_parts = {
        "en": "https://www.newsinlevels.com/level/level-",
        "de": "https://germaninlevels.com/level/stufe-",
    }
    levels_parser = LevelsParser(url_parts)
    base_path = os.path.join("data")
    full_path = os.path.join(base_path, "reading_levels.db")
    Path(base_path).mkdir(parents=True, exist_ok=True)
    levels_parser.create_or_update_dataset(full_path)
