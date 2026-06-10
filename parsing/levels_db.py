"""The module creates or updates the database
containing articles written for different levels
of foreign language proficiency
"""

import sqlite3
from typing import List


class LevelsDB:
    """The class LevelsDB works with the database
    containing articles written for different levels
    of foreign language proficiency"""

    def __init__(self, db_name: str) -> None:
        """Create a connection and a cursor to a database with a given name.
        If the database doesn't exist yet, create it.
        If the tables Articles and Questions don't exist, create them."""
        self._conn = sqlite3.connect(db_name)
        self._cur = self._conn.cursor()
        self._create_table_articles()
        self._create_table_questions()

    def _create_table_articles(self) -> None:
        """If the table Articles doesn't exist yet, create it.
        Fields:
        - article_id INTEGER PRIMARY KEY
        - date DATE
        - language VARCHAR(2) NOT NULL
        - level INTEGER NOT NULL
        - title TEXT
        - article_text TEXT NOT NULL
        Article texts are unique
        """
        self._cur.execute("""CREATE TABLE IF NOT EXISTS Articles(
            article_id INTEGER PRIMARY KEY,
            date DATE,
            language VARCHAR(2) NOT NULL,
            level INTEGER NOT NULL,
            title TEXT,
            article_text TEXT NOT NULL,
            UNIQUE(article_text) ON CONFLICT IGNORE)""")
        self._conn.commit()

    def _create_table_questions(self) -> None:
        """If the table Questions doesn't exist yet, create it.
        Fields:
        - question_id INTEGER PRIMARY KEY
        - article_id INTEGER NOT NULL
        - question_text TEXT NOT NULL
        """
        self._cur.execute(
            """CREATE TABLE IF NOT EXISTS Questions(
            question_id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES Articles (article_id) ON DELETE CASCADE)"""
        )
        self._conn.commit()

    def add_article(
        self, date: str, language: str, level: int, title: str, text: str
    ) -> int:
        self._cur.execute(
            """INSERT INTO Articles(date, language, level, title, article_text)
            VALUES(?,?,?,?,?)""",
            (date, language, level, title, text),
        )
        self._conn.commit()
        return self._cur.lastrowid

    def add_articles(self, articles: list) -> None:
        """Add articles to the table Articles"""
        self._cur.executemany(
            """INSERT INTO Articles(
            date, language, level, title, article_text
            ) VALUES(?,?,?,?,?)""",
            articles,
        )
        self._conn.commit()

    def add_questions(self, questions: list) -> None:
        """Add questions to the table Questions"""
        self._cur.executemany(
            """INSERT INTO Questions(
            article_id, question_text
            ) VALUES(?,?)""",
            questions,
        )
        self._conn.commit()

    def get_content(self, lang: str) -> List[List]:
        """Get content of the database for a given language.

        Return a list.
        Each element of it is a row.
        The elements of each row:
        0) level
        1) title
        2) article text
        3) question text"""
        self._cur.execute(
            """SELECT level, title, article_text, question_text
            FROM Articles
            LEFT JOIN Questions
            ON Articles.article_id = Questions.article_id
            WHERE language = ?""",
            (lang,),
        )
        return self._cur.fetchall()

    def __del__(self) -> None:
        """Close the cursor and the connection to the database"""
        self._cur.close()
        self._conn.close()
