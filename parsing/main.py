import os

from parsing.levels_db import LevelsDB

db = LevelsDB(os.path.join("data", "reading_levels.db"))
content = db.get_content("en")
print("Content:", content)
del db
