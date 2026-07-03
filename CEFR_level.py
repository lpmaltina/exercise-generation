import re
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class CEFRLevelParser:
    def __init__(self, url, headless=True):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self._driver = webdriver.Chrome(options=options)
        self._driver.get(url)

    def _cloze_cookie_consent_banner(self):
        try:
            accept_cookie = WebDriverWait(self._driver, 5).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[contains(text(), 'Accept') or contains(text(), 'Accept All') or contains(text(), 'OK')]",
                    )
                )
            )
            accept_cookie.click()
            time.sleep(1)  # Wait for banner to disappear
        except TimeoutException:
            pass

    def _enter_text(self, text):
        text_area = WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )
        text_area.clear()
        text_area.send_keys(text)

    def _click_button(self):
        check_button = WebDriverWait(self._driver, 10).until(
            EC.element_to_be_clickable((By.ID, "analyze-btn"))
        )
        check_button.click()

    def get_CEFR_level(self, text):
        self._cloze_cookie_consent_banner()
        self._enter_text(text)
        self._click_button()
        result_element = WebDriverWait(self._driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.score-value[class*='cefr-']")
            )
        )
        result_text = result_element.text
        match = re.search(r"\b([A-C][12])\b", result_text)
        CEFR_level = match.group(1)
        return CEFR_level

    def quit(self):
        self._driver.quit()
