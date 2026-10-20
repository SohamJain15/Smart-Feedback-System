"""Browser smoke test against the containerized application."""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8080")
# Empty/unset SELENIUM_URL -> use Chrome installed on this machine; set it for a remote/containerised Selenium.
SELENIUM_URL = os.getenv("SELENIUM_URL", "")

def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if os.getenv("CHROME_BIN"):
        options.binary_location = os.environ["CHROME_BIN"]
    if not SELENIUM_URL:
        return webdriver.Chrome(options=options)
    last_error = None
    for _ in range(20):
        try:
            return webdriver.Remote(command_executor=SELENIUM_URL, options=options)
        except Exception as error:
            last_error = error
            time.sleep(2)
    raise last_error

def test_submit_analyse_save_and_open_dashboard():
    driver = create_driver()
    wait = WebDriverWait(driver, 25)
    try:
        driver.get(f"{APP_URL}/feedback")
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, "textarea")))
        Select(driver.find_elements(By.TAG_NAME, "select")[0]).select_by_visible_text("Laboratory")
        driver.find_element(By.TAG_NAME, "textarea").send_keys("The laboratory computers are not good during automated testing.")
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Analyse feedback')]"))).click()
        wait.until(EC.visibility_of_element_located((By.XPATH, "//*[contains(., 'Model result')]")))
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Save feedback')]"))).click()
        wait.until(EC.visibility_of_element_located((By.XPATH, "//*[contains(., 'Feedback was saved')]")))
        driver.get(f"{APP_URL}/dashboard")
        wait.until(EC.visibility_of_element_located((By.XPATH, "//*[contains(., 'Total feedback')]")))
    finally:
        driver.quit()
