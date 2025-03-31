from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

CHROME_DRIVER_PATH = "chromedriver.exe"

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")

service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

url = "https://www.eucafloor.com.br/linha/prime"
driver.get(url)

# Espera o body aparecer (mais seguro que time.sleep)
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, "body"))
)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# Remove tags desnecessárias
for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
    tag.extract()

text = soup.get_text(separator="\n")

with open("saida.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Conteúdo salvo com sucesso.")
driver.quit()
