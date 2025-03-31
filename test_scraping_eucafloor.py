# test_scraping_eucafloor.py
import requests
from bs4 import BeautifulSoup

url = "https://www.eucafloor.com.br/linha/prime"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# Remove elementos de navegação, scripts, estilos etc.
for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
    tag.extract()

# Extrai e limpa o texto
clean_text = soup.get_text(separator="\n")

# Salva em arquivo para leitura mais confortável
with open("saida_eucafloor_prime.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("[✓] Texto extraído e salvo em: saida_eucafloor_prime.txt")
