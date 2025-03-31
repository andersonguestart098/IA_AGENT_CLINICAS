# scheduler.py
import schedule
import time
import subprocess
from datetime import datetime

# Define o horário desejado para execução
HORARIO_EXECUCAO = "09:30"  # Altere aqui para o horário desejado (formato HH:MM)

# Função que roda o scraper
def run_scraper():
    print(f"[{datetime.now()}] Executando web_scraper.py...")
    subprocess.run(["python", "web_scraper.py"], check=True)

# Agenda a tarefa
schedule.every().day.at(HORARIO_EXECUCAO).do(run_scraper)

print(f"[+] Agendador iniciado. Scraping ocorrerá todos os dias às {HORARIO_EXECUCAO}")

# Loop de verificação
while True:
    schedule.run_pending()
    time.sleep(60)
