# generate.py
import json
import random
from datetime import datetime, timedelta

def generate_lottery_data(start_date, end_date, company_games):
    date_range = (end_date - start_date).days
    lottery_data = []

    for i in range(date_range):
        current_date = start_date + timedelta(days=i)
        for company, games in company_games.items():
            entry = {
                "fecha_solicitud": current_date.strftime("%d-%m-%Y"),
                "compania": company,
                "juegos": {}
            }
            for game, numbers_per_play in games.items():
                entry["juegos"][game] = {
                    "fecha": current_date.strftime("%d-%m-%Y"),
                    "numeros": [str(n) for n in sorted(random.sample(range(1, 101), numbers_per_play))]
                }
            lottery_data.append(entry)

    # Guardar datos de lotería
    with open('lottery_data.json', 'w') as f:
        json.dump(lottery_data, f, indent=4)

    # Generar predicciones futuras
    future_data = []
    for i in range(3):  # Agregar 3 días más a partir de hoy
        future_date = end_date + timedelta(days=i+1)
        for company, games in company_games.items():
            entry = {
                "fecha_solicitud": future_date.strftime("%d-%m-%Y"),
                "compania": company,
                "juegos": {}
            }
            for game, numbers_per_play in games.items():
                entry["juegos"][game] = {
                    "fecha": future_date.strftime("%d-%m-%Y"),
                    "numeros": [str(n) for n in sorted(random.sample(range(1, 101), numbers_per_play))]
                }
            future_data.append(entry)

    # Guardar datos futuros
    with open('future_lottery_data.json', 'w') as f:
        json.dump(future_data, f, indent=4)

# Parámetros de generación de datos
if __name__ == "__main__":
    start_date = datetime.strptime('2000-02-17', '%Y-%m-%d')
    end_date = datetime.now()
    company_games = {
        "Nacional": {
            "Lotería Nacional": 3,
            "Pega 3 Más": 3
        },
        "Leidsa": {
            "Super Kino TV": 20,
            "Quiniela Leidsa": 3,
            "Loto - Loto Más": 6,
            "Super Palé": 2
        }
    }

    generate_lottery_data(start_date, end_date, company_games)
