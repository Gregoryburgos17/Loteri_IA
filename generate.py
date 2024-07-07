# generate.py
import json
import random
from datetime import datetime, timedelta

def generate_lottery_data(start_date, end_date, types_of_play, numbers_per_play, plays_per_day):
    date_range = (end_date - start_date).days
    lottery_data = []

    for i in range(date_range):
        current_date = start_date + timedelta(days=i)
        for play_type in types_of_play:
            for _ in range(plays_per_day):
                play_numbers = sorted(random.sample(range(1, 101), numbers_per_play))
                lottery_data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "play_type": play_type,
                    "numbers": play_numbers
                })
    
    with open('lottery_data.json', 'w') as f:
        json.dump(lottery_data, f, indent=4)

# Parámetros de generación de datos
if __name__ == "__main__":
    start_date = datetime.strptime('2000-02-17', '%Y-%m-%d')
    end_date = datetime.now()
    types_of_play = ["nacional"]
    numbers_per_play = 3
    plays_per_day = 3

    generate_lottery_data(start_date, end_date, types_of_play, numbers_per_play, plays_per_day)
