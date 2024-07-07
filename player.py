# player.py
import json
import random
import numpy as np
import sqlite3
import tensorflow as tf

def simulate_player(model, date, play_type):
    guess = sorted(random.sample(range(1, 101), 3))  # Rango de 1 a 100
    prediction = model.predict(np.array([guess]))
    accuracy = calculate_accuracy(prediction[0], guess)
    return {
        "date": date,
        "play_type": play_type,
        "guess": guess,
        "accuracy": accuracy
    }

def calculate_accuracy(prediction, guess):
    # Aquí calculamos la precisión basándonos en la cercanía de los números
    return 1 - np.mean(np.abs(np.array(prediction) - np.array(guess)) / 100)

def save_to_database(results):
    conn = sqlite3.connect('lottery_predictions.db')
    c = conn.cursor()
    for result in results:
        c.execute('''
            INSERT INTO predictions (date, play_type, guess, accuracy)
            VALUES (?, ?, ?, ?)
        ''', (result['date'], result['play_type'], str(result['guess']), result['accuracy']))
    conn.commit()
    conn.close()

def setup_database():
    conn = sqlite3.connect('lottery_predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            play_type TEXT,
            guess TEXT,
            accuracy REAL
        )
    ''')
    conn.commit()
    conn.close()

def run_simulation():
    setup_database()
    
    with open('lottery_data.json', 'r') as f:
        lottery_data = json.load(f)

    model = tf.keras.models.load_model('lottery_model.h5')

    successful_players = []

    for entry in lottery_data:
        result = simulate_player(model, entry['date'], entry['play_type'])
        if result['accuracy'] > 0.5:  # Guardar solo si el acierto es mayor a 50%
            successful_players.append(result)

    save_to_database(successful_players)

if __name__ == "__main__":
    run_simulation()

