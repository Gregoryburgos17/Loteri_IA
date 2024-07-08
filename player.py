import matplotlib
matplotlib.use('Agg')  # Usar el backend 'Agg' para evitar problemas con tkinter

import random
import numpy as np
import sqlite3
import tensorflow as tf
import os
import json
import networkx as nx
import matplotlib.pyplot as plt

def simulate_player(model, date, company, game, numbers_per_play, player_id):
    guess = sorted(random.sample(range(1, 101), numbers_per_play))  # Rango de 1 a 100
    prediction_input = np.array([guess])
    if prediction_input.shape[1] != model.input_shape[1]:
        raise ValueError(f"Expected input shape {model.input_shape[1]}, but got {prediction_input.shape[1]}")
    prediction = model.predict(prediction_input)
    accuracy = calculate_accuracy(prediction[0], guess)
    matches, fractional_accuracy = calculate_fractional_accuracy(prediction[0], guess, numbers_per_play)
    result_type = classify_result(accuracy)
    print(f"Simulated player {player_id}: {result_type} with accuracy {accuracy} and {matches}/{numbers_per_play} matches")
    return {
        "player_id": player_id,
        "date": date,
        "company": company,
        "game": game,
        "guess": guess,
        "accuracy": accuracy,
        "fractional_accuracy": fractional_accuracy,
        "prediction": prediction[0].tolist(),
        "result_type": result_type
    }

def calculate_fractional_accuracy(prediction, guess, numbers_per_play):
    matches = sum(1 for p, g in zip(prediction, guess) if p == g)
    fractional_accuracy = matches / numbers_per_play
    return matches, fractional_accuracy

def classify_result(accuracy):
    if accuracy == 1.0:
        return "exitoso"
    elif accuracy > 0.5:
        return "parcial"
    else:
        return "fallo"

def calculate_accuracy(prediction, guess):
    return 1 - np.mean(np.abs(np.array(prediction) - np.array(guess)) / 100)

def save_to_database(results):
    conn = sqlite3.connect('lottery_predictions.db')
    c = conn.cursor()
    for result in results:
        if result['result_type'] == "fallo":
            c.execute('''
                INSERT INTO failures (player_id, date, company, game, guess, accuracy, fractional_accuracy, prediction, result_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result['player_id'], result['date'], result['company'], result['game'], str(result['guess']), result['accuracy'], result['fractional_accuracy'], str(result['prediction']), result['result_type']))
        else:
            c.execute('''
                INSERT INTO predictions (player_id, date, company, game, guess, accuracy, fractional_accuracy, prediction, result_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result['player_id'], result['date'], result['company'], result['game'], str(result['guess']), result['accuracy'], result['fractional_accuracy'], str(result['prediction']), result['result_type']))
    conn.commit()
    conn.close()
    print(f"Saved {len(results)} results to the database.")

def setup_database():
    conn = sqlite3.connect('lottery_predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            date TEXT,
            company TEXT,
            game TEXT,
            guess TEXT,
            accuracy REAL,
            fractional_accuracy REAL,
            prediction TEXT,
            result_type TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            date TEXT,
            company TEXT,
            game TEXT,
            guess TEXT,
            accuracy REAL,
            fractional_accuracy REAL,
            prediction TEXT,
            result_type TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            number INTEGER PRIMARY KEY,
            total_appearances INTEGER,
            first_place_appearances INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("Database setup completed.")

def generate_statistics():
    conn = sqlite3.connect('lottery_predictions.db')
    c = conn.cursor()

    c.execute("DELETE FROM statistics")  # Limpiar tabla de estadísticas

    c.execute("SELECT guess FROM predictions")
    guesses = c.fetchall()

    stats = {}
    for guess in guesses:
        numbers = eval(guess[0])
        for i, number in enumerate(numbers):
            if number not in stats:
                stats[number] = [0, 0]
            stats[number][0] += 1
            if i == 0:  # Primera posición
                stats[number][1] += 1

    for number, (total, first_place) in stats.items():
        c.execute('''
            INSERT INTO statistics (number, total_appearances, first_place_appearances)
            VALUES (?, ?, ?)
            ON CONFLICT(number) DO UPDATE SET
            total_appearances=total_appearances+excluded.total_appearances,
            first_place_appearances=first_place_appearances+excluded.first_place_appearances
        ''', (number, total, first_place))

    conn.commit()
    conn.close()
    print("Statistics generated and saved to the database.")

def run_simulation():
    setup_database()
    
    with open('lottery_data.json', 'r') as f:
        lottery_data = json.load(f)

    all_predictions = []

    model_paths = {
        3: 'lottery_model_3.h5',
        5: 'lottery_model_5.h5',
        6: 'lottery_model_6.h5',
        20: 'lottery_model_20.h5'
    }

    models = {}
    for numbers_per_play, model_path in model_paths.items():
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            models[numbers_per_play] = model
            print(f"Loaded model for {numbers_per_play} numbers.")
        else:
            print(f"Model file {model_path} not found, skipping.")

    simulation_count = 0
    max_simulations = 100  # Limite de simulaciones

    for entry in lottery_data:
        if simulation_count >= max_simulations:
            break
        for player_id in range(4):
            if simulation_count >= max_simulations:
                break
            for game, details in entry['juegos'].items():
                numbers_per_play = len(details['numeros'])
                if numbers_per_play in models:
                    model = models[numbers_per_play]
                    result = simulate_player(model, entry['fecha_solicitud'], entry['compania'], game, numbers_per_play, player_id)
                    all_predictions.append(result)
                    simulation_count += 1
                    if simulation_count >= max_simulations:
                        break

    save_to_database(all_predictions)
    generate_statistics()
    generate_search_graph(all_predictions)

def generate_search_graph(predictions):
    G = nx.DiGraph()
    for i, prediction in enumerate(predictions):
        node_label = f"Player-{prediction['player_id']}\nRed-{i+1}\n{prediction['date']}\n{prediction['company']}\n{prediction['game']}\n{prediction['guess']}\n{prediction['accuracy']:.2f}"
        G.add_node(node_label, label=node_label)
        if i > 0:
            G.add_edge(previous_node, node_label)
        previous_node = node_label

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color="lightblue", font_size=8, font_weight="bold", arrows=True)
    plt.title("Grafo de Búsqueda de Predicciones")
    plt.savefig('search_graph.png')
    plt.close()

if __name__ == "__main__":
    run_simulation()
