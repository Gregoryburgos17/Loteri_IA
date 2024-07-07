# main.py
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib

# Configurar backend alternativo para Matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_data():
    os.system('python generate.py')

def train_model():
    os.system('python gym.py')

def simulate_players():
    os.system('python player.py')

def show_results():
    conn = sqlite3.connect('lottery_predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    dates = pd.to_datetime(df['date'])
    accuracies = df['accuracy']
    fallos = 1 - accuracies  # Calcular los fallos de predicción

    # Graficar precisión
    plt.figure(figsize=(10, 6))
    plt.plot(dates, accuracies, label='Proximidad a la Predicción')
    plt.xlabel('Fecha')
    plt.ylabel('Precisión')
    plt.title('Proximidad de Predicción de la Lotería')
    plt.legend()
    plt.savefig('predictions_accuracy_plot.png')
    plt.close()

    # Graficar fallos
    plt.figure(figsize=(10, 6))
    plt.plot(dates, fallos, label='Fallos de la Predicción', color='red')
    plt.xlabel('Fecha')
    plt.ylabel('Fallos')
    plt.title('Fallos de Predicción de la Lotería')
    plt.legend()
    plt.savefig('predictions_fallos_plot.png')
    plt.close()

def manual_prediction():
    # Solicitar entrada manual de tres números
    user_numbers = input("Ingrese tres números separados por comas (ej. 5,12,23): ")
    user_numbers = [int(num) for num in user_numbers.split(",")]

    conn = sqlite3.connect('lottery_predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    dates = pd.to_datetime(df['date'])
    manual_accuracy = []

    for guess in df['guess']:
        guess = eval(guess)  # Convertir cadena de texto a lista
        accuracy = 1 - np.mean(np.abs(np.array(guess) - np.array(user_numbers)) / 100)
        manual_accuracy.append(accuracy)

    # Graficar precisión manual
    plt.figure(figsize=(10, 6))
    plt.plot(dates, manual_accuracy, label='Precisión de Predicción Manual', color='green')
    plt.xlabel('Fecha')
    plt.ylabel('Precisión')
    plt.title('Proximidad de Predicción Manual de la Lotería')
    plt.legend()
    plt.savefig('manual_prediction_accuracy_plot.png')
    plt.close()

def main():
    generate_data()
    train_model()
    simulate_players()
    show_results()
    manual_prediction()

if __name__ == "__main__":
    main()
