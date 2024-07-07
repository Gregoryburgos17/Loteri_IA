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
    df_predictions = pd.read_sql_query("SELECT * FROM predictions", conn)
    df_failures = pd.read_sql_query("SELECT * FROM failures", conn)
    conn.close()

    dates_predictions = pd.to_datetime(df_predictions['date'])
    accuracies = df_predictions['accuracy']
    fallos_predictions = 1 - accuracies  # Calcular los fallos de predicción

    dates_failures = pd.to_datetime(df_failures['date'])
    fallos_failures = df_failures['accuracy']

    # Graficar precisión
    plt.figure(figsize=(10, 6))
    plt.plot(dates_predictions, accuracies, label='Proximidad a la Predicción')
    plt.xlabel('Fecha')
    plt.ylabel('Precisión')
    plt.title('Proximidad de Predicción de la Lotería')
    plt.legend()
    plt.savefig('predictions_accuracy_plot.png')
    plt.close()

    # Graficar fallos
    plt.figure(figsize=(10, 6))
    plt.plot(dates_predictions, fallos_predictions, label='Fallos de la Predicción', color='red')
    plt.xlabel('Fecha')
    plt.ylabel('Fallos')
    plt.title('Fallos de Predicción de la Lotería')
    plt.legend()
    plt.savefig('predictions_fallos_plot.png')
    plt.close()

    # Graficar fallos en la tabla de fallos
    plt.figure(figsize=(10, 6))
    plt.plot(dates_failures, fallos_failures, label='Fallos (Tabla de Fallos)', color='orange')
    plt.xlabel('Fecha')
    plt.ylabel('Fallos')
    plt.title('Fallos Registrados en la Tabla de Fallos')
    plt.legend()
    plt.savefig('failures_fallos_plot.png')
    plt.close()

    # Mostrar intentos de las redes
    successful_attempts = df_predictions[df_predictions['result_type'] == 'exitoso']
    partial_attempts = df_predictions[df_predictions['result_type'] == 'parcial']
    failed_attempts = df_failures

    print("Intentos Exitosos:")
    print(successful_attempts)

    print("Intentos Parciales:")
    print(partial_attempts)

    print("Intentos Fallidos:")
    print(failed_attempts)

def show_statistics():
    conn = sqlite3.connect('lottery_predictions.db')
    df = pd.read_sql_query("SELECT * FROM statistics", conn)
    conn.close()

    df = df.sort_values(by='total_appearances', ascending=False)

    print("Estadísticas de Números:")
    for _, row in df.iterrows():
        print(f"{row['number']} {row['total_appearances']} ({row['first_place_appearances']})")

    # Graficar estadísticas
    plt.figure(figsize=(10, 6))
    plt.bar(df['number'], df['total_appearances'], label='Total de Apariciones')
    plt.xlabel('Número')
    plt.ylabel('Total de Apariciones')
    plt.title('Total de Apariciones de Números en la Lotería')
    plt.legend()
    plt.savefig('number_statistics_plot.png')
    plt.close()

def manual_prediction():
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
    show_statistics()
    manual_prediction()

if __name__ == "__main__":
    main()
