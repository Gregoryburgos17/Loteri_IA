# gym.py
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np
import json

def preprocess_data(data):
    X = []
    y = []
    for entry in data:
        for game, details in entry['juegos'].items():
            numbers = [int(n) for n in details['numeros']]
            X.append(numbers)
            y.append(numbers)
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_shape[0], activation='linear'))  # Usamos activación linear para regresión
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(epochs=1000):
    with open('lottery_data.json', 'r') as f:
        lottery_data = json.load(f)
    
    X, y = preprocess_data(lottery_data)
    
    model = create_model((X.shape[1],))
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    model.save('lottery_model.h5')

if __name__ == "__main__":
    train_model(epochs=1000)
