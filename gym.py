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
        numbers = entry['numbers']
        X.append(numbers)
        y.append(numbers)  # En este caso, tratamos de predecir los mismos números (simplificación)
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='linear'))  # Usamos activación linear para regresión
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(epochs=1000):
    with open('lottery_data.json', 'r') as f:
        lottery_data = json.load(f)
    
    X, y = preprocess_data(lottery_data)
    
    model = create_model((3,))
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    model.save('lottery_model.h5')

if __name__ == "__main__":
    train_model(epochs=100)
